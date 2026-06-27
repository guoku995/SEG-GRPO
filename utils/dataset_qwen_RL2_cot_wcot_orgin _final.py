import glob
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as maskUtils

from .refer import REFER
from PIL import Image
from qwen_vl_utils import process_vision_info
from .data_processing import get_mask_from_json
# sam2固定点

class TrainDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        train_dataset,
        processor=None,
    ):

        self.processor = processor
        self.base_image_dir = base_image_dir
        splits = train_dataset.split("|")
        if len(splits) == 3:
            ds, splitBy, split = splits
            base_image_dir = os.path.join(base_image_dir, "refer_seg")
            refer_api = REFER(base_image_dir, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"
        elif len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])
            sampled_sents = sents

            masks = []
            for i, ann_id in enumerate(ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = maskUtils.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = maskUtils.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

            masks = [mask_json]

        image_sam2 = Image.open(image_path)
        image_sam2 = image_sam2.convert('RGB')

        # preprocess image for qwen
        image_qwen_orgin = Image.open(image_path)
        image_qwen_orgin = image_qwen_orgin.convert('RGB')
        original_width, original_height = image_qwen_orgin.size
        resize_size = 336
        # 缩放比例用于 在原图像中找点
        resize_factor = original_width / resize_size, original_height / resize_size

        if not isinstance(masks, torch.Tensor):
            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)

        # QUESTION_TEMPLATE = \
        #     "Find '{Question}'." \
        #     "The image size is 256*256. Find the target's position in the image through reasoning." \
        #     "Output the thinking process briefly in <think> </think> and final answer in <answer> </answer> tags." \
        #     "Output the final answer with one bbox and two points inside the interested object in JSON format." \
        #     "i.e., <think> thinking process here </think>" \
        #     "<answer>{Answer}</answer>" \

        QUESTION_TEMPLATE = \
            "Find '{Question}'." \
            "The image size is 336*336, locate the most closely matched one." \
            "Directly output the answer with one bbox and two points inside the interested object in <answer> </answer> tags. i.e., <answer>{Answer}</answer> in JSON format." \


        messages = []
        i = 0
        cut_index=1
        while i < cut_index:
            message = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_qwen_orgin.resize((resize_size, resize_size), Image.BILINEAR)
                    },
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(Question=sampled_sents[i].lower().strip("."),
                                                         Answer='{"bbox": [10,100,200,210], "points_1": [30,110], "points_2": [35,180]}')
                    }
                ],
            },
            ]
            messages.append(message)
            i += 1

        text = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            #videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return {
            "image_path": image_path,
            "images": image_sam2,#image.to(dtype=torch.float16), # image_sam2
            "masks": masks[:cut_index].float(), #1 H W  （必须要batch为1,refcoco要截断，reasonseg不用截断）
            "resize_factor":resize_factor,
            "msg":messages,  #msg包含图片和文本信息，只传这个就行
            "input_ids": inputs["input_ids"],#.to(dtype=torch.int32),
            "attention_mask": inputs["attention_mask"],#.to(dtype=torch.int32), 必须啊int64 ？
            "pixel_values": inputs["pixel_values"].to(dtype=torch.bfloat16),
            "image_grid_thw": inputs.get("image_grid_thw", None),#.to(dtype=torch.int32),

            "sampled_classes": sampled_sents[:cut_index],
        }
