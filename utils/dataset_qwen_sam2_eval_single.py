import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as maskUtils
from .data_processing import get_mask_from_json
from .refer import REFER
from PIL import Image
from qwen_vl_utils import process_vision_info



def collate_fn(
    batch, tokenizer=None,
):
    images_list = []
    masks_list = []
    resize_factor_list=[]

    input_ids_list = []
    attention_mask_list = []
    pixel_values_list = []
    image_grid_thw_list = []

    for (
        images,
        masks,
        resize_factor,
        inputs,
    ) in batch:

        images_list.append(images)
        masks_list.append(masks.float())
        resize_factor_list.append(resize_factor)

        input_ids_list.append(inputs["input_ids"])
        attention_mask_list.append(inputs["attention_mask"])
        pixel_values_list.append(inputs["pixel_values"])
        image_grid_thw_list.append(inputs.get("image_grid_thw", None))

    input_ids = torch.cat(input_ids_list, dim=0)
    attention_masks = torch.cat(attention_mask_list, dim=0)

    pixel_values = torch.cat(pixel_values_list, dim=0)
    image_grid_thw = torch.cat(image_grid_thw_list, dim=0) if None not in image_grid_thw_list else None



    return {
        "images":images_list,#torch.stack(images_list, dim=0),#,   images_list
        "masks_list": masks_list,
        "resize_factor_list":resize_factor_list,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw
    }


class VALDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        val_dataset,
        processor=None,
    ):

        self.processor = processor
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
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
        x = (x - self.pixel_mean) / self.pixel_std
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
        resize_size = 512
        resize_factor = original_width / resize_size, original_height / resize_size


        if not isinstance(masks, torch.Tensor):
            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)

        QUESTION_TEMPLATE = \
            "Find '{Question}'." \
            "The image size is 512*512, locate the most closely matched one." \
            "Directly output the answer with one bbox and two points inside the interested object in <answer> </answer> tags. i.e., <answer>{Answer}</answer> in JSON format." \

        messages = []
        i = 0
        while i < len(sampled_sents):
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
                                                         #Answer="[SEG]")
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
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return (
            image_sam2,
            masks,        # N,640,480
            resize_factor,
            inputs,
        )