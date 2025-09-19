import os
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as maskUtils
from transformers import CLIPImageProcessor

from model.segment_anything.utils.transforms import ResizeLongestSide
from .refer import REFER
from torchvision import transforms
import json
from PIL import Image
from torchvision.transforms.functional import resize, to_pil_image
from qwen_vl_utils import process_vision_info



def collate_fn(
    batch, tokenizer=None, local_rank=-1
):
    image_path_list = []
    images_list = []
    masks_list = []
    label_list = []
    resize_list = []
    resize_factor_list=[]
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []

    input_ids_list = []
    attention_mask_list = []
    pixel_values_list = []
    image_grid_thw_list = []

    for (
        image_path,
        images,
        masks,
        label,
        resize,
        resize_factor,
        inputs,
        sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        resize_factor_list.append(resize_factor)
        sampled_classes_list.extend(sampled_classes)

        cnt += len(sampled_classes)
        offset_list.append(cnt)
        inferences.append(inference)

        # 提取 inputs 中的多模态数据
        input_ids_list.append(inputs["input_ids"])
        attention_mask_list.append(inputs["attention_mask"])
        pixel_values_list.append(inputs["pixel_values"])
        image_grid_thw_list.append(inputs.get("image_grid_thw", None))

    input_ids = torch.cat(input_ids_list, dim=0)
    attention_masks = torch.cat(attention_mask_list, dim=0)

    pixel_values = torch.cat(pixel_values_list, dim=0)
    image_grid_thw = torch.cat(image_grid_thw_list, dim=0) if None not in image_grid_thw_list else None



    return {
        "image_paths": image_path_list,
        "images":images_list,#torch.stack(images_list, dim=0),#,   images_list
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "resize_factor_list":resize_factor_list,
        "offset": torch.LongTensor(offset_list),
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],

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
        image_size=336,
        processor=None,
        transform=ResizeLongestSide(1024)
    ):

        assert isinstance(transform, ResizeLongestSide)
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

        self.transform = transform

        self.image_preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), interpolation=3, antialias=None),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.labels)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
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


        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_sam2 = Image.open(image_path)
        image_sam2 = image_sam2.convert('RGB')

        # preprocess image for qwen
        image_qwen_orgin = Image.open(image_path)
        image_qwen_orgin = image_qwen_orgin.convert('RGB')
        original_width, original_height = image_qwen_orgin.size
        resize_size = 840
        resize_factor = original_width / resize_size, original_height / resize_size

        # preprocess image for sam
        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if not isinstance(masks, torch.Tensor):
            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = False


        # QUESTION_TEMPLATE = "Please find '{Question}'."
        # QUESTION_TEMPLATE = \
        #     "Find '{Question}'." \
        #     "The image size is 256*256, locate the most closely matched one." \
        #     "Output the observing process briefly in <observe> </observe> and final answer in <answer> </answer> tags." \
        #     "Output the final answer with one bbox and two points inside the interested object in JSON format." \
        #     "i.e., <observe> observing process here </observe>" \
        #     "<answer>{Answer}</answer>" \

        QUESTION_TEMPLATE = \
            "Find '{Question}'." \
            "The image size is 256*256, locate the most closely matched one." \
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

        # pdb.set_trace()
        image_inputs, video_inputs = process_vision_info(messages)
        # pdb.set_trace()
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return (
            image_path,
            images,
            masks,        # N,640,480
            labels,       #  640,480
            resize,       # (1024,768)
            resize_factor,
            inputs,
            sampled_sents,
            inference,
        )