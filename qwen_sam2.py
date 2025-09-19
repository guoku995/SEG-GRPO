#'for qwen sam2 val' https://github.com/guoku995/SEG-GRPO.git
import argparse
import os
import sys
import json
import numpy as np
from torch.cuda import OutOfMemoryError

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from utils.utils import AverageMeter, Summary, dict_to_cuda, intersectionAndUnionGPU

import re
from peft import LoraConfig, get_peft_model, PeftModel
from sam2.sam2_image_predictor import SAM2ImagePredictor
import tqdm
import torch.optim as optim
from utils.dataset_qwen_sam2 import collate_fn, VALDataset
from torch.utils.data import DataLoader
from functools import partial
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The default value of the antialias parameter")
#TODO: 返回模型输出隐藏层
#使用sam2,额外添加 hydra iopath
def parse_args(args):
    parser = argparse.ArgumentParser(description="GRPO training")
    parser.add_argument("--image_size", default=224, type=int, help="image size")
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"],
                        help="Precision for training")
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    # parser.add_argument("--train_dataset", default="refcoco+|unc|train", type=str,
    #                     choices=["refcoco|unc|train",
    #                              "refcoco+|unc|train",
    #                              "refcocog|umd|train"])
    parser.add_argument("--val_dataset", default="refcoco|unc|testA", type=str,
                        choices=["refcoco|unc|val", "refcoco|unc|testA", "refcoco|unc|testB",
                                 "refcoco+|unc|val", "refcoco+|unc|testA", "refcoco+|unc|testB",
                                 "refcocog|umd|val", "refcocog|umd|test"])
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")

    parser.add_argument("--log_dir", default="./runs")
    return parser.parse_args(args)


def extract_bbox_points_think(output_text, x_factor, y_factor):
    json_pattern = r'{[^}]+}'  # metch json
    json_match = re.search(json_pattern, output_text)

    content_bbox, points = None, None

    if json_match:
        try:
            data = json.loads(json_match.group(0))

            #bbox
            bbox_key = next((key for key in data.keys() if 'bbox' in key.lower()), None)
            if bbox_key and isinstance(data[bbox_key], list) and len(data[bbox_key]) == 4:
                try:
                    content_bbox = data[bbox_key]
                    content_bbox = [round(int(content_bbox[0]) * x_factor), round(int(content_bbox[1]) * y_factor),
                                    round(int(content_bbox[2]) * x_factor), round(int(content_bbox[3]) * y_factor)]
                except (ValueError, TypeError, IndexError):
                    content_bbox = None

            # points
            points_keys = [key for key in data.keys() if 'points' in key.lower()]
            if len(points_keys) >= 2:
                try:
                    point1 = data[points_keys[0]]
                    point2 = data[points_keys[1]]
                    if isinstance(point1, list) and len(point1) == 2 and \
                            isinstance(point2, list) and len(point2) == 2:
                        point1 = [round(int(point1[0]) * x_factor), round(int(point1[1]) * y_factor)]
                        point2 = [round(int(point2[0]) * x_factor), round(int(point2[1]) * y_factor)]
                        points = [point1, point2]
                except (ValueError, TypeError, IndexError):
                    points = None

        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始文本: {json_match.group(0)}")
            content_bbox, points = None, None


    default_bbox = [0, 0, 0, 0]
    default_points = [[0, 0], [0, 0]]

    content_bbox = content_bbox if content_bbox is not None else default_bbox
    points = points if points is not None else default_points

    return content_bbox, points
def main(args):
    args = parse_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained("outputs/checkpoint-16000",
                                              use_fast=False,
                                              padding_side="left")
    tokenizer = processor.tokenizer

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "evfsamconfig/qwen7B",
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    for param in reasoning_model.parameters():
        param.requires_grad = False
    reasoning_model = PeftModel.from_pretrained(reasoning_model, "outputs/checkpoint-16000")

    segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    val_dataset = VALDataset(
        args.dataset_dir,
        args.val_dataset,
        args.image_size,
        processor
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )

    for input_dict in tqdm.tqdm(val_loader):
        try:
            torch.cuda.empty_cache()
            input_dict = dict_to_cuda(input_dict)

            image = input_dict["images"][0]
            input_ids =input_dict["input_ids"]
            attention_mask =input_dict["attention_mask"]
            pixel_values =input_dict["pixel_values"]
            image_grid_thw =input_dict["image_grid_thw"]

            generated_ids = reasoning_model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                pixel_values = pixel_values,
                image_grid_thw = image_grid_thw,
                use_cache=True, max_new_tokens=400, do_sample=False#, temperature = 0.8,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]

            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            x_factor, y_factor =input_dict["resize_factor_list"][0]

            pred_masks=[]
            for batch_idx in range(input_ids.shape[0]):
                bbox, points, think = extract_bbox_points_think(output_text[batch_idx], x_factor, y_factor)

                default_bbox = [0, 0, 0, 0]
                default_points = [[0, 0], [0, 0]]

                if bbox is None:
                    bbox = default_bbox
                if points is None :
                    points = default_points

                # mask
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch_dtype):
                    segmentation_model.set_image(image)
                    masks, scores, _ = segmentation_model.predict(
                        point_coords=points,
                        point_labels=[1, 1],
                        box=bbox
                    )
                    sorted_ind = torch.argsort(scores, dim=-1, descending=True)
                    masks = masks[sorted_ind]
                    pred_masks.append(masks[0])
            pred_masks = torch.stack(pred_masks,dim=0)
            pred_masks = (pred_masks > 0).int()
            output_tensor = pred_masks
            output_tensor = output_tensor.to("cuda")
            masks_tensor = input_dict["masks_list"][0].int()  #input_dict["masks_list"] is list, [0] for tensor
            masks_tensor = masks_tensor.to("cuda")
            #  Intersection and union
            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i, output_i in zip(masks_tensor, output_tensor):
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-10)
                acc_iou[union_i == 0] += 1.0  # no-object target

            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / masks_tensor.shape[0]

            intersection_meter.update(intersection)
            union_meter.update(union)
            acc_iou_meter.update(acc_iou, n=masks_tensor.shape[0])
            print("acciou{}".format(acc_iou))
        except OutOfMemoryError:
            print(f"out of memory")
            torch.cuda.empty_cache()
            continue
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    print("giou{:.3f}_ciou{:.3f}".format(giou, ciou))
    print("样本数:{}".format(union_meter.sum))


if __name__ == "__main__":
    main(sys.argv[1:])
