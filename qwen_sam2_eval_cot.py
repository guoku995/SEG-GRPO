#'for qwen sam2 val' https://github.com/guoku995/SEG-GRPO.git
import argparse
import os
import sys
import json

import numpy as np
from torch.cuda import OutOfMemoryError

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor,AutoModelForCausalLM,Qwen3VLForConditionalGeneration
from utils.utils import AverageMeter, Summary, dict_to_cuda, intersectionAndUnionGPU

import re
from peft import PeftModel
from sam2.sam2_image_predictor import SAM2ImagePredictor
#from sam2.build_sam import build_sam2
import tqdm
from utils.dataset_qwen_sam2_eval_cot import collate_fn, VALDataset
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The default value of the antialias parameter")
from qwen_sam2_eval import extract_sam_prompt_raw, scale_sam_prompt

#使用sam2,需要额外添加依赖包 hydra iopath
def parse_args(args):
    parser = argparse.ArgumentParser(description="GRPO evaluation")
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"],
                        help="Precision for training")
    parser.add_argument("--dataset_dir", default="./dataset", type=str)

    parser.add_argument("--val_dataset", default="refcoco|unc|testA", type=str,
                        choices=["refcoco|unc|val", "refcoco|unc|testA", "refcoco|unc|testB",
                                 "refcoco+|unc|val", "refcoco+|unc|testA", "refcoco+|unc|testB",
                                 "refcocog|umd|val", "refcocog|umd|test", "ReasonSeg|test"])

    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")

    parser.add_argument("--cot_reasoning_level", default="all", type=str,
                        choices=["light", "medium", "heavy", "all"],
                        help="Evaluate one reasoning level or all three levels")
    parser.add_argument("--max_new_tokens", default=220, type=int,
                        help="Maximum new tokens for Qwen generation")
    parser.add_argument("--log_dir", default="./runs")
    return parser.parse_args(args)


def extract_bbox_points_think(output_text, x_factor, y_factor):
    json_pattern = r'{[^}]+}'  # metch json
    json_match = re.search(json_pattern, output_text)

    content_bbox, points = None, None

    if json_match:
        try:
            #data = json.loads(json_match.group(0))

            json_str = json_match.group(0)
            #修复不完整的浮点数（小数点后没有数字的情况）
            json_str = re.sub(r'(\d+)\.(?=\D)', r'\1.0', json_str)
            data = json.loads(json_str)

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

    return content_bbox, points

def main(args):
    args = parse_args(args)
    project_root = Path(__file__).resolve().parent
    qwen_model_path = project_root / "checkpoints/qwen2.5_3B"
    processor = AutoProcessor.from_pretrained(str(qwen_model_path),
                                              #use_fast=False,
                                              padding_side="left",
                                              #local_files_only=True
                                              )
    tokenizer = processor.tokenizer

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(qwen_model_path),
        dtype=torch_dtype,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        #local_files_only=True
    )

    for param in reasoning_model.parameters():
        param.requires_grad = False
    reasoning_model = PeftModel.from_pretrained(reasoning_model, "outputs/checkpoint-14500-cvprV2")

    segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)

    def evaluate_level(cot_reasoning_level):
        print(f"===== Evaluating CoT reasoning level: {cot_reasoning_level} =====")
        intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
        acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

        val_dataset = VALDataset(
            args.dataset_dir,
            args.val_dataset,
            processor,
            cot_reasoning_level=cot_reasoning_level,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1, #只能是1
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
                x_factor, y_factor = input_dict["resize_factor_list"][0]

                generated_ids = reasoning_model.generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    pixel_values = pixel_values,
                    image_grid_thw = image_grid_thw,
                    use_cache=True, max_new_tokens=args.max_new_tokens, do_sample=False#, temperature = 0.8,
                )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
                ]

                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                pred_masks=[]
                for batch_idx in range(input_ids.shape[0]):
                    sam_prompt = extract_sam_prompt_raw(output_text[batch_idx])

                    if sam_prompt is None:
                        continue
                    bbox_prompt, point_coords_prompt, point_labels = sam_prompt
                    bbox, point_coords, point_labels = scale_sam_prompt(
                        bbox_prompt,
                        point_coords_prompt,
                        point_labels,
                        x_factor,
                        y_factor,
                    )

                    # mask
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch_dtype):
                        segmentation_model.set_image(image)
                        masks, scores, _ = segmentation_model.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            box=bbox
                        )
                        # sorted_ind = torch.argsort(scores, dim=-1, descending=True)
                        # mask = masks[sorted_ind][0]
                        sorted_ind = np.argsort(scores)[::-1]  # masks和scores都是numpy
                        mask = masks[sorted_ind][0]
                        pred_mask_tensor = torch.from_numpy(mask)
                        pred_masks.append(pred_mask_tensor)

                if len(pred_masks) == 0:
                    print("当前batch没有有效样本，跳过")
                    continue

                pred_masks = torch.stack(pred_masks,dim=0)
                pred_masks = (pred_masks > 0).int()
                output_tensor = pred_masks
                output_tensor = output_tensor.to("cuda")
                masks_tensor = input_dict["masks_list"][0].int()  #input_dict["masks_list"] is list, [0] for tensor
                masks_tensor = masks_tensor.to("cuda")
                if pred_masks.shape[0] != masks_tensor.shape[0]:
                    print(f"警告：预测mask数量({pred_masks.shape[0]}) != GT mask数量({masks_tensor.shape[0]})，跳过该样本")
                    continue
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
        print("[{}] giou{:.3f}_ciou{:.3f}".format(cot_reasoning_level, giou, ciou))
        print("[{}] 样本数:{}".format(cot_reasoning_level, union_meter.sum))
        return giou, ciou

    levels = ["light", "medium", "heavy"] if args.cot_reasoning_level == "all" else [args.cot_reasoning_level]
    results = {}
    for cot_reasoning_level in levels:
        results[cot_reasoning_level] = evaluate_level(cot_reasoning_level)
    if len(results) > 1:
        print("===== CoT level summary =====")
        for cot_reasoning_level, (giou, ciou) in results.items():
            print("{}: giou{:.3f}_ciou{:.3f}".format(cot_reasoning_level, giou, ciou))


if __name__ == "__main__":
    main(sys.argv[1:])
