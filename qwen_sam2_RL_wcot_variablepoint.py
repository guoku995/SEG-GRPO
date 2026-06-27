import argparse
import os
import sys
import json
import warnings

import numpy as np
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen3VLForConditionalGeneration
from utils.dataset_qwen_RL2 import TrainDataset
from utils.utils import intersectionAndUnionGPU
from trl import GRPOConfig, GRPOTrainer
import re
from peft import LoraConfig, get_peft_model, PeftModel
from sam2.sam2_image_predictor import SAM2ImagePredictor
warnings.filterwarnings("ignore", category=UserWarning, message="The default value of the antialias parameter")
from pathlib import Path


class CompatGRPOTrainer(GRPOTrainer):
    """Compatibility shim for TRL 0.16.x with Transformers 4.57+."""

    def _get_train_sampler(self, train_dataset=None):
        return super()._get_train_sampler()


#使用sam2,额外添加 hydra iopath
def parse_args(args):
    parser = argparse.ArgumentParser(description="GRPO training")
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"],
                        help="Precision for training")
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--train_dataset", default="refcoco|unc|train", type=str,
                        choices=["refcoco|unc|train",
                                 "refcoco+|unc|train",
                                 "refcocog|umd|train",
                                 "ReasonSeg|train"])
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    #parser.add_argument("--segmentation_model_path", type=str, default="checkpoints/sam2")
    parser.add_argument("--output_dir", default="outputs", type=str, help="Training output directory")
    parser.add_argument("--reprompt_threshold", default=1, type=float,
                        help="IoU reward threshold used to stop training-time re-prompting")
    parser.add_argument("--max_reprompt_rounds", default=1, type=int,
                        help="Maximum number of corrective re-prompt rounds after the initial generation")
    parser.add_argument("--revision_reward_weight", default=1.0, type=float,
                        help="Weight for the revision improvement reward shaping term")
    parser.add_argument("--revision_advantage_weight", default=1.0, type=float,
                        help="Weight for scaling GRPO advantages by revision improvement")
    parser.add_argument("--reprompt_overlay_alpha", default=0.45, type=float,
                        help="Opacity of the red previous-mask overlay used for training-time re-prompting")
    return parser.parse_args(args)

def main(args):

    args = parse_args(args)
    project_root = Path(__file__).resolve().parent
    qwen_model_path = project_root / "checkpoints/qwen2.5_3B"
    # ========== 关键步骤1：定义统一的目录（解决变量引用顺序问题） ==========
    output_dir = Path(args.output_dir)  # 从命令行参数获取，或用默认值
    tb_log_dir = output_dir / "tb_logs"  # TensorBoard 日志目录
    tb_log_dir.mkdir(parents=True, exist_ok=True)  # 自动创建目录（父目录不存在也创建）
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_config = {
        "train_dataset": args.train_dataset,
        "precision": args.precision,
        "segmentation_model_path": args.segmentation_model_path,
        "reprompt_threshold": args.reprompt_threshold,
        "max_reprompt_rounds": args.max_reprompt_rounds,
        "revision_reward_weight": args.revision_reward_weight,
        "revision_advantage_weight": args.revision_advantage_weight,
        "reprompt_overlay_alpha": args.reprompt_overlay_alpha,
        "invalid_reward": -1.0,
        "format_valid_reward": 1.0,
        "max_positive_points": 1,
        "max_negative_points": 1,
        "max_extra_positive_points": 2,
        "max_extra_negative_points": 2,
        "max_extra_points": 4,
        "min_positive_points": 1,
        "min_negative_points": 1,
        "extra_improvement_weight": 2.0,
        "point_error_map_weight": 0.5,
        "point_adaptive_count_weight": 0.8,
        "harmful_point_weight": 0.2,
        "point_redundancy_weight": 0.01,
        "point_count_cost": 0.01,
        "positive_error_ratio_threshold": 0.03,
        "negative_error_ratio_threshold": 0.03,
        "point_error_ratio_step": 0.15,
        "missing_positive_penalty": 0.5,
        "missing_negative_penalty": 0.7,
        "unneeded_positive_penalty": 0.15,
        "unneeded_negative_penalty": 0.15,
        "reward_weights": [0.5, 3.0, 0.5, 1.0],
    }
    with open(output_dir / "reprompt_experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, ensure_ascii=False, indent=2)
    print(f"Re-prompt experiment config: {experiment_config}")

    # ========== 关键步骤2：配置 TensorBoard 日志（可选，让自定义指标也写入TensorBoard） ==========
    from torch.utils.tensorboard import SummaryWriter
    # 创建 TensorBoard 写入器（记录自定义指标，如 IoU、格式奖励）
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    global_step = 0  # 记录当前训练步数（用于TensorBoard标度）

    processor = AutoProcessor.from_pretrained(str(qwen_model_path),
                                              use_fast=False,
                                              padding_side="left",
                                              local_files_only=True)

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(qwen_model_path),
        dtype=torch_dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    #reasoning_model = PeftModel.from_pretrained(reasoning_model, "outputs/checkpoint-16994-1")
    segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)

    # 先提取<answer>之后的内容，再提取</answer>之前的内容
    def extract_xml_answer(text: str) -> str:
        answer = text.split("<answer>")[-1].split("</answer>")[0]
        answer = answer.replace("\n", "").strip()
        answer = re.sub(r'(\d+)\.(?=\D)', r'\1.0', answer)

        return answer

    def _is_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    INVALID_REWARD = -1.0
    FORMAT_VALID_REWARD = 1.0
    MAX_POSITIVE_POINTS = 1
    MAX_NEGATIVE_POINTS = 1
    MAX_EXTRA_POSITIVE_POINTS = 2
    MAX_EXTRA_NEGATIVE_POINTS = 2
    MAX_EXTRA_POINTS = MAX_EXTRA_POSITIVE_POINTS + MAX_EXTRA_NEGATIVE_POINTS
    MIN_POSITIVE_POINTS = 1
    MIN_NEGATIVE_POINTS = 1
    EXTRA_IMPROVEMENT_WEIGHT = 2.0
    POINT_ERROR_MAP_WEIGHT = 0.5
    POINT_ADAPTIVE_COUNT_WEIGHT = 0.8
    HARMFUL_POINT_WEIGHT = 0.2
    POINT_REDUNDANCY_WEIGHT = 0.01
    POINT_COUNT_COST = 0.01
    POSITIVE_ERROR_RATIO_THRESHOLD = 0.03
    NEGATIVE_ERROR_RATIO_THRESHOLD = 0.03
    POINT_ERROR_RATIO_STEP = 0.15
    MISSING_POSITIVE_PENALTY = 0.5
    MISSING_NEGATIVE_PENALTY = 0.7
    UNNEEDED_POSITIVE_PENALTY = 0.15
    UNNEEDED_NEGATIVE_PENALTY = 0.15

    def _to_float(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _normalize_bbox(value):
        if not (isinstance(value, list) and len(value) == 4):
            return None
        if not all(_is_number(coord) for coord in value):
            return None
        return [round(float(coord)) for coord in value]

    def _bbox_in_prompt_bounds(bbox):
        x1, y1, x2, y2 = bbox
        return 0 <= x1 < x2 <= 256 and 0 <= y1 < y2 <= 256

    def _normalize_point(value):
        if not (isinstance(value, list) and len(value) == 2):
            return None
        if not all(_is_number(coord) for coord in value):
            return None
        return [round(float(value[0])), round(float(value[1]))]

    def _point_in_prompt_bounds(point):
        x, y = point
        return 0 <= x < 256 and 0 <= y < 256

    def _normalize_points(value, require_point_list=False):
        if value is None:
            return []
        if require_point_list:
            if not isinstance(value, list):
                return None
            points = []
            for item in value:
                point = _normalize_point(item)
                if point is None:
                    return None
                points.append(point)
            return points
        single_point = _normalize_point(value)
        if single_point is not None:
            return [single_point]
        if not isinstance(value, list):
            return None
        points = []
        for item in value:
            point = _normalize_point(item)
            if point is None:
                return None
            points.append(point)
        return points

    def _find_point_key(data, tokens):
        for key in data.keys():
            lower_key = key.lower()
            if "extra" in lower_key:
                continue
            if "point" in lower_key and any(token in lower_key for token in tokens):
                return key
        return None

    def _find_extra_point_key(data, tokens):
        for key in data.keys():
            lower_key = key.lower()
            if (
                "point" in lower_key
                and "extra" in lower_key
                and any(token in lower_key for token in tokens)
            ):
                return key
        return None

    def _extract_sam_prompt(data, require_new_schema=False):
        if not isinstance(data, dict):
            return None

        bbox_key = next((key for key in data.keys() if "bbox" in key.lower()), None)
        if bbox_key is None:
            return None
        bbox = _normalize_bbox(data.get(bbox_key))
        if bbox is None:
            return None
        if require_new_schema and not _bbox_in_prompt_bounds(bbox):
            return None

        positive_key = _find_point_key(data, ("positive", "foreground", "include", "add"))
        negative_key = _find_point_key(data, ("negative", "background", "exclude", "remove"))
        extra_positive_key = _find_extra_point_key(data, ("positive", "foreground", "include", "add"))
        extra_negative_key = _find_extra_point_key(data, ("negative", "background", "exclude", "remove"))
        has_extra_schema = extra_positive_key is not None or extra_negative_key is not None
        has_new_schema = positive_key is not None or negative_key is not None
        if require_new_schema and (positive_key is None or negative_key is None):
            return None

        if has_new_schema:
            positive_points = (
                _normalize_points(data.get(positive_key), require_point_list=require_new_schema)
                if positive_key is not None
                else []
            )
            negative_points = (
                _normalize_points(data.get(negative_key), require_point_list=require_new_schema)
                if negative_key is not None
                else []
            )
            extra_positive_points = (
                _normalize_points(data.get(extra_positive_key), require_point_list=True)
                if extra_positive_key is not None
                else []
            )
            extra_negative_points = (
                _normalize_points(data.get(extra_negative_key), require_point_list=True)
                if extra_negative_key is not None
                else []
            )
            if extra_positive_points is None or extra_negative_points is None:
                return None
        else:
            if require_new_schema:
                return None
            positive_points = []
            for key in data.keys():
                if "point" not in key.lower():
                    continue
                points = _normalize_points(data.get(key))
                if points is None:
                    return None
                positive_points.extend(points)
            negative_points = []
            extra_positive_points = []
            extra_negative_points = []
            has_extra_schema = False

        if (
            positive_points is None
            or negative_points is None
            or len(positive_points) < MIN_POSITIVE_POINTS
            or len(negative_points) < MIN_NEGATIVE_POINTS
        ):
            return None
        if require_new_schema and (
            len(positive_points) > MAX_POSITIVE_POINTS
            or len(negative_points) > MAX_NEGATIVE_POINTS
            or len(extra_positive_points) > MAX_EXTRA_POSITIVE_POINTS
            or len(extra_negative_points) > MAX_EXTRA_NEGATIVE_POINTS
        ):
            return None

        extra_points = extra_positive_points + extra_negative_points
        extra_labels = [1] * len(extra_positive_points) + [0] * len(extra_negative_points)
        all_points = positive_points + negative_points + extra_points
        if require_new_schema and not all(_point_in_prompt_bounds(point) for point in all_points):
            return None

        base_point_coords = positive_points + negative_points
        base_point_labels = [1] * len(positive_points) + [0] * len(negative_points)
        return bbox, base_point_coords, base_point_labels, extra_points, extra_labels, has_extra_schema

    def _resize_factor_at(resize_factors, index):
        factor = resize_factors[index] if isinstance(resize_factors, list) else resize_factors
        if isinstance(factor, torch.Tensor):
            factor = factor.detach().cpu().tolist()
        return _to_float(factor[0]), _to_float(factor[1])

    def _scale_sam_prompt(prompt, resize_factors, index):
        bbox, point_coords, point_labels, extra_points, extra_labels, has_extra_schema = prompt
        rfx, rfy = _resize_factor_at(resize_factors, index)
        scaled_bbox = [
            round(float(bbox[0]) * rfx),
            round(float(bbox[1]) * rfy),
            round(float(bbox[2]) * rfx),
            round(float(bbox[3]) * rfy),
        ]
        scaled_points = [
            [round(float(point[0]) * rfx), round(float(point[1]) * rfy)]
            for point in point_coords
        ]
        scaled_extra_points = [
            [round(float(point[0]) * rfx), round(float(point[1]) * rfy)]
            for point in extra_points
        ]
        return scaled_bbox, scaled_points, point_labels, scaled_extra_points, extra_labels, has_extra_schema

    def _json_prompt_from_completion(completion):
        try:
            data = json.loads(extract_xml_answer(completion))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def _compute_binary_iou(pred_binary, gt_mask_tensor):
        intersection, union, _ = intersectionAndUnionGPU(
            pred_binary.contiguous().clone(),
            gt_mask_tensor.contiguous(),
            2,
            ignore_index=255
        )
        if union[1] > 0:
            return (intersection[1] / (union[1] + 1e-10)).item()
        return 1.0

    def _best_binary_mask(masks, scores, device):
        masks = torch.from_numpy(masks).to(device=device, dtype=torch_dtype)
        scores = torch.from_numpy(scores).to(device=device, dtype=torch_dtype)
        sorted_ind = torch.argsort(scores, dim=-1, descending=True)
        return (masks[sorted_ind][0] > 0).int().unsqueeze(0)

    def _prepare_gt_mask(gt_mask, device):
        gt_mask_tensor = gt_mask if isinstance(gt_mask, torch.Tensor) else torch.as_tensor(gt_mask)
        gt_mask_tensor = gt_mask_tensor.to(device=device).int()
        if gt_mask_tensor.dim() == 2:
            gt_mask_tensor = gt_mask_tensor.unsqueeze(0)
        return gt_mask_tensor

    def _desired_error_points(error_ratio, threshold, max_points):
        if error_ratio < threshold:
            return 0
        return min(max_points, max(1, int(np.ceil(error_ratio / POINT_ERROR_RATIO_STEP))))

    def _point_error_map_reward(points, point_labels, gt_mask_tensor, box_binary):
        gt_2d = gt_mask_tensor.squeeze(0)
        box_2d = box_binary.squeeze(0).bool()
        valid = gt_2d != 255
        gt_bool = (gt_2d > 0) & valid
        box_bool = box_2d & valid
        false_positive = box_bool & (~gt_bool) & valid
        false_negative = (~box_bool) & gt_bool & valid
        height, width = gt_2d.shape[-2:]
        box_area = max(1.0, float(box_bool.sum().item()))
        target_area = max(1.0, float(gt_bool.sum().item()))
        false_positive_ratio = float(false_positive.sum().item()) / box_area
        false_negative_ratio = float(false_negative.sum().item()) / target_area
        desired_positive_points = _desired_error_points(
            false_negative_ratio,
            POSITIVE_ERROR_RATIO_THRESHOLD,
            MAX_EXTRA_POSITIVE_POINTS,
        )
        desired_negative_points = _desired_error_points(
            false_positive_ratio,
            NEGATIVE_ERROR_RATIO_THRESHOLD,
            MAX_EXTRA_NEGATIVE_POINTS,
        )

        point_rewards = []
        harmful_points = 0
        helpful_positive_points = 0
        helpful_negative_points = 0
        for point, label in zip(points, point_labels):
            x, y = round(float(point[0])), round(float(point[1]))
            if x < 0 or x >= width or y < 0 or y >= height or not bool(valid[y, x].item()):
                point_rewards.append(-1.0)
                harmful_points += 1
                continue

            in_gt = bool(gt_bool[y, x].item())
            if label == 1:
                if not in_gt:
                    point_rewards.append(-0.5)
                    harmful_points += 1
                elif bool(false_negative[y, x].item()):
                    point_rewards.append(1.0)
                    helpful_positive_points += 1
                else:
                    point_rewards.append(0.1)
            else:
                if in_gt:
                    point_rewards.append(-0.5)
                    harmful_points += 1
                elif bool(false_positive[y, x].item()):
                    point_rewards.append(1.0)
                    helpful_negative_points += 1
                else:
                    point_rewards.append(0.0)

        if not point_rewards:
            point_error_map_reward = 0.0
            harmful_point_rate = 0.0
        else:
            point_error_map_reward = float(np.mean(point_rewards))
            harmful_point_rate = harmful_points / max(1, len(point_rewards))

        return {
            "point_error_map": point_error_map_reward,
            "harmful_point_rate": harmful_point_rate,
            "false_positive_ratio": false_positive_ratio,
            "false_negative_ratio": false_negative_ratio,
            "desired_positive_points": desired_positive_points,
            "desired_negative_points": desired_negative_points,
            "helpful_positive_points": helpful_positive_points,
            "helpful_negative_points": helpful_negative_points,
        }

    def _adaptive_point_count_reward(result):
        if not result.get("has_extra_schema", False):
            return 0.0

        desired_extra_positive = result["desired_positive_points"]
        num_extra_positive = result["num_positive_points"]

        if desired_extra_positive > 0:
            positive_reward = min(result["helpful_positive_points"], desired_extra_positive) / desired_extra_positive
            missing_extra_positive = max(0, desired_extra_positive - num_extra_positive)
            if result["helpful_positive_points"] == 0:
                positive_reward -= MISSING_POSITIVE_PENALTY
            if missing_extra_positive > 0:
                positive_reward -= MISSING_POSITIVE_PENALTY * missing_extra_positive / desired_extra_positive
            if num_extra_positive > desired_extra_positive:
                positive_reward -= UNNEEDED_POSITIVE_PENALTY * (num_extra_positive - desired_extra_positive)
        else:
            positive_reward = -UNNEEDED_POSITIVE_PENALTY * num_extra_positive

        desired_negative = result["desired_negative_points"]
        if desired_negative > 0:
            negative_reward = min(result["helpful_negative_points"], desired_negative) / desired_negative
            missing_negative = max(0, desired_negative - result["num_negative_points"])
            if result["helpful_negative_points"] == 0:
                negative_reward -= MISSING_NEGATIVE_PENALTY
            if missing_negative > 0:
                negative_reward -= MISSING_NEGATIVE_PENALTY * missing_negative / desired_negative
            if result["num_negative_points"] > desired_negative:
                negative_reward -= UNNEEDED_NEGATIVE_PENALTY * (result["num_negative_points"] - desired_negative)
        else:
            negative_reward = -UNNEEDED_NEGATIVE_PENALTY * result["num_negative_points"]

        return positive_reward + negative_reward

    def _near_duplicate_penalty(points):
        duplicate_pairs = 0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dx = float(points[i][0]) - float(points[j][0])
                dy = float(points[i][1]) - float(points[j][1])
                if (dx * dx + dy * dy) ** 0.5 < 8.0:
                    duplicate_pairs += 1
        return duplicate_pairs / max(1, len(points))

    reward_eval_cache = {"key": None, "results": None}

    def _evaluate_sam_prompts(completions, **kwargs):
        batch_images = kwargs.get("images")
        batch_masks = kwargs.get("masks")
        resize_factor = kwargs.get("resize_factor")
        image_paths = tuple(str(path) for path in kwargs.get("image_path", []))
        sampled_classes = tuple(str(item) for item in kwargs.get("sampled_classes", []))
        cache_key = (image_paths, sampled_classes, tuple(completions))
        if reward_eval_cache["key"] == cache_key:
            return reward_eval_cache["results"]

        results = [{"valid": False} for _ in completions]
        if not batch_images or not batch_masks:
            reward_eval_cache["key"] = cache_key
            reward_eval_cache["results"] = results
            return results

        batch = len(batch_images)
        completions_per_image = 1
        if batch > 0 and len(completions) % batch == 0:
            completions_per_image = max(1, len(completions) // batch)

        for batch_idx, image in enumerate(batch_images):
            start = batch_idx * completions_per_image
            end = min(start + completions_per_image, len(completions))
            if start >= len(completions):
                break

            gt_mask_tensor = None
            image_is_set = False
            box_mask_cache = {}
            for completion_idx in range(start, end):
                prompt = _extract_sam_prompt(
                    _json_prompt_from_completion(completions[completion_idx]),
                    require_new_schema=True,
                )
                if prompt is None:
                    continue

                content_bbox, base_points, base_labels, extra_points, extra_labels, has_extra_schema = _scale_sam_prompt(
                    prompt,
                    resize_factor,
                    batch_idx,
                )
                try:
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch_dtype):
                        if not image_is_set:
                            segmentation_model.set_image(image)
                            image_is_set = True
                        box_key = tuple(content_bbox)
                        if box_key not in box_mask_cache:
                            box_masks, box_scores, _ = segmentation_model.predict(
                                box=content_bbox,
                                multimask_output=False,
                            )
                            box_mask_cache[box_key] = _best_binary_mask(box_masks, box_scores, "cuda")
                        box_binary = box_mask_cache[box_key]
                        base_masks, base_scores, _ = segmentation_model.predict(
                            point_coords=base_points,
                            point_labels=base_labels,
                            box=content_bbox,
                            multimask_output=False,
                        )
                        base_binary = _best_binary_mask(base_masks, base_scores, "cuda")
                        if extra_points:
                            final_points = base_points + extra_points
                            final_labels = base_labels + extra_labels
                            extra_masks, extra_scores, _ = segmentation_model.predict(
                                point_coords=final_points,
                                point_labels=final_labels,
                                box=content_bbox,
                                multimask_output=False,
                            )
                            extra_binary = _best_binary_mask(extra_masks, extra_scores, "cuda")
                        else:
                            final_points = base_points
                            final_labels = base_labels
                            extra_binary = base_binary
                    if gt_mask_tensor is None:
                        gt_mask_tensor = _prepare_gt_mask(batch_masks[batch_idx], extra_binary.device)

                    box_iou = _compute_binary_iou(box_binary, gt_mask_tensor)
                    base_iou = _compute_binary_iou(base_binary, gt_mask_tensor)
                    extra_iou = _compute_binary_iou(extra_binary, gt_mask_tensor)
                    extra_iou_improvement = extra_iou - base_iou
                    point_stats = _point_error_map_reward(
                        extra_points,
                        extra_labels,
                        gt_mask_tensor,
                        base_binary,
                    )
                    base_prompt_points = prompt[1]
                    extra_prompt_points = prompt[3]
                    results[completion_idx] = {
                        "valid": True,
                        "box_iou": box_iou,
                        "base_iou": base_iou,
                        "extra_iou": extra_iou,
                        "extra_iou_improvement": extra_iou_improvement,
                        "full_iou": extra_iou,
                        "iou_improvement": extra_iou_improvement,
                        **point_stats,
                        "num_points": len(extra_points),
                        "num_positive_points": sum(1 for label in extra_labels if label == 1),
                        "num_negative_points": sum(1 for label in extra_labels if label == 0),
                        "num_base_points": len(base_points),
                        "num_base_positive_points": sum(1 for label in base_labels if label == 1),
                        "num_base_negative_points": sum(1 for label in base_labels if label == 0),
                        "has_extra_schema": has_extra_schema,
                        "duplicate_penalty": _near_duplicate_penalty(extra_prompt_points),
                        "all_duplicate_penalty": _near_duplicate_penalty(base_prompt_points + extra_prompt_points),
                    }
                except Exception:
                    continue

        reward_eval_cache["key"] = cache_key
        reward_eval_cache["results"] = results
        return results

    # completions是LLM的生成内容
    def iou_reward_func(completions, **kwargs) -> list[float]:
        eval_results = _evaluate_sam_prompts(completions, **kwargs)
        rewards = [
            result["full_iou"] if result.get("valid") else INVALID_REWARD
            for result in eval_results
        ]
        nonlocal global_step
        avg_iou = np.mean(rewards) if rewards else 0.0
        tb_writer.add_scalar("Custom/Avg_IoU", avg_iou, global_step)
        global_step += 1

        return rewards


    def soft_format_reward_func(completions, **kwargs) -> list[float]:
        #pattern = r"<observe>.*?</observe>\s*<answer>.*?</answer>"
        pattern = r"\s*<answer>.*?</answer>\s*"
        responses = [completion for completion in completions]
        matches = [re.fullmatch(pattern, r, flags=re.DOTALL) for r in responses]
        return [FORMAT_VALID_REWARD if match else INVALID_REWARD for match in matches]



    def seg_segmentation_format_reward(completions, **kwargs) -> list[float]:
        rewards=[]
        for responds in completions:       # 当前组的batch个生成回复
            #print("{}\n".format(responds))
            data = _json_prompt_from_completion(responds)
            prompt = _extract_sam_prompt(data, require_new_schema=True)
            rewards.append(FORMAT_VALID_REWARD if prompt is not None else INVALID_REWARD)

        nonlocal global_step
        avg_format = np.mean(rewards) if rewards else 0.0
        tb_writer.add_scalar("Custom/Avg_Format_Reward", avg_format, global_step)
        global_step += 1

        return rewards

    def point_quality_reward_func(completions, **kwargs) -> list[float]:
        eval_results = _evaluate_sam_prompts(completions, **kwargs)
        rewards = []
        valid_results = []
        for result in eval_results:
            if not result.get("valid"):
                rewards.append(0.0)
                continue
            adaptive_count_reward = _adaptive_point_count_reward(result)
            point_reward = (
                EXTRA_IMPROVEMENT_WEIGHT * result["extra_iou_improvement"]
                + POINT_ERROR_MAP_WEIGHT * result["point_error_map"]
                + POINT_ADAPTIVE_COUNT_WEIGHT * adaptive_count_reward
                - HARMFUL_POINT_WEIGHT * result["harmful_point_rate"]
                - POINT_REDUNDANCY_WEIGHT * result["duplicate_penalty"]
                - POINT_COUNT_COST * result["num_points"]
            )
            result["adaptive_count_reward"] = adaptive_count_reward
            rewards.append(point_reward)
            valid_results.append(result)

        nonlocal global_step
        tb_writer.add_scalar("Custom/Avg_Point_Quality_Reward", np.mean(rewards) if rewards else 0.0, global_step)
        if valid_results:
            tb_writer.add_scalar("Custom/Avg_Box_IoU", np.mean([r["box_iou"] for r in valid_results]), global_step)
            tb_writer.add_scalar("Custom/Avg_Base_IoU", np.mean([r["base_iou"] for r in valid_results]), global_step)
            tb_writer.add_scalar("Custom/Avg_Extra_IoU", np.mean([r["extra_iou"] for r in valid_results]), global_step)
            tb_writer.add_scalar("Custom/Avg_Full_IoU", np.mean([r["full_iou"] for r in valid_results]), global_step)
            tb_writer.add_scalar(
                "Custom/Avg_Extra_IoU_Improvement",
                np.mean([r["extra_iou_improvement"] for r in valid_results]),
                global_step,
            )
            tb_writer.add_scalar(
                "Custom/Avg_Point_Error_Map",
                np.mean([r["point_error_map"] for r in valid_results]),
                global_step,
            )
            tb_writer.add_scalar(
                "Custom/Avg_Harmful_Point_Rate",
                np.mean([r["harmful_point_rate"] for r in valid_results]),
                global_step,
            )
            tb_writer.add_scalar(
                "Custom/Avg_False_Positive_Ratio",
                np.mean([r["false_positive_ratio"] for r in valid_results]),
                global_step,
            )
            tb_writer.add_scalar(
                "Custom/Avg_False_Negative_Ratio",
                np.mean([r["false_negative_ratio"] for r in valid_results]),
                global_step,
            )
            tb_writer.add_scalar(
                "Custom/Avg_Adaptive_Count_Reward",
                np.mean([r["adaptive_count_reward"] for r in valid_results]),
                global_step,
            )
            tb_writer.add_scalar("Custom/Avg_Num_Points", np.mean([r["num_points"] for r in valid_results]), global_step)
            tb_writer.add_scalar(
                "Custom/Avg_Num_Base_Points",
                np.mean([r["num_base_points"] for r in valid_results]),
                global_step,
            )
            tb_writer.add_scalar(
                "Custom/Avg_Num_Positive_Points",
                np.mean([r["num_positive_points"] for r in valid_results]),
                global_step,
            )
            tb_writer.add_scalar(
                "Custom/Avg_Num_Negative_Points",
                np.mean([r["num_negative_points"] for r in valid_results]),
                global_step,
            )
            tb_writer.add_scalar(
                "Custom/Avg_Desired_Positive_Points",
                np.mean([r["desired_positive_points"] for r in valid_results]),
                global_step,
            )
            tb_writer.add_scalar(
                "Custom/Avg_Desired_Negative_Points",
                np.mean([r["desired_negative_points"] for r in valid_results]),
                global_step,
            )
            tb_writer.add_scalar(
                "Custom/Avg_Helpful_Positive_Points",
                np.mean([r["helpful_positive_points"] for r in valid_results]),
                global_step,
            )
            tb_writer.add_scalar(
                "Custom/Avg_Helpful_Negative_Points",
                np.mean([r["helpful_negative_points"] for r in valid_results]),
                global_step,
            )
        global_step += 1

        return rewards

    train_dataset = TrainDataset(
        args.dataset_dir,
        args.train_dataset,
        processor
    )

    training_args = GRPOConfig(
        use_vllm=False,
        learning_rate=1e-5,  #5e-6
        adam_beta1=0.9,
        adam_beta2=0.99,
        beta = 0.01,     #KL coefficient，应该0.001
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        #lr_scheduler_type="constant",  # 固定学习率
        optim="adafactor",   #
        logging_steps=1,
        bf16=True,   #
        fp16=False,
        per_device_train_batch_size=5,    #重复后的batch，实际batch=per_device_train_batch_size/num_generations
        gradient_accumulation_steps=1,
        num_generations=5,  # 每个组的数量
        reward_weights=[0.5, 3.0, 0.5, 1.0],
        reprompt_threshold=args.reprompt_threshold,
        max_reprompt_rounds=args.max_reprompt_rounds,
        revision_reward_weight=args.revision_reward_weight,
        revision_advantage_weight=args.revision_advantage_weight,
        max_prompt_length=None,
        max_completion_length=160,
        temperature =1,
        num_train_epochs=1,  # Set to 1 for a full training run
        max_grad_norm =1,
        logging_dir=str(tb_log_dir),
        output_dir=str(output_dir),
    )

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,  # 缩放系数 32
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    trainer = CompatGRPOTrainer(
        model = reasoning_model,
        reward_funcs=[soft_format_reward_func,iou_reward_func,seg_segmentation_format_reward,point_quality_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
        peft_config=lora_config
    )
    trainer.reprompt_segmentation_model = segmentation_model
    trainer.reprompt_torch_dtype = torch_dtype
    trainer.reprompt_overlay_alpha = args.reprompt_overlay_alpha

    # ========== 关键步骤5：启动训练 + 关闭 TensorBoard 写入器 ==========
    try:
        trainer.train()
    finally:
        # 训练结束后，关闭 TensorBoard 写入器（确保日志保存完整）
        tb_writer.close()
        print(f"TensorBoard 日志已保存到：{tb_log_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
