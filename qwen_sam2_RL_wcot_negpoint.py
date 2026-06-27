import argparse
import os
import sys
import json
import warnings

import numpy as np
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from utils.dataset_qwen_RL2_negpoint import TrainDataset
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
    parser.add_argument("--reprompt_threshold", default=0.3, type=float,
                        help="IoU reward threshold used to stop training-time re-prompting")
    parser.add_argument("--max_reprompt_rounds", default=2, type=int,
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
        "point_schema": "one_positive_point_one_negative_point",
        "point_location_reward": "positive_only_no_penalty_for_misses",
        "reward_weights": [1.0, 1.0, 1.0, 0.2],
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
                                              #use_fast=False,
                                              padding_side="left",
                                              local_files_only=True)

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(qwen_model_path),
        dtype=torch_dtype,   #模型fp32内存不够，fp16自动缩放出问题，最好是bf16
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

    INVALID_REWARD = -1.0
    FORMAT_VALID_REWARD = 1.0

    def _is_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _to_float(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _json_prompt_from_completion(completion):
        try:
            data = json.loads(extract_xml_answer(completion))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def _normalize_bbox(value):
        if not (isinstance(value, list) and len(value) == 4):
            return None
        if not all(_is_number(coord) for coord in value):
            return None
        return [round(float(coord)) for coord in value]

    def _bbox_in_prompt_bounds(bbox):
        x1, y1, x2, y2 = bbox
        return 0 <= x1 < x2 <= 336 and 0 <= y1 < y2 <= 336

    def _normalize_point(value):
        if not (isinstance(value, list) and len(value) == 2):
            return None
        if not all(_is_number(coord) for coord in value):
            return None
        return [round(float(value[0])), round(float(value[1]))]

    def _point_in_prompt_bounds(point):
        x, y = point
        return 0 <= x < 336 and 0 <= y < 336

    def _find_point_key(data, tokens):
        matches = []
        for key in data.keys():
            lower_key = key.lower()
            if "point" in lower_key and any(token in lower_key for token in tokens):
                matches.append(key)
        return matches[0] if len(matches) == 1 else None

    def _extract_two_point_prompt(data):
        if not isinstance(data, dict):
            return None

        bbox_key = next((key for key in data.keys() if "bbox" in key.lower()), None)
        point_keys = [key for key in data.keys() if "point" in key.lower()]
        if bbox_key is None or len(point_keys) != 2:
            return None

        positive_key = _find_point_key(data, ("positive", "foreground", "include", "add"))
        negative_key = _find_point_key(data, ("negative", "background", "exclude", "remove"))
        if positive_key is None or negative_key is None or positive_key == negative_key:
            return None

        bbox = _normalize_bbox(data.get(bbox_key))
        positive_point = _normalize_point(data.get(positive_key))
        negative_point = _normalize_point(data.get(negative_key))
        if bbox is None or positive_point is None or negative_point is None:
            return None
        if not (
            _bbox_in_prompt_bounds(bbox)
            and _point_in_prompt_bounds(positive_point)
            and _point_in_prompt_bounds(negative_point)
        ):
            return None

        return {
            "bbox": bbox,
            "positive_point": positive_point,
            "negative_point": negative_point,
        }

    def _sample_index(completion_idx, num_completions, batch_size):
        if batch_size <= 1:
            return 0
        if num_completions == batch_size:
            return completion_idx
        if num_completions % batch_size == 0:
            return min(completion_idx // max(1, num_completions // batch_size), batch_size - 1)
        return min(completion_idx, batch_size - 1)

    def _resize_factor_at(resize_factors, index):
        factor = resize_factors
        if isinstance(resize_factors, (list, tuple)) and resize_factors:
            first = resize_factors[0]
            if isinstance(first, (list, tuple, torch.Tensor)):
                factor = resize_factors[min(index, len(resize_factors) - 1)]
        if isinstance(factor, torch.Tensor):
            factor = factor.detach().cpu().tolist()
        return _to_float(factor[0]), _to_float(factor[1])

    def _scale_prompt(prompt, resize_factors, index):
        rfx, rfy = _resize_factor_at(resize_factors, index)
        bbox = prompt["bbox"]
        positive_point = prompt["positive_point"]
        negative_point = prompt["negative_point"]
        scaled_bbox = [
            round(float(bbox[0]) * rfx),
            round(float(bbox[1]) * rfy),
            round(float(bbox[2]) * rfx),
            round(float(bbox[3]) * rfy),
        ]
        scaled_points = [
            [round(float(positive_point[0]) * rfx), round(float(positive_point[1]) * rfy)],
            [round(float(negative_point[0]) * rfx), round(float(negative_point[1]) * rfy)],
        ]
        return scaled_bbox, scaled_points, [1, 0]

    def _prepare_gt_mask(gt_mask, device):
        gt_mask_tensor = gt_mask if isinstance(gt_mask, torch.Tensor) else torch.as_tensor(gt_mask)
        gt_mask_tensor = gt_mask_tensor.to(device=device).int()
        if gt_mask_tensor.dim() == 2:
            gt_mask_tensor = gt_mask_tensor.unsqueeze(0)
        if gt_mask_tensor.dim() == 3 and gt_mask_tensor.shape[0] > 1:
            gt_mask_tensor = gt_mask_tensor[:1]
        return gt_mask_tensor

    def _best_binary_mask(masks, scores, device):
        masks = torch.from_numpy(masks).to(device=device, dtype=torch_dtype)
        scores = torch.from_numpy(scores).to(device=device, dtype=torch_dtype)
        sorted_ind = torch.argsort(scores, dim=-1, descending=True)
        return (masks[sorted_ind][0] > 0).int().unsqueeze(0)

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

    # completions是LLM的生成内容，可以直接生成点进行计算。 那么就没有隐藏层了？ prompts在问答模板里，answer是答案(不需要答案，直接计算iou)
    # 注意：completions是生成的文本列表，masks/images等是当前批次的对应数据
    # completions维度[batch_size][num_generations]  prompts维度[batch_size]
    # 奖励函数个数也应该与input_id相同，不要分组计算
    def iou_reward_func(completions, **kwargs) -> list[float]:
        batch_images = kwargs.get("images")      #  list[ [C, H, W] ]
        batch_masks = kwargs.get("masks")
        resize_factor = kwargs.get("resize_factor")
        rewards = []

        if not batch_images or not batch_masks:
            return [INVALID_REWARD for _ in completions]

        last_image_idx = None
        for completion_idx, completion in enumerate(completions):
            prompt = _extract_two_point_prompt(_json_prompt_from_completion(completion))
            if prompt is None:
                rewards.append(INVALID_REWARD)
                continue

            sample_idx = _sample_index(completion_idx, len(completions), len(batch_images))
            content_bbox, points_input, point_labels = _scale_prompt(prompt, resize_factor, sample_idx)
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch_dtype):
                    if last_image_idx != sample_idx:
                        segmentation_model.set_image(batch_images[sample_idx])
                        last_image_idx = sample_idx
                    masks, scores, _ = segmentation_model.predict(
                        point_coords=points_input,
                        point_labels=point_labels,
                        box=content_bbox
                    )
                    pred_binary = _best_binary_mask(masks, scores, "cuda")
                gt_mask = _prepare_gt_mask(batch_masks[sample_idx], pred_binary.device)
                rewards.append(_compute_binary_iou(pred_binary, gt_mask))
            except Exception:
                rewards.append(INVALID_REWARD)

        nonlocal global_step
        avg_iou = np.mean(rewards) if rewards else 0.0
        tb_writer.add_scalar("Custom/Avg_IoU", avg_iou, global_step)
        global_step += 1

        return rewards

    def point_location_reward_func(completions, **kwargs) -> list[float]:
        batch_masks = kwargs.get("masks")
        resize_factor = kwargs.get("resize_factor")
        rewards = []
        positive_hits = []
        negative_hits = []

        if not batch_masks:
            return [0.0 for _ in completions]

        for completion_idx, completion in enumerate(completions):
            prompt = _extract_two_point_prompt(_json_prompt_from_completion(completion))
            if prompt is None:
                rewards.append(0.0)
                positive_hits.append(0.0)
                negative_hits.append(0.0)
                continue

            sample_idx = _sample_index(completion_idx, len(completions), len(batch_masks))
            content_bbox, points_input, _ = _scale_prompt(prompt, resize_factor, sample_idx)
            gt_mask = _prepare_gt_mask(batch_masks[sample_idx], "cpu").squeeze(0)
            height, width = gt_mask.shape[-2:]
            valid = gt_mask != 255
            gt_bool = (gt_mask > 0) & valid

            positive_point, negative_point = points_input
            pos_x, pos_y = positive_point
            neg_x, neg_y = negative_point
            x1, y1, x2, y2 = content_bbox

            positive_ok = (
                0 <= pos_x < width
                and 0 <= pos_y < height
                and bool(valid[pos_y, pos_x].item())
                and bool(gt_bool[pos_y, pos_x].item())
            )
            negative_in_image = (
                0 <= neg_x < width
                and 0 <= neg_y < height
                and bool(valid[neg_y, neg_x].item())
            )
            negative_in_bbox = (
                max(0, x1) <= neg_x <= min(width - 1, x2)
                and max(0, y1) <= neg_y <= min(height - 1, y2)
            )
            negative_ok = (
                negative_in_image
                and negative_in_bbox
                and not bool(gt_bool[neg_y, neg_x].item())
            )

            positive_hits.append(1.0 if positive_ok else 0.0)
            negative_hits.append(1.0 if negative_ok else 0.0)
            rewards.append((0.5 if positive_ok else 0.0) + (0.5 if negative_ok else 0.0))

        nonlocal global_step
        tb_writer.add_scalar("Custom/Avg_Point_Location_Reward", np.mean(rewards) if rewards else 0.0, global_step)
        tb_writer.add_scalar("Custom/Positive_Point_On_GT", np.mean(positive_hits) if positive_hits else 0.0, global_step)
        tb_writer.add_scalar(
            "Custom/Negative_Point_In_Box_Out_GT",
            np.mean(negative_hits) if negative_hits else 0.0,
            global_step,
        )
        global_step += 1

        return rewards

    def soft_format_reward_func(completions, **kwargs) -> list[float]:
        pattern = r"\s*<answer>.*?</answer>\s*"
        responses = [completion for completion in completions]
        matches = [re.fullmatch(pattern, r, flags=re.DOTALL) for r in responses]
        return [FORMAT_VALID_REWARD if match else INVALID_REWARD for match in matches]

    def seg_segmentation_format_reward(completions, **kwargs) -> list[float]:
        rewards=[]
        for responds in completions:       # 当前组的batch个生成回复
            prompt = _extract_two_point_prompt(_json_prompt_from_completion(responds))
            rewards.append(FORMAT_VALID_REWARD if prompt is not None else INVALID_REWARD)

        nonlocal global_step
        avg_format = np.mean(rewards) if rewards else 0.0
        tb_writer.add_scalar("Custom/Avg_Format_Reward", avg_format, global_step)
        global_step += 1

        return rewards


    def seg_compute_score(completions, **kwargs) -> float:
        nonlocal global_step  # 引用外部的 global_step（记录步数）
        soft_format_reward = soft_format_reward_func(completions, **kwargs)
        segmentation_format_reward = seg_segmentation_format_reward(completions, **kwargs)
        iou_reward = iou_reward_func(completions, **kwargs)
        point_location_reward = point_location_reward_func(completions, **kwargs)

        # 计算各指标平均值（用于 TensorBoard 可视化）
        avg_iou = np.mean(iou_reward) if iou_reward else 0.0
        avg_format = np.mean(segmentation_format_reward) if segmentation_format_reward else 0.0
        avg_soft = np.mean(soft_format_reward) if soft_format_reward else 0.0
        avg_point = np.mean(point_location_reward) if point_location_reward else 0.0
        avg_total = np.mean([
            i + s + f + p
            for i, s, f, p in zip(
                iou_reward,
                soft_format_reward,
                segmentation_format_reward,
                point_location_reward,
            )
        ])

        # 将自定义指标写入 TensorBoard（和训练步数关联）
        tb_writer.add_scalar("Custom/Avg_IoU", avg_iou, global_step)
        tb_writer.add_scalar("Custom/Avg_Format_Reward", avg_format, global_step)
        tb_writer.add_scalar("Custom/Avg_Soft_Reward", avg_soft, global_step)
        tb_writer.add_scalar("Custom/Avg_Point_Location_Reward", avg_point, global_step)
        tb_writer.add_scalar("Custom/Avg_Total_Reward", avg_total, global_step)

        total_rewards = [
            iou + soft + seg + point
            for iou, soft, seg, point in zip(
                iou_reward,
                soft_format_reward,
                segmentation_format_reward,
                point_location_reward,
            )
        ]
        global_step += 1
        return total_rewards

    train_dataset = TrainDataset(
        args.dataset_dir,
        args.train_dataset,
        processor
    )
    #6.13调整训练参数 ：学习率，weight_decay ，scale_rewards，iou权重
    # training_args = GRPOConfig(
    #     use_vllm=False,  # use vLLM for fast inference!
    #     learning_rate=5e-6,  # 1e-5
    #     adam_beta1=0.9,
    #     adam_beta2=0.99,
    #     beta = 0.002,     #KL coefficient，应该0.001
    #     weight_decay=0.01,
    #     warmup_ratio=0.03,
    #     lr_scheduler_type="cosine",
    #     #lr_scheduler_type="constant",  # 固定学习率
    #     optim="adafactor",   #      adamw_torch
    #     logging_steps=1,
    #     bf16=True,   #
    #     fp16=False,   #模型权重和中间值, False就是fp32，启用fp16但是梯度和损失依然是fp32。 False/True内存占用一样。
    #     per_device_train_batch_size=6,    #重复后的batch，实际batch=per_device_train_batch_size/num_generations
    #     gradient_accumulation_steps=1,
    #     num_generations=6,  # 每个组的数量
    #     reprompt_threshold=args.reprompt_threshold,
    #     max_reprompt_rounds=args.max_reprompt_rounds,
    #     revision_reward_weight=args.revision_reward_weight,
    #     revision_advantage_weight=args.revision_advantage_weight,
    #     max_prompt_length=None,
    #     max_completion_length=100,
    #     temperature =1,  #还是需要一定温度，否则回答全部都是一样的  ，之前0.4效果不好，继续修改为1
    #     num_train_epochs=1,  # Set to 1 for a full training run
    #     max_grad_norm =0.5, #1
    #     scale_rewards=False, #新增
    #     reward_weights=[1, 2.0, 1],
    #     #max_steps=16000,
    #     #save_steps=5000,
    #     logging_dir=str(tb_log_dir),
    #     output_dir=str(output_dir),
    # )
    training_args = GRPOConfig(
        use_vllm=False,  # use vLLM for fast inference!
        learning_rate=1e-5,  # 5e-6
        adam_beta1=0.9,
        adam_beta2=0.99,
        beta=0.001,  # KL coefficient，应该0.001
        weight_decay=0.0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        # lr_scheduler_type="constant",  # 固定学习率
        optim="adafactor",  # adamw_torch
        logging_steps=1,
        bf16=True,  #
        fp16=False,  # 模型权重和中间值, False就是fp32，启用fp16但是梯度和损失依然是fp32。 False/True内存占用一样。
        per_device_train_batch_size=6,  # 重复后的batch，实际batch=per_device_train_batch_size/num_generations
        gradient_accumulation_steps=1,
        num_generations=6,  # 每个组的数量
        reprompt_threshold=args.reprompt_threshold,
        max_reprompt_rounds=args.max_reprompt_rounds,
        revision_reward_weight=args.revision_reward_weight,
        revision_advantage_weight=args.revision_advantage_weight,
        max_prompt_length=None,
        max_completion_length=100,
        temperature=1,  # 还是需要一定温度，否则回答全部都是一样的  ，之前0.4效果不好，继续修改为1
        num_train_epochs=1,  # Set to 1 for a full training run
        max_grad_norm=1,
        reward_weights=[1.0, 1.0, 1.0, 0.2],
        # max_steps=16000,
        # save_steps=5000,
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
        #reward_funcs=[seg_compute_score],
        reward_funcs=[
            soft_format_reward_func,
            iou_reward_func,
            seg_segmentation_format_reward,
            point_location_reward_func,
        ],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
        peft_config=lora_config
    )
    trainer.reprompt_segmentation_model = segmentation_model
    trainer.reprompt_torch_dtype = torch_dtype
    trainer.reprompt_overlay_alpha = args.reprompt_overlay_alpha
    #trainer.train(resume_from_checkpoint="outputs/checkpoint-14000-cvpr-wcot-0.6")

    # ========== 关键步骤5：启动训练 + 关闭 TensorBoard 写入器 ==========
    try:
        trainer.train()
    finally:
        # 训练结束后，关闭 TensorBoard 写入器（确保日志保存完整）
        tb_writer.close()
        print(f"TensorBoard 日志已保存到：{tb_log_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
