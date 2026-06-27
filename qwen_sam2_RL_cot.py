import argparse
import os
import sys
import json
import warnings

import numpy as np
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen3VLForConditionalGeneration
from utils.dataset_qwen_RL2_cot import TrainDataset
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
    parser.add_argument("--train_dataset", default="refcoco+|unc|train", type=str,
                        choices=["refcoco|unc|train",
                                 "refcoco+|unc|train",
                                 "refcocog|umd|train",
                                 "ReasonSeg|train"])
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    #parser.add_argument("--segmentation_model_path", type=str, default="checkpoints/sam2")
    parser.add_argument("--output_dir", default="outputs", type=str, help="Training output directory")
    parser.add_argument("--reprompt_threshold", default=0.3, type=float,
                        help="IoU reward threshold used to stop training-time re-prompting")
    parser.add_argument("--max_reprompt_rounds", default=1, type=int,
                        help="Maximum number of corrective re-prompt rounds after the initial generation")
    parser.add_argument("--revision_reward_weight", default=1.0, type=float,
                        help="Weight for the revision improvement reward shaping term")
    parser.add_argument("--revision_advantage_weight", default=1.0, type=float,
                        help="Weight for scaling GRPO advantages by revision improvement")
    parser.add_argument("--cot_length_mode", default="structure", type=str, choices=["structure"],
                        help="Use structured light/medium/heavy reasoning rewards")
    parser.add_argument("--cot_length_targets", default="32,64,128", type=str,
                        help="Comma-separated CoT range anchors mapped to light, medium, and heavy reasoning")
    parser.add_argument("--cot_structure_reward_weight", default=0.6, type=float,
                        help="Weight for matching the requested light/medium/heavy reasoning structure")
    parser.add_argument("--cot_length_reward_weight", default=0.3, type=float,
                        help="Weight for keeping CoT token length inside the requested range")
    parser.add_argument("--cot_repetition_penalty_weight", default=0.2, type=float,
                        help="Penalty weight for repeated n-grams in CoT")
    parser.add_argument("--cot_unique_penalty_weight", default=0.2, type=float,
                        help="Penalty weight for low unique-token ratio in CoT")
    parser.add_argument("--cot_prompt_copy_penalty_weight", default=0.3, type=float,
                        help="Penalty weight for copying n-grams from the referring expression")
    parser.add_argument("--max_completion_length", default=240, type=int,
                        help="Maximum generated tokens; must cover the requested CoT plus answer JSON")
    return parser.parse_args(args)

def main(args):

    args = parse_args(args)
    cot_length_targets = [int(length.strip()) for length in args.cot_length_targets.split(",") if length.strip()]
    if len(cot_length_targets) != 3:
        raise ValueError("--cot_length_targets must contain exactly three anchors for light, medium, and heavy reasoning")
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
        "cot_length_mode": args.cot_length_mode,
        "cot_length_targets": cot_length_targets,
        "cot_structure_reward_weight": args.cot_structure_reward_weight,
        "cot_length_reward_weight": args.cot_length_reward_weight,
        "cot_repetition_penalty_weight": args.cot_repetition_penalty_weight,
        "cot_unique_penalty_weight": args.cot_unique_penalty_weight,
        "cot_prompt_copy_penalty_weight": args.cot_prompt_copy_penalty_weight,
        "max_completion_length": args.max_completion_length,
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

    def count_text_tokens(text: str) -> int:
        tokenized = processor.tokenizer(text, add_special_tokens=False)
        return len(tokenized["input_ids"])

    def expand_cot_targets(cot_target_lengths, completions_count):
        if not cot_target_lengths:
            return [cot_length_targets[0]] * completions_count
        targets = [int(target) for target in cot_target_lengths]
        if len(targets) == completions_count:
            return targets
        if completions_count % len(targets) == 0:
            repeat_count = completions_count // len(targets)
            return [target for target in targets for _ in range(repeat_count)]
        return [targets[idx % len(targets)] for idx in range(completions_count)]

    def expand_values(values, completions_count, default=None):
        if not values:
            return [default] * completions_count
        if len(values) == completions_count:
            return values
        if completions_count % len(values) == 0:
            repeat_count = completions_count // len(values)
            return [value for value in values for _ in range(repeat_count)]
        return [values[idx % len(values)] for idx in range(completions_count)]

    def cot_level_for_target(target_length):
        if not cot_length_targets:
            return "medium"
        if int(target_length) == cot_length_targets[0]:
            return "light"
        if int(target_length) == cot_length_targets[-1]:
            return "heavy"
        return "medium"

    def cot_required_sections(level):
        if level == "light":
            return ["Observation"]
        if level == "heavy":
            return ["Observation", "Comparison", "Context reasoning"]
        return ["Observation", "Comparison"]

    def cot_length_range(level):
        if level == "light":
            return 12, 30
        if level == "heavy":
            return 85, 160
        return 35, 80

    def length_range_score(cot_len, min_len, max_len):
        if min_len <= cot_len <= max_len:
            return 1.0
        if cot_len < min_len:
            return max(0.0, cot_len / max(min_len, 1))
        return max(0.0, 1.0 - (cot_len - max_len) / max(max_len, 1))

    def structure_score(think_text, level):
        lower_text = think_text.lower()
        required = cot_required_sections(level)
        matched = sum(1 for section in required if f"{section.lower()}:" in lower_text)
        score = matched / max(len(required), 1)

        disallowed = []
        if level == "light":
            disallowed = ["comparison:", "context reasoning:"]
        elif level == "medium":
            disallowed = ["context reasoning:"]
        extra_sections = sum(1 for section in disallowed if section in lower_text)
        return max(0.0, score - 0.25 * extra_sections)

    def text_tokens(text):
        return re.findall(r"[A-Za-z0-9]+", text.lower())

    def ngram_repeat_penalty(text, n=3, threshold=0.25):
        tokens = text_tokens(text)
        if len(tokens) < n * 2:
            return 0.0
        ngrams = [tuple(tokens[idx:idx + n]) for idx in range(len(tokens) - n + 1)]
        repeat_ratio = 1.0 - len(set(ngrams)) / max(len(ngrams), 1)
        return max(0.0, repeat_ratio - threshold)

    def unique_token_penalty(text, threshold=0.45):
        tokens = text_tokens(text)
        if len(tokens) < 8:
            return 0.0
        unique_ratio = len(set(tokens)) / len(tokens)
        return max(0.0, threshold - unique_ratio)

    def prompt_copy_penalty(think_text, question_text, n=3, threshold=0.20):
        think_tokens = text_tokens(think_text)
        query_tokens = text_tokens(question_text)
        if len(think_tokens) < n or len(query_tokens) < n:
            return 0.0
        think_ngrams = {tuple(think_tokens[idx:idx + n]) for idx in range(len(think_tokens) - n + 1)}
        query_ngrams = {tuple(query_tokens[idx:idx + n]) for idx in range(len(query_tokens) - n + 1)}
        overlap = len(think_ngrams & query_ngrams) / max(len(think_ngrams), 1)
        return max(0.0, overlap - threshold)

    def first_sampled_class(sampled_class):
        if isinstance(sampled_class, (list, tuple)):
            return str(sampled_class[0]) if sampled_class else ""
        return str(sampled_class or "")

    # completions是LLM的生成内容，可以直接生成点进行计算。 那么就没有隐藏层了？ prompts在问答模板里，answer是答案(不需要答案，直接计算iou)
    # 注意：completions是生成的文本列表，masks/images等是当前批次的对应数据
    # completions维度[batch_size][num_generations]  prompts维度[batch_size]
    # 奖励函数个数也应该与input_id相同，不要分组计算
    def iou_reward_func(completions, **kwargs) -> list[float]:
        batch_images = kwargs.get("images")      #  list[ [C, H, W] ]
        batch_masks = kwargs.get("masks")
        resize_factor = kwargs.get("resize_factor")
        rfx, rfy = resize_factor[0]  #重复采样，只取一个
        batch = len(batch_images)  #group

        # completions 调整成batch个
        grouped_completions = []
        for index in range(batch):
            start = index * batch_masks[0].shape[0]
            end = (index + 1) * batch_masks[0].shape[0]
            group = completions[start:end]  # 取出当前样本的所有生成回复
            grouped_completions.append(group)
        completions = grouped_completions

        image = batch_images[0]
        rewards = []
        for batch_idx in range(len(batch_images)):
            sample_completions = completions[batch_idx]
            gt_mask = batch_masks[batch_idx] #1，428,640

            # 为每个生成计算独立奖励：pred_masks维度1，gt_mask维度1。奖励要么因为格式问题为0，要么是一对一计算iou
            pred_masks=[]
            for completion in sample_completions:       # 当前组的batch个生成回复
                try:
                    data = json.loads(extract_xml_answer(completion))
                except json.JSONDecodeError:
                    rewards.append(-1.0)  # 仅对当前completion给0奖励
                    continue
                if not isinstance(data, dict):  # 检查是否是字典
                    rewards.append(-1.0)  # 仅对当前completion给0奖励
                    continue

                bbox_key = None
                points_keys = []
                for key in data.keys():
                    if "bbox" in key.lower() and not bbox_key:
                        bbox_key = key  # 此时bbox_key就是字符 "bbox"
                    elif "point" in key.lower():
                        points_keys.append(key)  # points_keys就是字符 "point"

                if not (bbox_key and len(points_keys) >= 2):
                    rewards.append(-1.0)  # 仅对当前completion给0奖励
                    continue

                check_point=False  # 检查point长度为2，且为数值
                if not isinstance(data[bbox_key], list):
                    rewards.append(-1.0)
                    continue
                if len(data[bbox_key]) != 4:
                    rewards.append(-1.0)
                    continue
                for coord in data[bbox_key]:
                    if not isinstance(coord, (int, float)):
                        check_point = True
                        break

                for point_key in points_keys[:2]:
                    if not isinstance(data[point_key], list):
                        check_point = True
                        break
                    if len(data[point_key]) != 2:
                        check_point=True
                        break
                    for coord in data[point_key]:
                        if not isinstance(coord,(int,float)):
                            check_point = True
                            break
                if check_point:
                    rewards.append(-1.0)
                    continue

                bbox = data[bbox_key]  # bbox的值
                points = [data[key] for key in points_keys]

                #sam2
                point1_sam2,point2_sam2 = points[0],points[1]
                point1 = [round(int(point1_sam2[0])*rfx), round(int(point1_sam2[1])*rfy)]
                point2 = [round(int(point2_sam2[0])*rfx), round(int(point2_sam2[1])*rfy)]
                points_input = [point1, point2]
                content_bbox = [round(int(bbox[0])*rfx), round(int(bbox[1])*rfy), round(int(bbox[2])*rfx), round(int(bbox[3])*rfy)]


            if len(rewards)==(batch_idx+1)*len(sample_completions): #不满足要求，不进入mask计算
                continue
            #sam2
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch_dtype):
                segmentation_model.set_image(image)
                masks, scores, _ = segmentation_model.predict(
                    point_coords=points_input,
                    point_labels=[1, 1],
                    box=content_bbox
                )
                masks = torch.from_numpy(masks).to("cuda", dtype=torch_dtype)
                scores = torch.from_numpy(scores).to("cuda", dtype=torch_dtype)
                sorted_ind = torch.argsort(scores, dim=-1, descending=True)  # 添加dim参数
                masks = masks[sorted_ind]
                pred_masks.append(masks[0])  # 直接使用张量，不转numpy
            #pred_masks = masks[0].astype(bool)
            pred_masks = torch.stack(pred_masks, dim=0)
            pred_binary = (pred_masks > 0).int()
                         #pred_binary = pred_binary.unsqueeze(0)   #使用sam2新加，不用sam2要去掉
            gt_mask = gt_mask.int()  #本身就是B，428,640

            intersection, union, _ = intersectionAndUnionGPU(
                pred_binary.contiguous().clone(),
                gt_mask.contiguous(),
                2,
                ignore_index=255
            )

            if union[1] > 0:  # 确保分母不为零，intersection为0？，预测完全错误？
                iou = intersection[1] / (union[1] + 1e-10)
            else:
                iou = torch.tensor(1.0)

            reward_iou = iou.item()
            rewards.append(reward_iou)
        nonlocal global_step
        avg_iou = np.mean(rewards) if rewards else 0.0
        tb_writer.add_scalar("Custom/Avg_IoU", avg_iou, global_step)
        global_step += 1

        return rewards


    def soft_format_reward_func(completions, **kwargs) -> list[float]:
        pattern = r"^\s*<think>(.*?)</think>\s*<answer>.*?</answer>\s*$"
        cot_targets = expand_cot_targets(kwargs.get("cot_target_length"), len(completions))
        sampled_classes = expand_values(kwargs.get("sampled_classes"), len(completions), default="")
        rewards = []
        cot_lengths = []
        structure_rewards = []
        length_rewards = []
        repetition_penalties = []
        unique_penalties = []
        copy_penalties = []
        for response, target_length, sampled_class in zip(completions, cot_targets, sampled_classes):
            level = cot_level_for_target(target_length)
            match = re.match(pattern, response, re.DOTALL)
            if not match:
                rewards.append(0.0)
                cot_lengths.append(0)
                structure_rewards.append(0.0)
                length_rewards.append(0.0)
                repetition_penalties.append(0.0)
                unique_penalties.append(0.0)
                copy_penalties.append(0.0)
                continue

            observe_content = match.group(1).strip()
            if not observe_content:
                rewards.append(0.0)
                cot_lengths.append(0)
                structure_rewards.append(0.0)
                length_rewards.append(0.0)
                repetition_penalties.append(0.0)
                unique_penalties.append(0.0)
                copy_penalties.append(0.0)
                continue
            cot_length = count_text_tokens(observe_content)
            min_len, max_len = cot_length_range(level)
            struct_score = structure_score(observe_content, level)
            length_score = length_range_score(cot_length, min_len, max_len)
            repeat_penalty = ngram_repeat_penalty(observe_content)
            unique_penalty = unique_token_penalty(observe_content)
            copy_penalty = prompt_copy_penalty(observe_content, first_sampled_class(sampled_class))

            cot_lengths.append(cot_length)
            structure_rewards.append(float(struct_score))
            length_rewards.append(float(length_score))
            repetition_penalties.append(float(repeat_penalty))
            unique_penalties.append(float(unique_penalty))
            copy_penalties.append(float(copy_penalty))
            rewards.append(
                1.0
                + args.cot_structure_reward_weight * float(struct_score)
                + args.cot_length_reward_weight * float(length_score)
                - args.cot_repetition_penalty_weight * float(repeat_penalty)
                - args.cot_unique_penalty_weight * float(unique_penalty)
                - args.cot_prompt_copy_penalty_weight * float(copy_penalty)
            )

        if rewards:
            tb_writer.add_scalar("Custom/Avg_CoT_Length", np.mean(cot_lengths), global_step)
            tb_writer.add_scalar("Custom/Avg_CoT_Anchor", np.mean(cot_targets), global_step)
            tb_writer.add_scalar("Custom/Avg_CoT_Structure_Reward", np.mean(structure_rewards), global_step)
            tb_writer.add_scalar("Custom/Avg_CoT_Length_Reward", np.mean(length_rewards), global_step)
            tb_writer.add_scalar("Custom/Avg_CoT_Repetition_Penalty", np.mean(repetition_penalties), global_step)
            tb_writer.add_scalar("Custom/Avg_CoT_Unique_Penalty", np.mean(unique_penalties), global_step)
            tb_writer.add_scalar("Custom/Avg_CoT_Copy_Penalty", np.mean(copy_penalties), global_step)
        return rewards

    def seg_segmentation_format_reward(completions, **kwargs) -> list[float]:
        rewards=[]
        for responds in completions:       # 当前组的batch个生成回复
            #print("{}\n".format(responds))
            try:
                data = json.loads(extract_xml_answer(responds))
            except json.JSONDecodeError:
                rewards.append(-1.0)  # 仅对当前completion给0奖励
                continue
            #此时data是3个元素的字典
            if not isinstance(data, dict):  # 检查是否是字典
                rewards.append(-1.0)  # 仅对当前completion给0奖励
                continue
            #print("进入坐标提取\n")
            bbox_key = None
            points_keys = []
            for key in data.keys():
                if "bbox" in key.lower() and not bbox_key:
                    bbox_key = key  # 此时bbox_key就是字符 "bbox"
                elif "point" in key.lower():
                    points_keys.append(key)  # points_keys就是字符 "point"
            if not (bbox_key and len(points_keys) >= 2):
                rewards.append(-1.0)  # 仅对当前completion给0奖励
                continue
            # 检查point的value长度为2，且为数值; 检查box的 value为数值，并长度为4
            check_point=False
            if not isinstance(data[bbox_key], list):
                rewards.append(-1.0)
                continue
            if len(data[bbox_key])!=4:
                rewards.append(-1.0)
                continue
            for coord in data[bbox_key]:
                if not isinstance(coord, (int, float)):
                    check_point = True
                    break

            for point_key in points_keys[:2]:
                if not isinstance(data[point_key],list):
                    check_point=True
                    break
                if len(data[point_key]) != 2:
                    check_point=True
                    break
                for coord in data[point_key]:
                    if not isinstance(coord,(int,float)):
                        check_point = True
                        break
            if check_point:
                rewards.append(-1.0)
                continue
            rewards.append(1.0)

        nonlocal global_step
        avg_format = np.mean(rewards) if rewards else 0.0
        tb_writer.add_scalar("Custom/Avg_Format_Reward", avg_format, global_step)
        global_step += 1

        return rewards


    train_dataset = TrainDataset(
        args.dataset_dir,
        args.train_dataset,
        processor,
        cot_length_targets=cot_length_targets,
        cot_length_mode=args.cot_length_mode,
    )
    #6.13 新训练参数更稳定
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
        optim="adamw_torch",  # adamw_torch
        logging_steps=1,
        bf16=True,  #
        fp16=False,  # 模型权重和中间值, False就是fp32，启用fp16但是梯度和损失依然是fp32。 False/True内存占用一样。
        per_device_train_batch_size=5,  # 重复后的batch，实际batch=per_device_train_batch_size/num_generations
        gradient_accumulation_steps=1,
        num_generations=5,  # 每个组的数量
        reprompt_threshold=args.reprompt_threshold,
        max_reprompt_rounds=args.max_reprompt_rounds,
        revision_reward_weight=args.revision_reward_weight,
        revision_advantage_weight=args.revision_advantage_weight,
        max_prompt_length=None,
        max_completion_length=args.max_completion_length,
        temperature=1,  # 还是需要一定温度，否则回答全部都是一样的  ，之前0.4效果不好，继续修改为1
        num_train_epochs=1,  # Set to 1 for a full training run
        max_grad_norm=1,
        #reward_weights=[1, 2.0, 1],
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
        reward_funcs=[soft_format_reward_func,iou_reward_func,seg_segmentation_format_reward],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
        peft_config=lora_config
    )
    #trainer.train(resume_from_checkpoint="outputs/checkpoint-10000")

    # ========== 关键步骤5：启动训练 + 关闭 TensorBoard 写入器 ==========
    try:
        trainer.train()
    finally:
        # 训练结束后，关闭 TensorBoard 写入器（确保日志保存完整）
        tb_writer.close()
        print(f"TensorBoard 日志已保存到：{tb_log_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
