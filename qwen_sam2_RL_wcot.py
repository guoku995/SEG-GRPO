import argparse
import os
import sys
import json
import random
import warnings

import numpy as np
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from utils.dataset_qwen_RL2_cot_wcot_orgin import TrainDataset
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
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed for reproducible training runs")
    parser.add_argument("--reprompt_threshold", default=0.3, type=float,   #原本是0.7-2
                        help="IoU reward threshold used to stop training-time re-prompting")
    parser.add_argument("--max_reprompt_rounds", default=2, type=int,
                        help="Maximum number of corrective re-prompt rounds after the initial generation")
    parser.add_argument("--revision_reward_weight", default=1.0, type=float,
                        help="Weight for the revision improvement reward shaping term")
    parser.add_argument("--revision_advantage_weight", default=1.0, type=float,
                        help="Weight for scaling GRPO advantages by revision improvement")
    return parser.parse_args(args)

def main(args):

    args = parse_args(args)
    project_root = Path(__file__).resolve().parent
    qwen_model_path = project_root / "checkpoints/qwen2.5_3B"
    output_dir = Path(args.output_dir)
    tb_log_dir = output_dir / "tb_logs"  # TensorBoard
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_config = {
        "train_dataset": args.train_dataset,
        "precision": args.precision,
        "segmentation_model_path": args.segmentation_model_path,
        "seed": args.seed,
        "reprompt_threshold": args.reprompt_threshold,
        "max_reprompt_rounds": args.max_reprompt_rounds,
        "revision_reward_weight": args.revision_reward_weight,
        "revision_advantage_weight": args.revision_advantage_weight,
        #"reprompt_overlay_alpha": args.reprompt_overlay_alpha,
    }
    with open(output_dir / "reprompt_experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, ensure_ascii=False, indent=2)
    print(f"Re-prompt experiment config: {experiment_config}")

    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    global_step = 0

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
    segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)

    def extract_xml_answer(text: str) -> str:
        answer = text.split("<answer>")[-1].split("</answer>")[0]
        answer = answer.replace("\n", "").strip()
        answer = re.sub(r'(\d+)\.(?=\D)', r'\1.0', answer)

        return answer


    def iou_reward_func(completions, **kwargs) -> list[float]:
        batch_images = kwargs.get("images")      #  list[ [C, H, W] ]
        batch_masks = kwargs.get("masks")
        resize_factor = kwargs.get("resize_factor")
        rfx, rfy = resize_factor[0]
        batch = len(batch_images)  #group

        # completions - > batch
        grouped_completions = []
        for index in range(batch):
            start = index * batch_masks[0].shape[0]
            end = (index + 1) * batch_masks[0].shape[0]
            group = completions[start:end]
            grouped_completions.append(group)
        completions = grouped_completions

        image = batch_images[0]
        rewards = []
        for batch_idx in range(len(batch_images)):
            sample_completions = completions[batch_idx]
            gt_mask = batch_masks[batch_idx] #1，428,640

            # reward for every completion
            pred_masks=[]
            for completion in sample_completions:
                try:
                    data = json.loads(extract_xml_answer(completion))
                except json.JSONDecodeError:
                    rewards.append(-1.0)
                    continue
                if not isinstance(data, dict):
                    rewards.append(-1.0)
                    continue

                bbox_key = None
                points_keys = []
                for key in data.keys():
                    if "bbox" in key.lower() and not bbox_key:
                        bbox_key = key  #
                    elif "point" in key.lower():
                        points_keys.append(key)  #

                if not (bbox_key and len(points_keys) >= 2):
                    rewards.append(-1.0)  #
                    continue

                check_point=False
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

                bbox = data[bbox_key]  # bbox
                points = [data[key] for key in points_keys]

                #sam2
                point1_sam2,point2_sam2 = points[0],points[1]
                point1 = [round(int(point1_sam2[0])*rfx), round(int(point1_sam2[1])*rfy)]
                point2 = [round(int(point2_sam2[0])*rfx), round(int(point2_sam2[1])*rfy)]
                points_input = [point1, point2]
                content_bbox = [round(int(bbox[0])*rfx), round(int(bbox[1])*rfy), round(int(bbox[2])*rfx), round(int(bbox[3])*rfy)]


            if len(rewards)==(batch_idx+1)*len(sample_completions): #if invalid no mask
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
                sorted_ind = torch.argsort(scores, dim=-1, descending=True)  #
                masks = masks[sorted_ind]
                pred_masks.append(masks[0])
            #pred_masks = masks[0].astype(bool)
            pred_masks = torch.stack(pred_masks, dim=0)
            pred_binary = (pred_masks > 0).int()
            gt_mask = gt_mask.int()  #B，428,640

            intersection, union, _ = intersectionAndUnionGPU(
                pred_binary.contiguous().clone(),
                gt_mask.contiguous(),
                2,
                ignore_index=255
            )

            if union[1] > 0:
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
        #pattern = r"<observe>.*?</observe>\s*<answer>.*?</answer>"
        pattern = r"<answer>.*?</answer>"
        responses = [completion for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [1.0 if match else 0.0 for match in matches]



    def seg_segmentation_format_reward(completions, **kwargs) -> list[float]:
        rewards=[]
        for responds in completions:
            #print("{}\n".format(responds))
            try:
                data = json.loads(extract_xml_answer(responds))
            except json.JSONDecodeError:
                rewards.append(-1.0)  #
                continue
            if not isinstance(data, dict):  #
                rewards.append(-1.0)  #
                continue
            bbox_key = None
            points_keys = []
            for key in data.keys():
                if "bbox" in key.lower() and not bbox_key:
                    bbox_key = key  #  "bbox"
                elif "point" in key.lower():
                    points_keys.append(key)  # points_keys "point"
            if not (bbox_key and len(points_keys) >= 2):
                rewards.append(-1.0)  #
                continue
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
        processor
    )

    training_args = GRPOConfig(
        use_vllm=False,  # use vLLM for fast inference!
        learning_rate=1e-5,  # 5e-6      #1e-5
        adam_beta1=0.9,
        adam_beta2=0.99,
        beta=0.001,  # KL coefficient，0.001
        weight_decay=0,            #0            0.01
        warmup_ratio=0.1,            # 0.1         0.03
        #scale_rewards=False,
        lr_scheduler_type="cosine",
        # lr_scheduler_type="constant",  #
        optim="adamw_torch",  # adamw_torch  adafactor
        logging_steps=1,
        bf16=True,  #
        fp16=False,
        #seed=args.seed,
        #data_seed=args.data_seed,
        per_device_train_batch_size=6,  # batch=per_device_train_batch_size/num_generations
        gradient_accumulation_steps=1,
        num_generations=6,  #
        reprompt_threshold=args.reprompt_threshold,
        max_reprompt_rounds=args.max_reprompt_rounds,
        revision_reward_weight=args.revision_reward_weight,
        revision_advantage_weight=args.revision_advantage_weight,
        max_prompt_length=None,
        #reward_weights=[1, 4, 1],
        max_completion_length=100,
        temperature=1,  #
        num_train_epochs=1,  # Set to 1 for a full training run
        max_grad_norm=1,
        # max_steps=16000,
        # save_steps=5000,
        logging_dir=str(tb_log_dir),
        output_dir=str(output_dir),
    )

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    trainer = CompatGRPOTrainer(
        model = reasoning_model,
        reward_funcs=[soft_format_reward_func,iou_reward_func,seg_segmentation_format_reward],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
        peft_config=lora_config
    )
    trainer.reprompt_segmentation_model = segmentation_model
    trainer.reprompt_torch_dtype = torch_dtype
    #trainer.train(resume_from_checkpoint="outputs/0.7-2/checkpoint-3000")

    try:
        trainer.train()
    finally:
        tb_writer.close()
        print(f"TensorBoard 日志已保存到：{tb_log_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
