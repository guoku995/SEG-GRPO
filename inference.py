import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from peft import PeftModel
from PIL import Image as PILImage
import re
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="outputs/checkpoint-16994")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")  #The person facing away，The restaurant that serves roast duck
    parser.add_argument("--text", type=str, default="The person facing away")  #
    parser.add_argument("--image_path", type=str, default="./assets/COCO_train2014_000000007104.jpg")
    parser.add_argument("--output_path", type=str, default="./assets/test_output.jpg")
    return parser.parse_args()

def extract_bbox_points_think(output_text, x_factor, y_factor):
    json_pattern = r'{[^}]+}'
    json_match = re.search(json_pattern, output_text)

    content_bbox, points = None, None

    if json_match:
        try:
            data = json.loads(json_match.group(0))

            # bbox
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



def main():
    args = parse_args()

    processor = AutoProcessor.from_pretrained("outputs/checkpoint-100000-refcoco",
                                              use_fast=False,
                                              padding_side="left")
    tokenizer = processor.tokenizer

    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "evfsamconfig/qwen2.5_3B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    reasoning_model.resize_token_embeddings(len(tokenizer))
    for param in reasoning_model.parameters():
        param.requires_grad = False
    reasoning_model = PeftModel.from_pretrained(reasoning_model, "outputs/checkpoint-100000-refcoco")

    segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)

    print("User question: ", args.text)

    QUESTION_TEMPLATE = \
        "Find '{Question}'." \
        "The image size is 840*840, locate the most closely matched one." \
        "Directly output the answer with one bbox and two points inside the interested object in <answer> </answer> tags. i.e., <answer>{Answer}</answer> in JSON format." \

    image = PILImage.open(args.image_path)
    original_width, original_height = image.size
    resize_size = 840
    x_factor, y_factor = original_width / resize_size, original_height / resize_size

    messages = []
    message = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=args.text.lower().strip("."),
                                                 Answer="{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180]}")
            }
        ]
    }]
    messages.append(message)

    # Preparation for inference
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    bbox, points = extract_bbox_points_think(output_text[0], x_factor, y_factor)

    print("Answer: ",  bbox, points)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        segmentation_model.set_image(image)
        masks, scores, _ = segmentation_model.predict(
            point_coords=points,
            point_labels=[1, 1],
            box=bbox
        )
        #sorted_ind = np.argsort(scores)[::-1]
        #sorted_ind = np.argsort(scores.cpu().numpy())[::-1]
        sorted_ind = torch.argsort(scores, descending=True)
        masks = masks[sorted_ind]

    mask = masks[0].cpu().numpy().astype(bool)

    plt.figure(dpi=300)  # Set high resolution output
    plt.imshow(image, alpha=0.9)
    mask_overlay = np.zeros_like(image)
    mask_overlay[mask == 1] = [255, 0, 0]  # Red mask
    plt.imshow(mask_overlay, alpha=0.4)
    plt.axis('off')  # Hide axes for cleaner output
    plt.tight_layout()
    plt.savefig(args.output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
