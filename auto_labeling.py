import os
import cv2
import torch
import argparse
import yaml
import json
import numpy as np
import glob as glob

from models.create_fasterrcnn_model import create_model
from utils.annotations import inference_annotations
from utils.general import set_infer_dir
from utils.transforms import infer_transforms, resize

def collect_all_images(dir_test):
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images    

def coco_box_format(box):
    """Convert [x1, y1, x2, y2] to [x, y, width, height]"""
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

def main(input, weights, output_json, data):
    imgsz = None
    threshold = 0.6
    device = torch.device('cuda:0')
    dtype = torch.float32
    model_name = 'fasterrcnn_resnet50_fpn'

    np.random.seed(42)
    
    with open(data) as file:
        data_configs = yaml.safe_load(file)
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']

    # Load weights
    checkpoint = torch.load(weights)
    model_fn = create_model[model_name]
    model = model_fn(num_classes=NUM_CLASSES, coco_model=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device=device, dtype=dtype).eval()

    test_images = collect_all_images(input)
    print(f"Test instances: {len(test_images)}")

    image_id = 0
    annotation_id = 0
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASSES)]
    }

    for img_path in test_images:
        image_id += 1
        image_name = os.path.basename(img_path)
        orig_image = cv2.imread(img_path)
        height, width = orig_image.shape[:2]

        if imgsz:
            RESIZE_TO = imgsz
        else:
            RESIZE_TO = width
        image_resized = resize(orig_image, RESIZE_TO)
        image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image = infer_transforms(image)
        image = torch.unsqueeze(image, 0)

        with torch.no_grad():
            outputs = model(image.to(device=device, dtype=dtype))
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        output = outputs[0]

        coco_output["images"].append({
            "id": image_id,
            "file_name": image_name,
            "height": height,
            "width": width
        })

        for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
            if score < threshold:
                continue
            x, y, w, h = coco_box_format(box.tolist())
            area = w * h
            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [x, y, w, h],
                "area": area,
                "iscrowd": 0,
                "score": float(score)
            })
            annotation_id += 1

        print(f"Processed: {image_name}")

    with open(output_json, 'w') as f:
        json.dump(coco_output, f, indent=4)
    print(f"\n Saved COCO JSON to {output_json}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input image folder')
    parser.add_argument('--weights', type=str, required=True, help='Path to trained .pt weights')
    parser.add_argument('--output', type=str, default='autolabel_coco.json', help='Output COCO json file')
    parser.add_argument('--data', type=str, default='data.yaml', help='train data yaml file')
    
    args = parser.parse_args()
    main(args.input, args.weights, args.output, args.data)
