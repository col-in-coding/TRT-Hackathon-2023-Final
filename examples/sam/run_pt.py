import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from collections import OrderedDict
from sam.segment_anything import SamPredictor, sam_model_registry


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='sam_models')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    model_path = Path(args.model_dir) / "sam_vit_h_4b8939.pth"
    device = "cuda"
    sam = sam_model_registry["vit_h"](checkpoint=model_path)
    sam.to(device)
    predictor = SamPredictor(sam)

    image = cv2.imread("jay.png")
    predictor.set_image(image, image_format="BGR")
    point_coords = np.array([[490, 324]])
    point_labels = np.array([1])
    masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels)

    for i, mask in enumerate(masks):
        masks = image * mask[..., None]
        cv2.imwrite(f"test{i}.png", masks)
