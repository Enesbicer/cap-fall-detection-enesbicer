import os
import torch
import platform
import urllib.request
from ultralytics import YOLO

weight_caption_path = '/storage/fall_model.pt'
weight_url = 'https://drive.google.com/file/d/1Q67vzqE5z1PW0gw8vm54xZK04x_mybnp/view?usp=sharing'
output_directory = '/storage/'

from sdks.novavision.src.base.download import Download

def load_models():
    model = {}
    if not os.path.exists(weight_caption_path):
        if Download.download_from_drive(weight_url, weight_caption_path) is not None:
            print(f"Model download failed")

    weight_path = weight_caption_path

    model["model"] = YOLO(weight_path)

    return model
