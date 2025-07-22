
import os
import torch
import platform

from ultralytics import YOLO
from sdks.novavision.src.base.download import Download
from sdks.novavision.src.base.logger import LoggerManager

weight_path = '/storage/best.pt'
weight_url = 'https://drive.google.com/file/d/1Q67vzqE5z1PW0gw8vm54xZK04x_mybnp/view?usp=sharing'
output_directory = '/storage/'

def select_device(device='', batch_size=0, newline=True):
    device = str(device).strip().lower().replace('gpu:', '').replace('none', '')
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = torch.device('cpu')
    else:
        if device:
            os.environ['CUDA_VISIBLE_DEVICES'] = device
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            devices = device.split(',') if device else [str(i) for i in range(n)]
            if n > 1 and batch_size > 0:
                assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
            device = torch.device(f'cuda:{devices[0]}')
        else:
            device = torch.device('cpu')
    return device

def load_models(config):
    model = {}
    device = select_device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model['device'] = device

    if not os.path.exists(weight_path):
        if Download.download_from_drive(weight_url, weight_path) is not None:
            print(f"Model download successfully: {'fall_model.pt'}")
        else:
            logger.error(f"Model download failed!!")

    yolo_model = YOLO(weight_path)
    yolo_model.to(device)
    model["model"] = yolo_model
    return model