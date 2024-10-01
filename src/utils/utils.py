import os
from ultralytics import YOLO
import torch
import platform

from sdks.novavision.src.base.application import Application
from sdks.novavision.src.base.download import Download


weight_fall_path = '/storage/fall_model.pt'
weight_url = 'https://drive.google.com/file/d/1Q67vzqE5z1PW0gw8vm54xZK04x_mybnp/view?usp=sharing'
output_directory = '/storage/'


def select_device(device='', batch_size=0, newline=True):
    s = f'PyTorch Python-{platform.python_version()} torch-{torch.__version__} '
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

            space = ' ' * (len(s) + 1)
            for i, d in enumerate(devices):
                device_name = torch.cuda.get_device_name(int(d))
                mem = torch.cuda.get_device_properties(int(d)).total_memory / (1 << 20)
                s += f"{'' if i == 0 else space}GPU:{d} ({device_name}, {mem:.0f}MiB)\n"

            device = torch.device(f'cuda:{devices[0]}')
        else:
            s += 'CPU\n'
            device = torch.device('cpu')

    if not newline:
        s = s.rstrip()
    print(s)
    return device

def load_models():
    models = {}
    model = {}
    application = Application()
    app_param_task = application.get_app_param("FallDetection", "ConfigExecutor")
    for i in app_param_task:
        key = str(list(i.keys())[0])
        config_device = i[key]['configs']['configDevice']['value']['value']
        device = select_device('cuda:0' if torch.cuda.is_available() else 'cpu')


        if not os.path.exists(weight_fall_path):
            if Download.download_from_drive(weight_url, weight_fall_path) is not None:
                print(f"Model download successfully: {'fall_model.pt'}")
            else:
                print(f"Model download failed: {'fall_model.pt'}")

        yolo_model = YOLO(weight_fall_path)
        yolo_model.to(device)
        model["model"] = yolo_model
        return model
