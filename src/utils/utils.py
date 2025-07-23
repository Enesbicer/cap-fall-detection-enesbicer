import os
import torch
import urllib.request
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device

from sdks.novavision.src.base.logger import LoggerManager
from sdks.novavision.src.base.application import Application

logger = LoggerManager()


def load_model(config: dict):
    """
        Loads the FallDetection.pt model downloaded from Google Drive,
        and applies device, FP16, confidence, and IOU threshold settings according to the user-provided config.
    """
    application = Application()

    # Retrieve parameters from the config.
    device_type = application.get_param(config=config, name="ConfigDevice")          # "CPU" or "GPU"
    use_half = application.get_param(config=config, name="Half")                    # True / False
    conf_thres = application.get_param(config=config, name="ConfidenceThreshold")
    iou_thres = application.get_param(config=config, name="IoUThreshold")

    # Prepare the model path and download it (skip if already available).
    weight_path = download_from_drive_if_not_exists(
        url="https://drive.google.com/uc?export=download&id=1VXo-WvftAC8M7yWcu0Ysewvl7zmYFP20",
        filename="FallDetection.pt"
    )

    # Device selection (use GPU if available and selected).
    device = select_device('cuda:0' if device_type == "GPU" and torch.cuda.is_available() else 'cpu')

    # Load the model and send it to the device.
    model = YOLO(weight_path).to(device)

    # Apply FP16 support if required.
    if use_half and device.type != 'cpu':
        model.fuse()
        model = model.half()

    # Apply the threshold values received from the user.
    model.overrides['conf'] = conf_thres
    model.overrides['iou'] = iou_thres

    logger.info(f"Model loaded: {weight_path} | Device: {device} | FP16: {use_half}")
    return model, device


def download_from_drive_if_not_exists(url: str, filename: str, storage_dir="/storage"):
    """
        Downloads the model file from Google Drive (if it hasn't been downloaded before).
    """
    path = Path(storage_dir) / filename
    if path.exists():
        logger.info(f"Model already exists: {path}")
        return str(path)

    try:
        logger.info(f"Model is downloading: {filename}")
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(path))
        logger.info(f"Model was downloaded successfully: {filename}")
        return str(path)
    except Exception as e:
        logger.error(f"FallDetection - Model could not be downloaded: {filename} | Hata: {e}")
        raise e
