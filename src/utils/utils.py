import os
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from sdks.novavision.src.base.download import Download
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
    device_type = application.get_param(config=config, name="ConfigDevice")  # "CPU" or "GPU"
    use_half = application.get_param(config=config, name="Half")  # True / False

    # Prepare the model path and download it (skip if already available).
    weight_path = download_from_drive_if_not_exists(
        url="https://drive.google.com/file/d/1VXo-WvftAC8M7yWcu0Ysewvl7zmYFP20/view?usp=sharing",
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

    logger.info(f"Model loaded: {weight_path} | Device: {device} | FP16: {use_half}")
    return model, device


def download_from_drive_if_not_exists(url: str, filename: str, storage_dir="/storage"):
    """
        Downloads the model file from Google Drive using Download SDK (if it hasn't been downloaded before).
    """
    path = Path(storage_dir) / filename

    if path.exists():
        logger.info(f"Model already exists: {path}")
        return str(path)

    try:
        logger.info(f"Model is downloading: {filename}")
        Path(storage_dir).mkdir(parents=True, exist_ok=True)


        download_result = Download.download_from_drive(url, str(path))

        if download_result is not None:
            logger.info(f"Model was downloaded successfully: {filename}")
            return str(path)
        else:
            logger.error(f"FallDetection - Model could not be downloaded: {filename}")
            raise Exception(f"Download failed for {filename}")

    except Exception as e:
        logger.error(f"FallDetection - Model could not be downloaded: {filename} | Hata: {e}")
        raise e