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
    Google Drive'dan indirilen FallDetection.pt modelini yükler,
    kullanıcıdan gelen config ayarlarına göre cihaz, FP16, confidence ve IOU threshold ayarlarını uygular.
    """
    application = Application()

    # Config'ten parametreleri al
    device_type = application.get_param(config=config, name="ConfigDevice")          # "CPU" ya da "GPU"
    use_half = application.get_param(config=config, name="Half")                    # True / False
    conf_thres = application.get_param(config=config, name="ConfidenceThreshold")   # örn: 0.3
    iou_thres = application.get_param(config=config, name="IoUThreshold")           # örn: 0.5

    # Model yolunu hazırla ve indir (varsa atla)
    weight_path = download_from_drive_if_not_exists(
        url="https://drive.google.com/uc?export=download&id=1VXo-WvftAC8M7yWcu0Ysewvl7zmYFP20",
        filename="FallDetection.pt"
    )

    # Cihaz seçimi (GPU varsa ve seçilmişse GPU kullanılır)
    device = select_device('cuda:0' if device_type == "GPU" and torch.cuda.is_available() else 'cpu')

    # Modeli yükle ve cihaza gönder
    model = YOLO(weight_path).to(device)

    # FP16 desteği gerekiyorsa uygula
    if use_half and device.type != 'cpu':
        model.fuse()
        model = model.half()

    # Kullanıcıdan gelen threshold değerlerini uygula
    model.overrides['conf'] = conf_thres
    model.overrides['iou'] = iou_thres

    logger.info(f"Model yüklendi: {weight_path} | Cihaz: {device} | FP16: {use_half}")
    return model, device


def download_from_drive_if_not_exists(url: str, filename: str, storage_dir="/storage"):
    """
    Google Drive'dan model dosyasını indirir (eğer daha önce indirilmemişse).
    URL doğrudan uc?id=... formatında olmalı.
    """
    path = Path(storage_dir) / filename
    if path.exists():
        logger.info(f"Model zaten mevcut: {path}")
        return str(path)

    try:
        logger.info(f"Model indiriliyor: {filename}")
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(path))
        logger.info(f"Model başarıyla indirildi: {filename}")
        return str(path)
    except Exception as e:
        logger.error(f"FallDetection - Model indirilemedi: {filename} | Hata: {e}")
        raise e
