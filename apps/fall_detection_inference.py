import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../'))
import requests
import cv2
import numpy as np
import json
from sdks.novavision.src.media.image import Image as image

from sdks.novavision.src.base.model import Image, Request

from capsules.FallDetection.src.models.PackageModel import PackageModel, PackageConfigs, DetectionConfigs, \
    DetectionInputs, \
    DetectionExecutor, DetectionRequest, ConfigType, InputImage, configTypeDetection, ConfigExecutor, \
    ConfigDrawBBoxTrue, ConfigDrawBBox, ConfigConfidentThreshold, ConfigDeviceCPU, ConfigDevice, ConfigHalfTrue, \
    ConfigHalf, ConfigDeviceGPU, ConfigIOUThreshold

ENDPOINT_URL = "http://127.0.0.1:8000/api"


def inference():
    image_data = Image(name="image", uID="323332", mimeType="image/jpg", encoding="base64", value=np.asarray(
        cv2.imread('capsules/FallDetection/resources/fall_people.jpg')).astype(np.float32),
                       type="Image")
    image_data = image.encode64(image_data)

    confidentThreshold = ConfigConfidentThreshold(value=0.65)
    param_ConfigDrawBBox = ConfigDrawBBoxTrue(value=True)
    configDrawBBox = ConfigDrawBBox(value=param_ConfigDrawBBox)
    configIOUThreshold = ConfigIOUThreshold(value=0.7)

    #configHalfTrue = ConfigHalfTrue(value=True)
    #configHalf = ConfigHalf(value=configHalfTrue)
    #configDeviceGPU = ConfigDeviceGPU(configHalf=configHalf)
    #configDevice = ConfigDevice(value=configDeviceGPU)

    configDeviceCPU = ConfigDeviceCPU(value="CPU")
    configDevice = ConfigDevice(value=configDeviceCPU)
    detectionConfigs = DetectionConfigs(configConfidentThreshold=confidentThreshold, configDevice=configDevice, configDrawBBox=configDrawBBox, configIOUThreshold=configIOUThreshold)
    inputImage = InputImage(value=image_data)
    detectionInputs = DetectionInputs(inputImage=inputImage)
    detectionRequest = DetectionRequest(inputs=detectionInputs, configs=detectionConfigs)
    detectionExecutor = DetectionExecutor(value=detectionRequest)
    executor = ConfigExecutor(value=detectionExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    request = PackageModel(configs=packageConfigs, name="FallDetection")
    request_json = json.loads(request.json())
    response = requests.post(ENDPOINT_URL, json=request_json)

    print(response.raise_for_status())
    print(response.json())

if __name__ =="__main__":
    inference()