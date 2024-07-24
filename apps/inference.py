import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../'))
import requests
import cv2
import numpy as np
import json
from sdks.novavision.src.media.image import Image as image

from sdks.novavision.src.base.model import Image, Request
from capsules.capsule.src.utils.config import Config
from capsules.capsule.src.configs.config import CFG

from capsules.capsule.src.models.PackageModel import PackageModel,PackageConfigs,SegmentationConfigs,SegmentationInputs,SegmentationExecutor,SegmentationRequest,ConfigType,InputImage,configTypeSegmentation,ConfigExecutor


ENDPOINT_URL = "http://127.0.0.1:8000/api"


def inference():
    config = Config.from_json(CFG)
    image_data =Image(name="image", uID="323332", mimeType="image/jpg", encoding="base64",value =image.encode64(np.asarray(cv2.imread(config.project.path +'/capsules/capsule/resources/yorkshire_terrier.jpg')).astype(np.float32),'image/jpg'), type="Image")
    segmentation = configTypeSegmentation()
    configTypevalue = ConfigType(value=segmentation)
    segmentationConfigs = SegmentationConfigs(configType=configTypevalue)
    inputImage = InputImage(value=image_data)
    segmentationInputs = SegmentationInputs(inputImage=inputImage)
    segmentationRequest = SegmentationRequest(inputs=segmentationInputs, configs=segmentationConfigs)
    segmentationExecutor = SegmentationExecutor(value=segmentationRequest)
    executor = ConfigExecutor(value=segmentationExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    request = PackageModel(configs=packageConfigs, name="Segmentation")
    request_json = json.loads(request.json())
    response = requests.post(ENDPOINT_URL, json =request_json)
    print(response.raise_for_status())
    print(response.json())



if __name__ =="__main__":
    inference()