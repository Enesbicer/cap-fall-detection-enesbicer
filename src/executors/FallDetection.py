import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from datetime import datetime
import torch
import requests
import json
import uuid
import urllib.request
import hashlib
from ultralytics import YOLO
from sdks.novavision.src.base.response import Response
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.base.model import Image as ImageModel
from sdks.novavision.src.base.model import BoundingBox

from capsules.FallDetection.src.models.PackageModel import PackageConfigs, ConfigExecutor, PackageModel, OutputImage, \
    OutputDetections, DetectionOutputs, DetectionExecutor, DetectionResponse, Detection

from capsules.FallDetection.src.utils.utils import load_models, select_device


class FallDetection(Capsule):
    def __init__(self, request, bootstrap={}):
        self.error_list = []
        super().__init__(request)
        self.start_time = datetime.now()
        self.debug = self.request.data.get("debug", False)
        self.request.model = PackageModel(**(self.request.data))
        self.conf_weights = self.request.get_param("Weights")
        self.bootstrap = bootstrap
        self.application = self.bootstrap["application"]
        self.model = self.bootstrap["model"]
        self.mode = self.application.get_app_mode()
        self.draw_bbox = self.request.get_param("DrawBBox")
        self.device = self.request.get_param("ConfigDevice")
        self.conf_thres = self.request.get_param("ConfidentThreshold")
        self.iou_thres = self.request.get_param("IOUThreshold")
        self.images = self.request.get_param("inputImage")
        self.is_list = Image.is_list(self.images)
        self.namedict = {"0": "Fall"}
        self.select_device = select_device()

    @staticmethod
    def bootstrap():
        model = load_models()
        return model

    def infer(self, image):
        if str(self.device).lower() == "gpu" and self.select_device == "cuda:0":
            self.half = self.request.get_param("Half")
            if self.half:
                output = self.model.predict(image, conf=self.conf_thres, iou=self.iou_thres, half=True)
            else:
                output = self.model.predict(image, conf=self.conf_thres, iou=self.iou_thres)
        else:
            output = self.model.predict(image, conf=self.conf_thres, iou=self.iou_thres)
        im = output[0].plot()

        return output, im

    def output_result(self, output, img_uid):
        output = output[0].cpu().numpy()
        bboxes = output.boxes.data
        detection_list = []
        for i in range(0, len(bboxes)):
            bbox = BoundingBox(
                left=bboxes[i][0], top=bboxes[i][1],
                width=bboxes[i][2] - bboxes[i][0],
                height=bboxes[i][3] - bboxes[i][1])
            newdetect = Detection(
                boundingBox=bbox, confidence=bboxes[i][4],
                detectedLabel=self.namedict[str(int(bboxes[i][5]))],
                imgUID=img_uid)
            detection_list.append(newdetect)
        return detection_list

    def detection_inference(self, img):
        output, im = self.infer(np.array(img.value))

        if int(self.draw_bbox) == True:

            img.value = im
            # img.value = Image.encode64(im, img.mimeType)
        else:
            # img.value = Image.encode64(img.value, img.mimeType)
            pass

        output_detection_list = self.output_result(output, img.uID)

        #image = ImageModel(name=img.name, uID=img.uID, mimeType=img.mimeType, encoding=img.encoding, value=img.value, type=img.type)

        return img, output_detection_list

    def run(self):
        if self.is_list:
            data_imgs = []
            detection_list = []
            for img in self.images:
                img = Image.get_image(img, self.debug)
                img, detects = self.detection_inference(img)
                detection_list.extend(detects)
                data_imgs.append(img)
            output_detection_list = detection_list
            imageList = data_imgs
        else:
            img = Image.get_image(img=self.images, bootstrap=self.bootstrap)
            img, detects = self.detection_inference(img)
            output_detection_list = detects

            a = Image.encode64(img)

            imageList = Image.set_image(img=img, package_uID=self.request.model.uID, bootstrap=self.bootstrap)


        output_image = OutputImage(value=imageList)
        outputDetections = OutputDetections(value=output_detection_list)
        detectOutputs = DetectionOutputs(outputImage=output_image, outputDetections=outputDetections)
        detectResponse = DetectionResponse(outputs=detectOutputs)
        detectExecutor = DetectionExecutor(value=detectResponse)
        executor = ConfigExecutor(value=detectExecutor)
        packageConfigs = PackageConfigs(executor=executor)
        packageModel = PackageModel(configs=packageConfigs)

        print(f"Start :", self.start_time.strftime("%Y-%m-%d %H:%M:%S:%f")[:-3])
        self.now = datetime.now()
        print(f"Stop :", self.now.strftime("%Y-%m-%d %H:%M:%S:%f")[:-3], '\n\n')

        return Response(model=packageModel, bootstrap=self.bootstrap, debug=self.debug).response()

if "__main__" == __name__:
    from sdks.novavision.src.base.application import Application
    from sdks.novavision.src.base.environment import Environment
    from sdks.novavision.src.base.redis import MqttClient

    application = Application()
    environment = Environment()
    mqtt_client = MqttClient(application=application, environment=environment)
    mqtt_client._subscribe(sys.argv[1])
