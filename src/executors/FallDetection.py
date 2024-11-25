
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.base.model import BoundingBox
from sdks.novavision.src.base.response import Response
from sdks.novavision.src.helper.executor import Executor
from capsules.FallDetection.src.utils.utils import load_models
from sdks.novavision.src.base.model import Image as ImageModel
from capsules.FallDetection.src.utils.response import build_response
from capsules.FallDetection.src.models.PackageModel import PackageModel, Detection


class FallDetection(Capsule):
    def __init__(self, request, bootstrap):
        self.error_list = []
        super().__init__(request)
        self.request.model = PackageModel(**(self.request.data))
        self.initialize_request_data(request, bootstrap)
        self.namedict = {"0": "Fall"}
        self.model = self.bootstrap["model"]
        self.half = self.request.get_param("Half")
        self.select_device = self.bootstrap["device"]
        self.image = self.request.get_param("inputImage")
        self.device = self.request.get_param("ConfigDevice")
        self.conf_weights = self.request.get_param("Weights")
        self.iou_thres = self.request.get_param("IOUThreshold")
        self.conf_thres = self.request.get_param("ConfidentThreshold"),

    @staticmethod
    def bootstrap() -> dict:
        model = load_models()
        return model

    def infer(self, image):
        if self.device == "GPU" and self.select_device == "cuda:0":
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
                classLabel=self.namedict[str(int(bboxes[i][5]))],
                classId=int(bboxes[i][5]),
                imgUID=img_uid)
            detection_list.append(newdetect)
        return detection_list

    def detection_inference(self, img):
        output, im = self.infer(np.array(img.value))
        output_detection_list = self.output_result(output, img.uID)
        image = ImageModel(name=img.name, uID=img.uID, mimeType=img.mimeType, encoding=img.encoding, value=img.value, type=img.type)
        return image, output_detection_list

    def run(self):
        self.image = Image.get_frame(img=self.image, redis_db=self.redis_db)
        self.image, self.detection = self.detection_inference(self.image)
        self.image = Image.set_frame(img=self.image, package_uID=self.uID, redis_db=self.redis_db)
        packageModel = build_response(context=self)
        return Response(model=packageModel, bootstrap=self.bootstrap).response()


if "__main__" == __name__:
    Executor(sys.argv[1]).run()