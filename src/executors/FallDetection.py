import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.base.model import BoundingBox
from sdks.novavision.src.base.response import Response
from sdks.novavision.src.helper.executor import Executor
from capsules.FallDetection.src.utils.utils import load_model
from sdks.novavision.src.base.model import Image as ImageModel
from capsules.FallDetection.src.utils.response import build_response
from capsules.FallDetection.src.models.PackageModel import PackageModel, Detection


class FallDetection(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**self.request.data)
        self.namedict = {"0": "Fall"}

        self.device_str = self.request.get_param("ConfigDevice")
        self.device = self.bootstrap.get("device")
        self.model = self.bootstrap.get("model")
        self.conf_thres = self.request.get_param("ConfidentThreshold")
        self.iou_thres = self.request.get_param("IOUThreshold")
        self.half = self.request.get_param("Half")
        self.image = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        model, device = load_model(config=config)
        return {
            "model": model,
            "device": device,
        }

    def infer(self, image):
        return self.model.predict(image, conf=self.conf_thres, iou=self.iou_thres)

    def output_result(self, output, img_uid):
        output = output[0].cpu().numpy()
        bboxes = output.boxes.data
        detection_list = []

        for i in range(len(bboxes)):
            bbox = BoundingBox(
                left=bboxes[i][0],
                top=bboxes[i][1],
                width=bboxes[i][2] - bboxes[i][0],
                height=bboxes[i][3] - bboxes[i][1]
            )
            new_detect = Detection(
                boundingBox=bbox,
                confidence=bboxes[i][4],
                classLabel=self.namedict[str(int(bboxes[i][5]))],
                classId=int(bboxes[i][5]),
                imgUID=img_uid
            )
            detection_list.append(new_detect)
        return detection_list

    def detection_inference(self, img):
        output = self.infer(np.array(img.value))
        return self.output_result(output, img.uID)

    def run(self):
        self.image = Image.get_frame(img=self.image, redis_db=self.redis_db)
        if not self.image:
            return Response(context=self).response()
        self.detection = self.detection_inference(self.image)
        packageModel = build_response(context=self)
        return packageModel


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
