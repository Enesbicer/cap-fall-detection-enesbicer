from pydantic import Field, validator
from typing import List, Optional, Union, Any, Dict, Literal
from sdks.novavision.src.base.model import Package, Input, Detection,  Output, Image, Config, Inputs, Configs, Outputs, Response, Request


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image],Image]
    type = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

    class Config:
        title = "Image"


class Detection(Detection):
    imgUID: str


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"


class DetectionInputs(Inputs):
    inputImage: InputImage


class ConfigIOUThreshold(Config):
    """
        It is necessary for Non-Maximum Suppression (NMS) which is used to filter redundant bounding boxes for the same object. A higher IOU threshold will result in fewer bounding boxes after NMS.
    """
    name: Literal["IOUThreshold"] = "IOUThreshold"
    value: float = Field(default=0.7, ge=0, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "IOU Threshold"


class ConfigConfidentThreshold(Config):
    """
        Detected Objects with a confidence score below this threshold will be ignored or filtered out.
    """
    name: Literal["ConfidentThreshold"] = "ConfidentThreshold"
    value: float = Field(default=0.3, ge=0, le=1)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Confidence Threshold"


class ConfigHalfTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ConfigHalfFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class ConfigHalf(Config):
    """
        It enables half-precision (FP16) inference, which can speed up model inference.
    """
    name: Literal["Half"] = "Half"
    value: Union[ConfigHalfTrue, ConfigHalfFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Half"


class ConfigDeviceGPU(Config):
    name: Literal["ConfigDeviceGPU"] = "ConfigDeviceGPU"
    configHalf: ConfigHalf
    value: Literal["GPU"] = "GPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "GPU"


class ConfigDeviceCPU(Config):
    name: Literal["ConfigDeviceCPU"] = "ConfigDeviceCPU"
    value: Literal["CPU"] = "CPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "CPU"


class ConfigDevice(Config):
    """
        It refers to whether the model should run on a CPU or a GPU.
        You can select the device type for inference or training process.
    """
    name: Literal["ConfigDevice"] = "ConfigDevice"
    value: Union[ConfigDeviceCPU, ConfigDeviceGPU]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Device"


class DetectionConfigs(Configs):
    configDevice: ConfigDevice
    configConfidentThreshold: ConfigConfidentThreshold
    configIOUThreshold: ConfigIOUThreshold


class DetectionOutputs(Outputs):
    outputImage: OutputImage
    outputDetections: OutputDetections


class DetectionResponse(Response):
    outputs: DetectionOutputs


class DetectionRequest(Request):
    inputs: Optional[DetectionInputs]
    configs: DetectionConfigs

    class Config:
        schema_extra = {
            "target": "configs"
        }


class DetectionExecutor(Config):
    name: Literal["FallDetection"] = "FallDetection"
    value: Union[DetectionRequest, DetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Detection"
        schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[DetectionExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        schema_extra = {
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["FallDetection"] = "FallDetection"
    uID = "1221112"