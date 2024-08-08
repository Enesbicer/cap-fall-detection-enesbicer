import numbers

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
        title = "Images"
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


class Detection(Detection):
    imgUID: str


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"

class configTypeDetection(Config):
    name: Literal["detection"] = "detection"
    value: Literal["detection"] = "detection"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "detection"


class ConfigType(Config):
    name: Literal["configType"] = "configType"
    value: Union[configTypeDetection]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Type"


# principle 7
class DetectionInputs(Inputs):
    inputImage: InputImage

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


class ConfigDeviceGPU(Config):
    name: Literal["ConfigDeviceGPU"] = "ConfigDeviceGPU"
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

class ConfigDrawBBoxFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"

class ConfigDrawBBoxTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ConfigDrawBBox(Config):
    name: Literal["DrawBBox"] = "DrawBBox"
    value: Union[ConfigDrawBBoxTrue, ConfigDrawBBoxFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Draw BBox"


class DetectionConfigs(Configs):
    configDevice: ConfigDevice
    configDrawBBox: ConfigDrawBBox
    configConfidentThreshold: ConfigConfidentThreshold

class DetectionOutputs(Outputs):
    outputImage: OutputImage
    outputDetections: OutputDetections

# principle 6
class DetectionResponse(Response):
    outputs: DetectionOutputs


# principle 5
class DetectionRequest(Request):
    inputs: Optional[DetectionInputs]
    configs: DetectionConfigs

    class Config:
        schema_extra = {
            "target": "configs"
        }


# principle 4
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


# Principle 3
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


# Principle 2
class PackageConfigs(Configs):
    executor: ConfigExecutor


# Principle 1
class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["FallDetection"] = "FallDetection"
    uID = "1221112"