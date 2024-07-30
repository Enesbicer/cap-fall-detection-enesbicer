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


class ConfigDrawBBoxTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ConfigDrawBBoxFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class ConfigDrawBBox(Config):
    name: Literal["DrawBBox"] = "DrawBBox"
    value: Union[ConfigDrawBBoxTrue, ConfigDrawBBoxFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Draw BBox"


class DetectionConfigs(Configs):
    configType: ConfigType
    configDrawBBox: ConfigDrawBBox

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
    name: Literal["Detection"] = "Detection"
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
    name: Literal["Detection"] = "Detection"
    uID = "1221112"
