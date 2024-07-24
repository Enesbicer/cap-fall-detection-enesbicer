import numbers

from pydantic import Field, validator
from typing import List, Optional, Union, Any, Dict,Literal

from sdks.novavision.src.base.model import Package,Input, Output, Image, Config, Inputs, Configs, Outputs, Response, Request


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
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
        title = "Images"

class OutputData(Output):
    name: Literal["outputData"] = "outputData"
    value: List
    type: Literal["list"] = "list"


class configTypeSegmentation(Config):
    name: Literal["segmentation"] = "segmentation"
    value: Literal["segmentation"] = "segmentation"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Segmentation"


class ConfigType(Config):
    name: Literal["configType"] = "configType"
    value: Union[configTypeSegmentation]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Type"


class SegmentationInputs(Inputs):
    inputImage: InputImage



class SegmentationConfigs(Configs):
    configType: ConfigType



class SegmentationOutputs(Outputs):
    outputData: OutputData



class SegmentationRequest(Request):
    inputs: Optional[SegmentationInputs]
    configs: SegmentationConfigs
    class Config:
        schema_extra = {
            "target": "configs"
        }


class SegmentationResponse(Response):
    outputs: SegmentationOutputs



class SegmentationExecutor(Config):
    name: Literal["Segmentation"] = "Segmentation"
    value: Union[SegmentationRequest, SegmentationResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Segmentation"
        schema_extra = {
            "target": {
                "value": 0
            }
        }


class BatchSize(Config):
    name: Literal["BatchSize"] = "BatchSize"
    value: int = Field(ge=1, le=100)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Batch Size"


class Path(Config):
    name: Literal["path"] = "path"
    value: str
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Path"

class TrainConfigs(Configs):
    configPath: Path
    batchSize: BatchSize

class TrainOutputs(Outputs):
    outputData: OutputData

class TrainRequest(Request):
    configs: TrainConfigs

    class Config:
        schema_extra = {
            "target": "configs"
        }


class TrainResponse(Response):
    outputs: TrainOutputs


class TrainExecutor(Config):
    name: Literal["Train"] = "Train"
    value: Union[TrainRequest, TrainResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Train"
        schema_extra = {
            "target": {
                "value": 0
            }
        }



class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[SegmentationExecutor,TrainExecutor]
    type:Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["Segmentation"] = "Segmentation"
    uID = "1221112"