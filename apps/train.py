# -*- coding: utf-8 -*-
""" train.py """
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../'))
import requests
import json
from capsules.capsule.src.models.PackageModel import PackageModel,ConfigExecutor,PackageConfigs,TrainExecutor,TrainRequest,Path,BatchSize,TrainConfigs
from capsules.capsule.src.utils.config import Config
from capsules.capsule.src.configs.config import CFG


ENDPOINT_URL = "http://127.0.0.1:8000/api"


def train():
    config = Config.from_json(CFG)
    path =Path(value=config.data.path)
    batchSize=BatchSize(value=64)
    trainConfigs =TrainConfigs(batchSize=batchSize,configPath=path)
    trainRequest = TrainRequest(configs=trainConfigs)
    trainExecutor = TrainExecutor(value=trainRequest)
    executor = ConfigExecutor(value=trainExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    request = PackageModel(configs=packageConfigs, name="Segmentation")
    request_json = json.loads(request.json())
    response = requests.post(ENDPOINT_URL, json=request_json)
    print(response.raise_for_status())
    print(response.json())


if __name__ == '__main__':
    train()