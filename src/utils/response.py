
from sdks.novavision.src.helper.package import PackageHelper
from capsules.FallDetection.src.models.PackageModel import PackageConfigs, ConfigExecutor, PackageModel, OutputDetections, DetectionOutputs, DetectionExecutor, DetectionResponse

def build_response(context):
    outputDetections = OutputDetections(value=context.detection)
    detectOutputs = DetectionOutputs(outputDetections=outputDetections)
    detectResponse = DetectionResponse(outputs=detectOutputs)
    detectExecutor = DetectionExecutor(value=detectResponse)
    executor = ConfigExecutor(value=detectExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel