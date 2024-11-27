
from sdks.novavision.src.helper.package import PackageHelper
from capsules.FallDetection.src.models.PackageModel import PackageConfigs, ConfigExecutor, PackageModel, OutputImage, OutputDetections, DetectionOutputs, DetectionExecutor, DetectionResponse

def build_response(context):
    output_image = OutputImage(value=context.image)
    outputDetections = OutputDetections(value=context.detection)
    detectOutputs = DetectionOutputs(outputImage=output_image, outputDetections=outputDetections)
    detectResponse = DetectionResponse(outputs=detectOutputs)
    detectExecutor = DetectionExecutor(value=detectResponse)
    executor = ConfigExecutor(value=detectExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel