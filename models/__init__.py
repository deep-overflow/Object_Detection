from .UNet import UNetSegmentationModel
from .DeepLab import DeepLabSegmentationModel

def get_model(configs):
    if configs.name == "UNet":
        return UNetSegmentationModel(configs)
    elif configs.name == "DeepLab":
        return DeepLabSegmentationModel(configs)