from .PascalVOC import VOCSegmentationDataLoader

def get_dataloader(configs):
    if configs.name == "VOCSegmentationDataLoader":
        return VOCSegmentationDataLoader(configs)