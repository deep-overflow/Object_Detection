from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class VOCSegmentationDataLoader:
    def __init__(self, configs):
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        
        target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        ])

        self.dataloader = {
            "train": DataLoader(
                dataset=datasets.VOCSegmentation(
                    root=configs.root,
                    year=configs.year,
                    image_set="train",
                    download=configs.download,
                    transform=transform,
                    target_transform=target_transform,
                ),
                batch_size=configs.batch_size,
                shuffle=True,
            ),
            "eval": DataLoader(
                dataset=datasets.VOCSegmentation(
                    root=configs.root,
                    year=configs.year,
                    image_set="val",
                    download=configs.download,
                    transform=transform,
                    target_transform=target_transform,
                ),
                batch_size=configs.batch_size,
                shuffle=False,
            ),
        }

    def __getitem__(self, index):
        return self.dataloader[index]