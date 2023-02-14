import torch
import torch.optim as optim
import matplotlib.pyplot as plt

def get_optimizer(configs, params):
    if configs.name == "SGD":
        return optim.SGD(params=params, lr=configs.lr)

def get_lr_scheduler(configs, optimizer):
    if configs.name == "ExponentialLR":
        return optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=configs.gamma)

def mask2image(mask):
    colormap = torch.tensor([[0, 0, 85],
                             [0, 0, 170],
                             [0, 0, 255],
                             [0, 85, 85],
                             [0, 85, 170],
                             [0, 85, 255],
                             [0, 170, 85],
                             [0, 170, 170],
                             [0, 170, 255],
                             [0, 255, 85],
                             [0, 255, 170],
                             [0, 255, 255],
                             [85, 170, 85],
                             [85, 170, 170],
                             [85, 170, 255],
                             [85, 255, 85],
                             [85, 255, 170],
                             [85, 255, 255],
                             [170, 0, 85],
                             [170, 0, 170]])
    
    image = colormap[mask]
    return image.squeeze()

def save_tensor_img(tensor, file_name):
    img = tensor.reshape(256, 256, 3).numpy()
    plt.imsave(file_name, img)