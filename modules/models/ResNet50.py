# coding : utf-8
# Author : yuxiang Zeng

import torch
from torchvision import models, transforms
from PIL import Image
from torchvision.models.resnet import ResNet50_Weights

class Image2tensor:
    def __init__(self):
        # 加载预训练的ResNet50模型，使用最新的权重
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet50.eval()
        self.preprocess = transforms.Compose([
            # 确保图像为3通道RGB格式
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def image2tensor(self, image):
        img_tensor = self.preprocess(image)
        img_tensor = img_tensor.unsqueeze(0)  # 添加batch维度
        with torch.no_grad():
            features = self.resnet50(img_tensor)
        return features


if __name__ == '__main__':
    imageTranfer = Image2tensor()
    image = Image.open('/Users/zengyuxiang/Documents/实战代码/4.1 泰迪杯实战/datasets/附件2/ImageData/Image14001007-4042.jpg')
    features = imageTranfer.image2tensor(image)
    print('该图像嵌入表示为:', features.shape)
