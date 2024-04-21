# coding : utf-8
# Author : yuxiang Zeng
import os
from PIL import Image
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from utils.config import get_config
from utils.utils import set_seed
from tqdm import *
import pandas as pd

class ZyxDataset(Dataset):
    def __init__(self, image_tensor, text_tensor):
        self.image_tensor = image_tensor
        self.text_tensor = text_tensor
    def __len__(self):
        return len(self.image_tensor)

    def __getitem__(self, idx):
        image_tensor = self.image_tensor[idx]
        text_tensor = self.text_tensor[idx]
        return image_tensor, text_tensor

def train():
    # 创建模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, preprocess = load_from_name("ViT-B-16", device=device, download_root='../BigDataSource/')

    optimizer = optim.Adam(model.parameters(), lr=1e-6, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 创建损失函数
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    # 加载数据集
    your_dataset = YourDataset(img_root='/images', meta_root='/meta', is_train=True, preprocess=preprocess)
    dataset_size_your = len(your_dataset)
    your_dataloader = DataLoader(your_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=False)

    phase = "train"
    model_name = "CLIP_Teddy"
    ckt_gap = 4
    for epoch in range(args.epoch):
        scheduler.step()
        total_loss = 0
        batch_num = 0
        with torch.cuda.amp.autocast(enabled=True):
            for images, label_tokens in your_dataloader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                batch_num += 1
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    logits_per_image, logits_per_text = model(images, label_tokens)
                    ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
                    cur_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                    total_loss += cur_loss
                    if phase == "train":
                        cur_loss.backward()
                        if device == "cpu":
                            optimizer.step()
                        else:
                            optimizer.step()
                            clip.model.convert_weights(model)
                if batch_num % 4 == 0:
                    print('{} epoch:{} loss:{}'.format(phase, epoch, cur_loss))
            epoch_loss = total_loss / batch_num
            torch.save(model.state_dict(), f"{model_name}_epoch_{epoch}.pth")
            print(f"weights_{epoch} saved")
            if epoch % ckt_gap == 0:
                checkpoint_path = f"{model_name}_ckt.pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}
                torch.save(checkpoint, checkpoint_path)
                print(f"checkpoint_{epoch} saved")
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))


if __name__ == '__main__':
    args = get_config()
    set_seed(2024)
    model, preprocess = load_from_name("ViT-B-16", download_root='../BigDataSource/')
    try:
        with open('../BigDataSource/Teddy2024/附件1/image_tensor.pkl', 'rb') as f:
            image_tensor = pickle.load(f)
        with open('../BigDataSource/Teddy2024/附件1/text_tensor.pkl', 'rb') as f:
            text_tensor = pickle.load(f)
    except:
        # 直接获取张量
        all_image = os.listdir('../BigDataSource/Teddy2024/附件1/ImageData')
        all_text = pd.read_csv('../BigDataSource/Teddy2024/附件1/ImageWordData.csv').to_numpy()[:, 1]
        image_tensor = []
        text_tensor = []
        for i in trange(len(all_image)):
            image_address = '../BigDataSource/Teddy2024/附件1/ImageData/'
            file_name = image_address + all_image[i]
            raw_image = Image.open(file_name)
            image_tensor.append(preprocess(raw_image))

            raw_text = all_text[i]
            text_tensor.append(clip.tokenize(raw_text))

        image_tensor = torch.stack(image_tensor)
        text_tensor = torch.stack(text_tensor).squeeze(1)
        # print(image_tensor.shape, text_tensor.shape)
        with open('../BigDataSource/Teddy2024/附件1/image_tensor.pkl', 'wb') as f:
            pickle.dump(image_tensor, f)
        with open('../BigDataSource/Teddy2024/附件1/text_tensor.pkl', 'wb') as f:
            pickle.dump(text_tensor, f)
        print('Raw tensor loaded!')

    # logits_per_image, logits_per_text, _ = model.forward(image_tensor, text_tensor)

    # 进入模型训练部分
    dataset = ZyxDataset(image_tensor[:10], text_tensor[:10])
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=False)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-6, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)

    for epoch in range(args.epochs):
        model.train()
        for train_batch in tqdm(train_loader):
            image_tensor, text_tensor = train_batch
            # 获取相似性
            image_features, text_features, _ = model.forward(image_tensor, text_tensor)
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            ground_truth = torch.arange(len(image_tensor), dtype=torch.long)

            # 微调
            optimizer.zero_grad()
            cur_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            cur_loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), f"./model_epoch.pth")
        model.eval()
