# coding : utf-8
# Author : yuxiang Zeng
# 本文件用于模型构建，测验模型准确度，为问题一问题二作准备
import os
import torch
import pickle
from tqdm import *
import numpy as np
import pandas as pd
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from eval_recall import get_performance
from utils.config import get_config
from utils.utils import set_seed
from time import time


class Raw2Vector:
    def __init__(self, image_model, text_model, args):
        self.args = args
        print("Available models:", available_models())
        # Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = load_from_name("ViT-B-16", device=device, download_root='../BigDataSource/')
        self.model.eval()

    def image2tensor(self, image):
        image = image.unsqueeze(0).to(self.args.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def text2tensor(self, text):
        # text = text.unsqueeze(0).to(self.args.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def retrieve(self, query, database):
        # query = query.unsqueeze(0).to(self.args.device)
        query, database = query.to('cuda'), database.to('cuda')
        logit_scale = self.model.logit_scale.exp().to('cuda')
        logits_per_query = logit_scale * query @ database.t()
        probs = logits_per_query.softmax(dim=-1).cpu().detach().numpy()
        return probs


def get_all_image():
    # 本地化保存
    try:
        with open('../BigDataSource/Teddy2024/附件1/image_features_cpu.pkl', 'rb') as f:
            all_image_features = pickle.load(f)
            # torch.load(f, map_location='cpu')
    except:
        # 获得附件1数据集代码
        all_image = os.listdir('../BigDataSource/Teddy2024/附件1/ImageData')

        # 获得预训练模型
        transfer = Raw2Vector('ViT-B-16', '1', args)

        # 准备用多线程代码迅速获得所有图像的张量
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def function(inputs):
            image_name = inputs
            image_address = '../BigDataSource/Teddy2024/附件1/ImageData/'
            file_name = image_address + image_name
            raw_image = Image.open(file_name)
            image_tensor = transfer.preprocess(raw_image)
            image_features = transfer.image2tensor(image_tensor)
            return image_name, image_features

        input_list = [image_name for image_name in all_image]
        all_image_features = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(function, inputs) for inputs in input_list]
            for future in tqdm(as_completed(futures), total=len(all_image)):
                image_name, image_features = future.result()
                all_image_features.append([image_name, image_features.cpu()])
        with open('../BigDataSource/Teddy2024/附件1/image_features_cpu.pkl', 'wb') as f:
            pickle.dump(all_image_features, f)
        print('图像数据预训练并存储完毕!')
    return all_image_features

def get_all_text():
    # 本地化保存
    try:
        with open('../BigDataSource/Teddy2024/附件1/text_features_cpu.pkl', 'rb') as f:
            all_text_features = pickle.load(f)
    except:
        # 获得附件1数据集代码
        all_text = pd.read_csv('../BigDataSource/Teddy2024/附件1/ImageWordData.csv').to_numpy()[:, 1]

        # 获得预训练模型
        transfer = Raw2Vector('ViT-B-16', '1', args)

        # 准备用多线程代码迅速获得所有文本的张量
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def function(inputs):
            raw_text = inputs
            text_tensor = clip.tokenize(raw_text).to(args.device)
            text_features = transfer.text2tensor(text_tensor)
            return raw_text, text_features

        input_list = [raw_text for raw_text in all_text]
        all_text_features = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(function, inputs) for inputs in input_list]
            for future in tqdm(as_completed(futures), total=len(all_text)):
                raw_text, text_features = future.result()
                all_text_features.append([raw_text, text_features.cpu()])
        with open('../BigDataSource/Teddy2024/附件1/text_features_cpu.pkl', 'wb') as f:
            pickle.dump(all_text_features, f)
        print('文本数据预训练并存储完毕!')
    return all_text_features


def numpy_top_k_indices(matrix, k, axis=1):
    if axis == 1:  # 处理每行
        k = min(k, matrix.shape[1])
        indices = np.argsort(matrix, axis=1)[:, -k:][:, ::-1]
    elif axis == 0:  # 处理每列
        k = min(k, matrix.shape[0])
        indices = np.argsort(matrix, axis=0)[-k:, :][::-1, :]
    return indices

def high_speed_retreive(database, query, model, k):
    if isinstance(database, torch.Tensor):
        database = database.cpu()
    if isinstance(query, torch.Tensor):
        query = query.cpu()
    # Group truth 自己检索自己，除了自己以外的排名，故 + 1
    from modules.models.Retrieve import LSH, L2Index
    d = 512  # 向量维度
    if model == 'L2':
        print('执行L2检索部分')
        t1 = time()
        l2index = L2Index(k=k, d=d)
        topk_distence, topk_indices = l2index.search_topk_embeds(database, query)
        t2 = time()
        print(f'L2: {t2 - t1 : .2f}s')
    elif model == 'KNN':
        print('执行KNN检索部分')
        probs = transfer.retrieve(database, query)
        t1 = time()
        topk_indices = numpy_top_k_indices(probs, k, 1)
        t2 = time()
        print(f'KNN: {t2 - t1 : .2f}s')
    elif model == 'LSH':
        print('执行LSH检索部分')
        t1 = time()
        lsh = LSH(k=k, d=d, nbits=2048)
        topk_distence, topk_indices = lsh.search_topk_embeds(database, query)
        t2 = time()
        print(f'LSH: {t2 - t1 : .2f}s')
    return topk_indices

def show_results(database, query):
    # ans = []
    # for i in range(len(all_pred_rank)):
    #     print('-' * 80)
    #     print(all_image_features[i][0])
    #     for j in range(len(all_pred_rank[i])):
    #         print(all_text_features[all_pred_rank[i][j]][0])
    #         if j > 5:
    #             break
    #     # break
    #     if i >= 15:
    #         break
    #
    # print(all_real_rank)
    # print(all_pred_rank)
    pass


if __name__ == '__main__':
    args = get_config()
    set_seed(2024)
    data = pd.read_csv('../BigDataSource/Teddy2024/附件1/ImageWordData.csv').to_numpy()
    transfer = Raw2Vector('ViT-B-16', '1', args)

    # 是否数据清洗
    # preprocess() "ImageWordData_new"

    # 首先获得所有的特征
    all_image_features = get_all_image()
    all_text_features = get_all_text()

    # 修正序号
    try:
        with open('../BigDataSource/Teddy2024/附件1/image_features_final.pkl', 'rb') as f:
            all_image_features = pickle.load(f)
        with open('../BigDataSource/Teddy2024/附件1/text_features_final.pkl', 'rb') as f:
            all_text_features = pickle.load(f)
    except:
        all_image_idx = []
        all_text_idx = []
        for i in range(len(all_text_features)):
            all_image_idx.append(all_image_features[i][0])
            all_text_idx.append(all_text_features[i][0])
        new_image_features = []
        new_text_features = []
        for i in trange(len(data)):
            image_idx = all_image_idx.index(data[i][0])
            text_idx = all_text_idx.index(data[i][1])
            new_image_features.append([data[i][0], data[i][1], all_image_features[image_idx][1]])
            new_text_features.append([data[i][0], data[i][1], all_text_features[text_idx][1]])
        all_image_features = new_image_features
        all_text_features = new_text_features
        with open('../BigDataSource/Teddy2024/附件1/image_features_final.pkl', 'wb') as f:
            pickle.dump(all_image_features, f)
        with open('../BigDataSource/Teddy2024/附件1/text_features_final.pkl', 'wb') as f:
            pickle.dump(all_text_features, f)

    # 直接获取张量
    image_features = []
    for i in range(len(all_image_features)):
        image_features.append(all_image_features[i][2])
    image_features = torch.stack(image_features).squeeze(1)
    print(image_features.shape)
    print(all_image_features[0][0])
    text_features = []
    for i in range(len(all_text_features)):
        text_features.append(all_text_features[i][2])
    text_features = torch.stack(text_features).squeeze(1)
    print(text_features.shape)
    print(all_text_features[0][1])

    # 高效执行检索部分
    retrieve_method = 'KNN'  # L2 KNN LSH

    # 图像检索文本
    print('-' * 80)
    print('图像检索文本')
    all_real_rank = np.array([[i] for i in range(len(all_text_features))])
    all_pred_rank = high_speed_retreive(image_features, text_features, retrieve_method, 100)
    # all_real_rank = high_speed_retreive(text_features, text_features, 'L2', 100)


    # 计算 NDCG， Recall
    for k in [5, 10, 20, 50]:
        results = get_performance(all_real_rank, all_pred_rank, k)
        string = f"Recall@{k}="
        print(f"{string:15s}{results['recalls'].mean():.4f}")
        string = f"NDCG@{k}="
        print(f"{string:15s}{results['ndcgs'].mean():.4f}")

    # 文本检索图像
    print('-' * 80)
    print('文本检索图像')
    all_real_rank = np.array([[i] for i in range(len(all_text_features))])
    all_pred_rank = high_speed_retreive(text_features, image_features, retrieve_method, 100)
    # 计算 NDCG， Recall
    for k in [5, 10, 20, 50]:
        results = get_performance(all_real_rank, all_pred_rank, k)
        string = f"Recall@{k}="
        print(f"{string:15s}{results['recalls'].mean():.4f}")
        string = f"NDCG@{k}="
        print(f"{string:15s}{results['ndcgs'].mean():.4f}")