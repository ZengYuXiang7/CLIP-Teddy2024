# coding : utf-8
# Author : yuxiang Zeng

"""
    基于图像检索的模型和算法，利用附件 2 中“word_test.csv”文件的文本信息，
    对附件 2 的 ImageData 文件夹的图像进行图像检索，并罗列检索相似度较高的前五张图像，
    将结果存放在“result1.csv”文件中（模板文件详见附件 4 的 result1.csv）。
    其中，ImageData 文件夹中的图像 ID 详见附件 2 的“image_data.csv”文件。

    总结：已知文本，检索图像最相似的前5张图像
"""
import os
import pandas as pd
import numpy as np
from tqdm import *

if __name__ == '__main__':
    allImages = os.listdir('../BigDataSource/Teddy2024/附件2/ImageData')
    allImagesList = pd.read_csv('../BigDataSource/Teddy2024/附件2/image_data.csv')

    import chardet
    from modules.models.SBERT import Text2Vec

    file_name = '../BigDataSource/Teddy2024/附件2/word_test.csv'
    # 首先，读取文件的一部分字节来检测编码
    with open(file_name, 'rb') as f:
        raw_data = f.read(5000)
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    allTextList = pd.read_csv(file_name, encoding=encoding)
    allTextList = np.array(allTextList)

    text2vec = Text2Vec()
    for i in trange(len(allTextList)):
        allTextList[i, 1] = text2vec.text2vec(allTextList[i, 1])
    print('Done!')
    from modules.models.ResNet50 import Image2tensor
    from PIL import Image

    # 这里是执行多线程代码，线性运算过慢，线性预算代码在上面，用时8mins
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # function
    def process_image(inputs):
        imageTranfer, image_name = inputs
        file_name = '../BigDataSource/Teddy2024/附件2/ImageData/' + image_name
        image = Image.open(file_name)
        features = imageTranfer.image2tensor(image)
        return features


    # input
    allImages = os.listdir('../BigDataSource/Teddy2024/附件2/ImageData')
    imageTranfer = Image2tensor()
    inputList = [(imageTranfer, imageNames) for imageNames in allImages]

    # Execute
    allImagesFeatures = []
    allImages = os.listdir('../BigDataSource/Teddy2024/附件2/ImageData')
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_image, inputs) for inputs in inputList]
        for future in tqdm(as_completed(futures), total=len(allImages)):
            allImagesFeatures.append(future.result())
    print('Done!')

    # 将所有变量转成numpy形式
    allImagesFeatures_now = []
    for i in range(len(allImagesFeatures)):
        imageFeatures = np.array(allImagesFeatures[i]).reshape(-1, )
        allImagesFeatures_now.append(imageFeatures)
    allImagesFeatures = np.array(allImagesFeatures_now)

    # 将所有变量转成numpy形式
    allTextFeatures_now = []
    for i in range(len(allTextList)):
        textFeatures = np.array(allTextList[i, 1]).reshape(-1, )
        allTextFeatures_now.append(textFeatures)
    allTextFeatures = np.array(allTextFeatures_now)

    from sklearn.decomposition import PCA
    import numpy as np

    # 假设 `visual_features` 是一个形状为 (n_samples, 1000) 的数组，

    # 初始化PCA对象，设置目标维度为384
    pca = PCA(n_components=384)

    # 对视觉特征进行降维
    scaled_allImagesFeatures = pca.fit_transform(allImagesFeatures)

    # 查看降维后的形状，确认是否为 (n_samples, 384)
    print("降维后的视觉特征形状:", scaled_allImagesFeatures.shape)
