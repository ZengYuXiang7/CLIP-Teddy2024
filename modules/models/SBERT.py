# coding : utf-8
# Author : yuxiang Zeng


import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

class Text2Vec:
    def __init__(self):
        # 初始化SBert模型，这里使用的是'all-MiniLM-L6-v2'，
        # 它是一个经过优化，适用于生成高质量文本嵌入的模型
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def text2vec(self, text):
        # 直接使用模型提取文本嵌入，这个方法为文本生成了高质量的嵌入向量，
        # 非常适合进行文本相似度计算等任务
        embeddings = self.model.encode(text)
        return embeddings


if __name__ == '__main__':
    text2vec = Text2Vec()
    text = "I like the computer!"
    embeddings = text2vec.text2vec(text)
    print(f'该文本[{text}]嵌入表示维度为:', embeddings.shape)

