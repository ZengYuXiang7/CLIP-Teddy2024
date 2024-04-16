# coding : utf-8
# Author : yuxiang Zeng

from time import time
import faiss  # 导入faiss
import numpy as np

class LSH:
    def __init__(self, k, d, nbits):
        self.d = d  # 向量维度
        self.k = k  # 搜索的Top-K结果数量
        self.nbits = nbits  # LSH哈希位数
        self.index = faiss.IndexLSH(d, nbits)  # 创建LSH索引
        self.is_data_added = False  # 标记是否已经添加了数据

    def search_topk_embeds(self, data_base, h_query):
        data_base, h_query = np.array(data_base).astype('float32'), np.array(h_query).astype('float32')
        if not self.is_data_added:
            self.index.add(data_base)
            self.is_data_added = True
        topk_distance, topk_indices = self.index.search(h_query, self.k)  # 执行Top-K搜索
        return topk_distance, topk_indices

class L2Index:
    def __init__(self, k, d):
        self.d = d  # 向量维度
        self.k = k  # 搜索的Top-K结果数量
        self.index = faiss.IndexFlatL2(d)  # 创建IndexFlatL2索引
        self.is_data_added = False  # 标记是否已经添加了数据

    def search_topk_embeds(self, data_base, h_query):
        data_base, h_query = np.array(data_base).astype('float32'), np.array(h_query).astype('float32')
        if not self.is_data_added:
            self.index.add(data_base)  # 第一次调用时添加数据
            self.is_data_added = True
        topk_distance, topk_indices = self.index.search(h_query, self.k)  # 执行Top-K搜索
        return topk_distance, topk_indices


class IVFFlatIndex:
    def __init__(self, k, d, nlist):
        self.d = d
        self.k = k
        self.nlist = nlist
        quantizer = faiss.IndexFlatL2(d)  # 使用FlatL2作为量化器
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist)
        self.is_trained = False

    def search_topk_embeds(self, data_base, h_query):
        data_base, h_query = np.array(data_base).astype('float32'), np.array(h_query).astype('float32')
        if not self.is_trained:
            self.index.train(data_base)
            self.is_trained = True
        if not self.index.is_trained:
            print("Index is not trained.")
            return None, None
        self.index.add(data_base)
        topk_distance, topk_indices = self.index.search(h_query, self.k)
        return topk_distance, topk_indices


class IVFPQIndex:
    def __init__(self, k, d, nlist, m):
        self.d = d
        self.k = k
        self.nlist = nlist  # 聚类中心的数量
        self.m = m  # PQ子向量数量
        quantizer = faiss.IndexFlatL2(d)  # 量化器
        self.index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  # 假设每个子向量8位
        self.is_trained = False

    def search_topk_embeds(self, data_base, h_query):
        data_base, h_query = np.array(data_base).astype('float32'), np.array(h_query).astype('float32')
        if not self.is_trained:
            self.index.train(data_base)
            self.is_trained = True
        if not self.index.is_trained:
            print("Index is not trained.")
            return None, None
        self.index.add(data_base)
        topk_distance, topk_indices = self.index.search(h_query, self.k)
        return topk_distance, topk_indices


class HNSWIndex:
    def __init__(self, k, d):
        self.d = d
        self.k = k
        self.index = faiss.IndexHNSWFlat(d, 32)  # 32是HNSW的层次数

    def search_topk_embeds(self, data_base, h_query):
        data_base, h_query = np.array(data_base).astype('float32'), np.array(h_query).astype('float32')
        self.index.add(data_base)
        topk_distance, topk_indices = self.index.search(h_query, self.k)
        return topk_distance, topk_indices


# 最大近邻搜索
if __name__ == '__main__':

    d = 64  # 向量维度
    nb = 190000  # 数据库大小
    nq = 256 * 10  # 查询向量的数量
    np.random.seed(1234)  # 保证可复现的结果

    # 生成一些随机样本向量
    xb = np.random.random((nb, d))
    xq = np.random.random((nq, d))

    t1 = time()
    lsh = LSH(k=3, d=d, nbits=d)
    topk_distence, topk_indices = lsh.search_topk_embeds(xb, xq)
    t2 = time()
    print(f'LSH: {t2 - t1 : .2f}s')

    t1 = time()
    l2index = L2Index(k=3, d=d)
    topk_distence, topk_indices = l2index.search_topk_embeds(xb, xq)
    t2 = time()
    print(f'L2: {t2 - t1 : .2f}s')

    t1 = time()
    ivfflatindex = IVFFlatIndex(k=3, d=d, nlist=128)
    topk_distence, topk_indices = ivfflatindex.search_topk_embeds(xb, xq)
    t2 = time()
    print(f'IVFFlat: {t2 - t1 : .2f}s')

    t1 = time()
    ivfpqindex = IVFPQIndex(k=3, d=d, nlist=128, m=8)
    topk_distence, topk_indices = ivfpqindex.search_topk_embeds(xb, xq)
    t2 = time()
    print(f'IVFPQ: {t2 - t1 : .2f}s')

    t1 = time()
    hnswindex = HNSWIndex(k=3, d=d)
    topk_distence, topk_indices = hnswindex.search_topk_embeds(xb, xq)
    t2 = time()
    print(f'HNSW: {t2 - t1 : .2f}s')

