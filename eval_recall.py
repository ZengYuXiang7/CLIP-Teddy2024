# coding : utf-8
# Author : yuxiang Zeng

import torch
import numpy as np


def recall_at_k(relevant, recommended, k):
    """
    计算 Recall@K
    :param relevant: 实际感兴趣的项目矩阵，每行代表一个用户的相关项目集合，二维数组
    :param recommended: 推荐系统给出的推荐项目矩阵，每行代表一个用户的推荐项目列表，二维数组
    :param k: 考虑的推荐列表的前K项
    :return: 每个用户的 Recall@K 的值，一维数组
    """
    recalls = []
    for rel_set, rec_list in zip(relevant, recommended):
        if k < len(rec_list):
            rec_list_k = rec_list[:k]
        else:
            rec_list_k = rec_list
        relevant_recommended = set(rel_set).intersection(rec_list_k)
        if len(rel_set) > 0:
            recalls.append(len(relevant_recommended) / len(set(rel_set)))
        else:
            recalls.append(0)
    return np.array(recalls)


def dcg_at_k(scores, k):
    """
    计算 DCG@K
    :param scores: 相关性得分列表，列表类型
    :param k: 考虑的推荐列表的前K项
    :return: DCG@K 的值
    """
    scores = scores[:k]
    if len(scores):
        return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))  # log base 2 of positions 2 through k+1
    return 0


def ndcg_at_k(relevant, recommended, k):
    """
    计算 NDCG@K
    :param relevant: 实际感兴趣的项目矩阵，二维数组
    :param recommended: 推荐项目矩阵，二维数组
    :param k: 考虑的推荐列表的前K项
    :return: 每个用户的 NDCG@K 的值，一维数组
    """
    ndcgs = []
    for rel_set, rec_list in zip(relevant, recommended):
        rel_set = set(rel_set)
        if k < len(rec_list):
            rec_list_k = rec_list[:k]
        else:
            rec_list_k = rec_list

        actual_scores = [1 if item in rel_set else 0 for item in rec_list_k]
        ideal_scores = [1] * min(len(rel_set), k)

        actual_dcg = dcg_at_k(actual_scores, k)
        ideal_dcg = dcg_at_k(ideal_scores, k)

        if ideal_dcg > 0:
            ndcgs.append(actual_dcg / ideal_dcg)
        else:
            ndcgs.append(0)
    return np.array(ndcgs)


def get_performance(relevant, recommended, k_value):
    # 计算 Recall@K 和 NDCG@K
    recalls = recall_at_k(relevant, recommended, k_value)
    ndcgs = ndcg_at_k(relevant, recommended, k_value)
    return {
        'recalls': recalls,
        'ndcgs': ndcgs
    }

if __name__ == '__main__':
    # 示例数据
    n = 5  # 假设有5个用户
    np.random.seed(0)
    relevant = np.random.randint(1, 500, (1, 1))
    recommended = np.random.randint(1, 500, (1, 100))
    print(relevant)
    print(recommended)

    # 计算 Recall@K 和 NDCG@K
    for k in [5, 10, 20, 50]:
        results = get_performance(relevant, recommended, k)
        string = f"Recall@{k}="
        print(f"{string:15s}{results['recalls'].mean():.4f}")
        string = f"NDCG@{k}="
        print(f"{string:15s}{results['ndcgs'].mean():.4f}")