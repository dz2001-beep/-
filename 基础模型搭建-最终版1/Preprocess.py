import pickle

import torch


'''计算跨视图区域相关性'''
import numpy as np


def compute_cross_view_correlations(S_list):
    """
    给定多个视图下区域相似度矩阵，计算所有视图对的跨视图区域相关性矩阵。
    S_list: List[Tensor], 每个 shape 为 (N, N)，表示一个视图下的区域相似度
    return: C_list: List[List[Tensor]], C[i][j] 表示视图 i 和 j 的跨视图区域相关性矩阵
    """
    num_views = len(S_list)
    C_list = [[None for _ in range(num_views)] for _ in range(num_views)]

    for i in range(num_views):
        for j in range(num_views):
            if i == j:
                continue  # 不计算视图内的
            # 计算 C^{(i,j)}_{i,j} = S^i * S^j
            C_list[i][j] = S_list[i] * S_list[j]  # element-wise multiply, shape: (N, N)

    return C_list


def save_C_list(C_list, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(C_list, f)

if __name__ == '__main__':
    S_in = np.load('data/destination_adj.npy')
    S_out = np.load('data/destination_adj.npy')
    S_poi = np.load('data/poi_similarity.npy')
    S_n = np.load('data/spatial_simi.npy')


    S_list = [S_in, S_out, S_poi, S_n]
    C_list = compute_cross_view_correlations(S_list)

    save_C_list(C_list, "./data/C_list.pkl")

    # for i in range(len(C_list)):
    #     for j in range(len(C_list[i])):
    #         if C_list[i][j] is not None:
    #             print(f"C_list[{i}][{j}] shape: {C_list[i][j].shape}")
    #         else:
    #             print(f"C_list[{i}][{j}] is None")