import pickle

from parse_args import args

import numpy as np
import torch
import scipy.sparse as sp


def load_data():
    data_path = args.data_path
    mobility_adj = np.load(data_path + args.mobility_adj, allow_pickle=True)
    mobility_adj = mobility_adj.squeeze()
    mobility = mobility_adj.copy()
    mobility = mobility / np.mean(mobility)

    poi_similarity = np.load(data_path + args.poi_similarity, allow_pickle=True)
    poi_similarity[np.isnan(poi_similarity)] = 0

    d_adj = np.load(data_path + args.destination_adj, allow_pickle=True)
    d_adj[np.isnan(d_adj)] = 0

    s_adj = np.load(data_path + args.source_adj, allow_pickle=True)
    s_adj[np.isnan(s_adj)] = 0

    neighbor = np.load(data_path + args.neighbor, allow_pickle=True)

    return poi_similarity, s_adj, d_adj, mobility, neighbor


def graph_to_COO(similarity, importance_k):
    graph = torch.eye(180)

    for i in range(180):
        graph[np.argsort(similarity[:, i])[-importance_k:], i] = 1
        graph[i, np.argsort(similarity[:, i])[-importance_k:]] = 1

    #将邻接矩阵转换成coo格式
    edge_index = sp.coo_matrix(graph)
    #转换成边索引的矩阵（两行，第一行值是行索引，第二行值是列索引）2*E大小
    edge_index = np.vstack((edge_index.row, edge_index.col))
    return edge_index


def create_graph(similarity, importance_k):
    edge_index = graph_to_COO(similarity, importance_k)
    return edge_index


def pair_sample(neighbor):
    positive = torch.zeros(180, dtype=torch.long)
    negative = torch.zeros(180, dtype=torch.long)

    for i in range(180):
        region_idx = np.random.randint(len(neighbor[i]))
        pos_region = neighbor[i][region_idx]
        positive[i] = pos_region
    for i in range(180):
        neg_region = np.random.randint(180)
        while neg_region in neighbor[i] or neg_region == i:
            neg_region = np.random.randint(180)
        negative[i] = neg_region
    return positive, negative


def create_neighbor_graph(neighbor):
    graph = np.eye(180)

    for i in range(len(neighbor)):
        for region in neighbor[i]:
            graph[i, region] = 1
            graph[region, i] = 1
    graph = sp.coo_matrix(graph)
    edge_index = np.stack((graph.row, graph.col))
    return edge_index


'''准备空间功能一致性得分矩阵'''
def prepare_S_dict(device='cuda'):
    proximity = np.load("data/spatial_simi.npy")
    poi_sim = np.load("data/poi_similarity.npy")
    s_sim = np.load("data/source_adj.npy")
    d_sim = np.load("data/destination_adj.npy")


    # 各个视图特征相似度 × 空间邻近
    S_poi = torch.tensor(poi_sim * proximity, dtype=torch.float32, device=device)
    S_s   = torch.tensor(s_sim * proximity, dtype=torch.float32, device=device)
    S_d   = torch.tensor(d_sim * proximity, dtype=torch.float32, device=device)
    S_n   = torch.tensor(proximity, dtype=torch.float32, device=device)  # 原始邻接仅考虑空间邻近度
    # S_n = torch.zeros_like(S_d)

    S_dict = {
        'poi': S_poi,
        's': S_s,
        'd': S_d,
        'n': S_n
    }
    return S_dict


def prepare_C_list(device='cuda'):
    C_list_raw = pickle.load(open("./data/C_list.pkl", "rb"))
    for p in range(4):
        for q in range(4):
            if C_list_raw[p][q] is None:
                # 保持 None
                C_list_raw[p][q] = None
            else:
                # 非 None 时，把 NumPy 数组转成 float32 的 Tensor 并搬到 device 上
                C_list_raw[p][q] = torch.from_numpy(C_list_raw[p][q]).float().to(device)
    C_list = C_list_raw
    return C_list