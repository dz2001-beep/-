import pickle

from parse_args import args
import utils
from utils import prepare_S_dict, prepare_C_list
from parse_args import args
from tasks import predict_crime, clustering, predict_check
from model import Model

import random
from tqdm import tqdm
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import time

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#通过固定随机种子，确保每次运行生成的随机数相同
seed = 2022 #3407  2022
torch.manual_seed(seed=seed)
np.random.seed(seed)
random.seed(seed)


#加载数据
poi_similarity, s_adj, d_adj, mobility, neighbor = utils.load_data()

#根据相似度构建图，返回边索引 [2,E]
poi_edge_index = utils.create_graph(poi_similarity, args.importance_k)
s_edge_index = utils.create_graph(s_adj, args.importance_k)
d_edge_index = utils.create_graph(d_adj, args.importance_k)
n_edge_index = utils.create_neighbor_graph(neighbor)

#边索引转张量类型
poi_edge_index = torch.tensor(poi_edge_index, dtype=torch.long).to(args.device)
s_edge_index = torch.tensor(s_edge_index, dtype=torch.long).to(args.device)
d_edge_index = torch.tensor(d_edge_index, dtype=torch.long).to(args.device)
n_edge_index = torch.tensor(n_edge_index, dtype=torch.long).to(args.device)

mobility = torch.tensor(mobility, dtype=torch.float32).to(args.device)
poi_similarity = torch.tensor(
    poi_similarity, dtype=torch.float32).to(args.device)

# 随机初始化区域表示、各个视图的关系表示
features = torch.randn(args.regions_num, args.embedding_size).to(args.device)
poi_r = torch.randn(args.embedding_size).to(args.device)
s_r = torch.randn(args.embedding_size).to(args.device)
d_r = torch.randn(args.embedding_size).to(args.device)
n_r = torch.randn(args.embedding_size).to(args.device)
rel_emb = [poi_r, s_r, d_r, n_r]
edge_index = [poi_edge_index, s_edge_index, d_edge_index, n_edge_index]
'''空间得分矩阵'''
S_sp = torch.tensor(np.load("data/spatial_simi.npy"),dtype=torch.float32)
'''空间功能一致性得分矩阵：POI、S、D、N'''
S_dict = prepare_S_dict(args.device)
'''跨视图区域相关性矩阵'''
C_list = prepare_C_list(args.device)

'''计算人流量重建损失'''
def mob_loss(s_emb, d_emb, mob):
    inner_prod = torch.mm(s_emb, d_emb.T)
    ps_hat = F.softmax(inner_prod, dim=-1)
    inner_prod = torch.mm(d_emb, s_emb.T)
    pd_hat = F.softmax(inner_prod, dim=-1)
    loss = torch.sum(-torch.mul(mob, torch.log(ps_hat)) -
                     torch.mul(mob, torch.log(pd_hat)))
    return loss


'''训练函数'''
def train(net):
    optimizer = optim.Adam(
        net.parameters(), lr=args.learning_rate, weight_decay=5e-3)
    loss_fn1 = torch.nn.TripletMarginLoss()
    loss_fn2 = torch.nn.MSELoss()

    best_rmse_crime = 10000
    best_mae_crime = 10000
    best_r2_crime = 0
    best_epoch_crime = 0

    '''添加签到预测最好指标'''
    best_rmse_check = 10000
    best_mae_check = 10000
    best_r2_check = 0
    best_epoch_check = 0
    '''添加土地使用最好指标'''
    best_nmi = 0
    best_ari = 0
    best_epoch_land = 0

    '''存储每个epoch的loss和mae'''
    train_loss_history = []
    train_mae_history = []

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        region_emb, n_emb, poi_emb, s_emb, d_emb = net(
            features, edge_index, S_dict)

        pos_idx, neg_idx = utils.pair_sample(neighbor)

        geo_loss = loss_fn1(n_emb, n_emb[pos_idx], n_emb[neg_idx])

        m_loss = mob_loss(s_emb, d_emb, mobility)

        poi_loss = loss_fn2(torch.mm(poi_emb, poi_emb.T), poi_similarity)
        loss = poi_loss + m_loss + geo_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # 1.犯罪预测最好效果
            mae_crime, rmse_crime, r2_crime = predict_crime(region_emb.detach().cpu().numpy())
            if rmse_crime < best_rmse_crime and mae_crime < best_mae_crime and best_r2_crime < r2_crime:
                best_rmse_crime = rmse_crime
                best_mae_crime = mae_crime
                best_r2_crime = r2_crime
                best_epoch_crime = epoch
                best_emb_crime = region_emb.detach().cpu().numpy()
                np.save("best_emb_crime.npy", best_emb_crime)
            print(f"【{epoch}|{args.epochs}】epochs:(mae){mae_crime},(rmse){rmse_crime},(r2){r2_crime}, (loss){loss.item()}.")

            # 2.签到 预测最好效果
            mae_check, rmse_check, r2_check = predict_check(region_emb.detach().cpu().numpy())
            if rmse_check < best_rmse_check and mae_check < best_mae_check and best_r2_check < r2_check:
                best_rmse_check = rmse_check
                best_mae_check = mae_check
                best_r2_check = r2_check
                best_epoch_check = epoch
                best_emb_check = region_emb.detach().cpu().numpy()
                np.save("best_emb_check.npy", best_emb_check)
            print(f"【{epoch}|{args.epochs}】epochs:(mae){mae_check},(rmse){rmse_check},(r2){r2_check}, (loss){loss.item()}.")

            #3.land use
            nmi, ari = clustering(region_emb.detach().cpu().numpy())
            if nmi > best_nmi and ari > best_ari:
                best_nmi = nmi
                best_ari = ari
                best_epoch_land = epoch
                best_emb_land = region_emb.detach().cpu().numpy()
                np.save("best_emb_land.npy", best_emb_land)
            print(f"【{epoch}|{args.epochs}】epochs:(nmi){nmi},(ari){ari}, (loss){loss.item()}.")
            np.save("emb.npy", region_emb.detach().cpu().numpy())
    print("=======================best_crime===========================")
    print('best_mae_crime:', best_mae_crime)
    print('best_rmse_crime:', best_rmse_crime)
    print('best_r2_crime:', best_r2_crime)
    print('best_epoch_crime:', best_epoch_crime)
    print("=======================best_check===========================")
    print('best_mae_check:', best_mae_check)
    print('best_rmse_check:', best_rmse_check)
    print('best_r2_check:', best_r2_check)
    print('best_epoch_check:', best_epoch_check)
    print("=======================best_land===========================")
    print('best_nmi:', best_nmi)
    print('best_ari:', best_ari)
    print('best_epoch_land:', best_epoch_land)


'''测试函数'''
def test(path):
    # region_emb = np.load("./emb.npy")
    region_emb = np.load(path)

    print('>>>>>>>>>>>>>>>>>   crime')
    mae, rmse, r2 = predict_crime(region_emb)
    print("MAE:  %.3f" % mae)
    print("RMSE: %.3f" % rmse)
    print("R2:   %.3f" % r2)
    print('>>>>>>>>>>>>>>>>>   check')
    mae, rmse, r2 = predict_check(region_emb)
    print("MAE:  %.3f" % mae)
    print("RMSE: %.3f" % rmse)
    print("R2:   %.3f" % r2)
    print('>>>>>>>>>>>>>>>>>   clustering')
    nmi, ari = clustering(region_emb)
    print("NMI: %.3f" % nmi)
    print("ARI: %.3f" % ari)


if __name__ == '__main__':

    net = Model(S_dict, C_list, args.embedding_size, args.gcn_layers,
                args.dropout, args.TopK, args.gamma).to(args.device)

    # 记录开始时间
    start_time = time.time()
    print('==================training start===================' )
    net.train()
    train(net)
    net.eval()
    print('===================downstream task test=================')
    path = "./emb.npy"
    test(path)
    # 记录结束时间
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"代码运行时间: {execution_time:.6f} 秒")


    # 最好的嵌入
    print("=================best_land==================")
    best_land_path = 'best_emb_land.npy'
    test(best_land_path)



