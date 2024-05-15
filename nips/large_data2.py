import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from utils import *
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import timedelta, datetime
from collections import Counter
import json
from sklearn import preprocessing
from scipy.sparse import lil_matrix

dataname = 'toy_large'
res_dir = 'results/{}/'.format(dataname)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)


def draw_graph_toy_1(adjacency_matrix_pred, dataname, noise_level=0):
    G = nx.from_numpy_array(np.array(adjacency_matrix_pred))
    pos = nx.spring_layout(G, scale=20, k=3 / np.sqrt(G.order()), seed=1)
    #pos = nx.random_layout(G, k=3 / np.sqrt(G.order()), seed=1)

    d = dict(G.degree)
    plt.figure(figsize=(24, 18), dpi=200)
    plt.title('{} | graph | noise: {}'.format(dataname, noise_level))
    nx.draw(
        G,
        pos,
        width=2,
        #with_labels=True,
        alpha=0.5,
        edge_color='#2a3f95',
        node_size=[200 + d[k] * 200 for k in d],
    )

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig('results/{}/{}_graph_noise_{}.png'.format(dataname, dataname, noise_level))


def draw_graph_toy_1_compare(
    adjacency_matrix_pred, adjacency_matrix_pred_noise, dataname, noise_level=0
):
    # 创建两个图
    G = nx.from_numpy_array(np.array(adjacency_matrix_pred))
    pos = nx.spring_layout(G, scale=20, k=3 / np.sqrt(G.order()), seed=seed)
    #pos = nx.random_layout(G, k=3 / np.sqrt(G.order()), seed=seed)

    d = dict(G.degree)
    plt.figure(figsize=(24, 18), dpi=200)
    plt.title('{} | graph | noise: {}'.format(dataname, noise_level))
    nx.draw(
        G,
        pos,
        width=4,
        #with_labels=True,
        alpha=0.5,
        edge_color='#2a3f95',
        node_size=[200 + d[k] * 200 for k in d],
    )

    G_noise = nx.from_numpy_array(np.array(adjacency_matrix_pred_noise))
    extra_edges = [edge for edge in G_noise.edges() if edge not in G.edges()]
    missing_edges = [edge for edge in G.edges() if edge not in G_noise.edges()]
    nx.draw_networkx_edges(G, pos, width=7, edgelist=extra_edges, edge_color='red', style='solid')
    nx.draw_networkx_edges(
        G, pos, width=7, edgelist=missing_edges, edge_color='orange', style='solid'
    )

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig('results/{}/{}_graph_noise_{}.png'.format(dataname, dataname, noise_level))


def generate_sparse_precision_matrix(n, density=0.2):
    # 创建一个低密度的随机矩阵
    matrix = lil_matrix((n, n))

    # 随机设置一些元素
    for i in range(n):
        for j in range(i, n):
            if np.random.random() < density:
                # value = np.random.randn()
                value = 0.5
                matrix[i, j] = value
                matrix[j, i] = value

    # # 确保对角线上的值足够大，使矩阵正定
    # for i in range(n):
    #     matrix[i, i] += n

    # 调整对角线上的值，使每个对角线元素稍大于其行的非对角线元素之和
    for i in range(n):
        row_sum = sum(abs(matrix[i, j]) for j in range(n) if j != i)
        # matrix[i, i] = row_sum + 0 # 或者可以使用更小的安全边际
        matrix[i, i] = 1

    print(matrix.toarray())
    return matrix.toarray()


def generate_data_sparse(num_nodes, num_samples, num_cases=1):
    '''随机的 presicion '''

    np.random.seed(1)

    res_dir = 'data/sparse_p{}_n{}/'.format(num_nodes, num_samples)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    threshold = 1e-10

    for data_id in range(num_cases):
        precision = generate_sparse_precision_matrix(n = num_nodes, density=0.2)

        covariance = np.linalg.inv(precision)

        mean = np.zeros(num_nodes)
        samples = np.random.multivariate_normal(mean, covariance, size=num_samples)
        print(samples.shape)

        np.fill_diagonal(precision, 0)
        precision = np.abs(precision)
        draw_precision(precision, dataname, noise_level="Truth")
        draw_cov_matrix(covariance, dataname, noise_level="Truth")


        adj_matrix = np.where(precision >= threshold, 1, 0)
        np.save(res_dir + 'toy_adj_matrix_{}.npy'.format(data_id), adj_matrix)
        df = pd.DataFrame(samples, columns=[f'Node_{i}' for i in range(num_nodes)])
        df.to_csv(res_dir + 'toy_data_{}.csv'.format(data_id), index=False)

num_nodes = 1000
num_samples = 1000

generate_data_sparse(num_nodes, num_samples, num_cases=1)


#generate_data_dense(num_nodes, num_samples, num_cases=10)

FPRs_all = []
TPRs_all = []
AUCs_all = []

time1 = time.time()
for data_id in range(10):  # 10 次实验
    # data_id = 0
    df = pd.read_csv('./data/sparse_p{}_n{}/toy_data_{}.csv'.format(num_nodes, num_samples, data_id))
    DATA = df.values
    adj_matrix = np.load('./data/sparse_p{}_n{}/toy_adj_matrix_{}.npy'.format(num_nodes, num_samples, data_id))  # 作为truth

    # note Truth part ############################################################################
    # section 画 adj_matrix ground Truth:
    # draw_adjacency_matrix(adj_matrix, dataname, noise_level='Truth')
    # section 画 graph ground Truth:
    # draw_graph_toy_1(adj_matrix, dataname, noise_level='Truth')


    # note model part ############################################################################
    # section 预处理：标准化节点特征?
    # scaler = StandardScaler()  # 去均值和方差归一化： 对每一个特征维度来做的，而不是针对样本。
    # DATA = scaler.fit_transform(DATA)
    norm = False
    seed = 1
    variance = np.var(DATA)
    L, N = DATA.shape
    print('DATA (L, N):', (L, N))
    print('variance of X:', variance)
    threshold = 0.0001
    n_splits = 10
    cross_val = False
    lambdas_exp = range(-1, 4)
    # lambdas_exp = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 1.1, 1.2, 1.3]
    #lambdas_exp = [0.0061, 0.0062, 0.0063, 0.0064, 0.0065, 0.0066, 0.0067]
    lambdas_exp = np.linspace(-4, 4, 20)
    lambdas_exp = np.round(lambdas_exp, 2)

    # --以下可以复制-------------------
    # section cov_matrix:
    cov_matrix = np.cov(DATA, rowvar=False)  # Calculate the sample covariance matrix
    draw_cov_matrix(cov_matrix, dataname)

    # section cross-val 选择 lambda
    if cross_val:
        # lambdas_exp = range(1, 6)
        lambdas = [10**i for i in lambdas_exp]  # this is for cd method

        SCOREs2 = cross_validation_for_lambda(lambdas, DATA, n_splits=n_splits)

        fig = plt.figure(figsize=(18, 6))
        plt.subplots_adjust(left=0.15, right=0.95, top=0.82, bottom=0.25)
        ax = fig.add_subplot(111)
        ax.set_title('Cross validation for lambda')
        ax.scatter(lambdas_exp, SCOREs2, label='mse')
        ax.set_xlabel('$\lambda$')
        ax.set_ylabel('-MSE')
        ax.set_xticks(
            lambdas_exp, ['$10^{{{}}}$'.format(i) for i in lambdas_exp]
        )  #  使用双大括号 {{}} 来转义大括号，使其在 format 调用中不被解释为格式化占位符
        ax.grid()
        plt.savefig(
            'results/{}/{}_cross_validation_lambda.pdf'.format(dataname, dataname), dpi=300
        )
        plt.show()

    # section GL 计算获得 precision_
    # best_rho2 = 10 ** -0.89 # todo case0
    # best_rho2 = 10**-0.06  # todo case0
    best_rho2 = 10 ** -0.04 # todo case0
    # best_rho2 = 10 ** 0.1

    covariance_, precision_, costs, iter = cd_graphical_lasso(emp_cov=cov_matrix, alpha=best_rho2)

    np.fill_diagonal(precision_, 0)
    precision_ = np.abs(precision_)
    print(precision_)

    draw_precision(precision_, dataname, noise_level=0)


    # section 阈值 得到 adjacency_matrix
    adjacency_matrix_pred = np.where(precision_ >= threshold, 1, 0)
    draw_adjacency_matrix(adjacency_matrix_pred, dataname, noise_level=0)

    # section 画 pred graph
    # draw_graph_toy_1(adjacency_matrix_pred, dataname, noise_level=0)

    plt.show()

    # note noise part ############################################################################
    if norm:
        noise_stds = [0.00001, 0.0001, 0.001, 0.01, 1, 0]
        # noise_stds = [0.3, 1, 1.2, 1.8,2,  0]
    else:
        noise_stds = [
            np.sqrt(variance / (10**10)),
            np.sqrt(variance / (10**8)),
            np.sqrt(variance / (10**6)),
            np.sqrt(variance / (10**4)),
            np.sqrt(variance / (10**2)),
            0,
        ]
    print('noise_stds:', noise_stds)
    noise_dbs_for_draw = [100, 80, 60, 40, 20, 'No']

    FPRs = []
    TPRs = []
    AUCs = []
    DATA_noises = []
    for i, noise_std in enumerate(noise_stds):
        DATA_noise = add_noise_for_time_series(DATA, noise_std_dev=noise_std, seed=seed)
        DATA_noises.append(DATA_noise)
        # print('noise_std:', noise_std, noise_dbs_for_draw[i],'db', 'L:', L, 'N:', N)

        # section cov_matrix_noise
        cov_matrix_noise = np.cov(
            DATA_noise, rowvar=False
        )  # Calculate the sample covariance matrix

        # section cov_matrix re-fine
        term = (1 - 1 / L) * (noise_std**2)
        cov_matrix_noise = cov_matrix_noise - term * np.eye(N)

        draw_cov_matrix(cov_matrix_noise, dataname, noise_level=noise_dbs_for_draw[i])

        # section precision_noise calculate
        covariance_noise, precision_noise, costs, iter = cd_graphical_lasso(
            emp_cov=cov_matrix_noise, alpha=best_rho2
        )

        # section precision_ ----> 对角线0 + MAX-MIN normlize
        np.fill_diagonal(precision_noise, 0)
        precision_noise = np.abs(precision_noise)

        draw_precision(precision_noise, dataname, noise_level=noise_dbs_for_draw[i])  # todo

        # section pred adjacency_matrix_pred
        adjacency_matrix_pred_noise = np.where(precision_noise >= threshold, 1, 0)
        draw_adjacency_matrix(
            adjacency_matrix_pred_noise, dataname, noise_level=noise_dbs_for_draw[i]
        )

        # section connection pred graph

        # draw_graph_toy_1_compare(
        #     adjacency_matrix_pred=adj_matrix,    # adjacency_matrix_pred
        #     adjacency_matrix_pred_noise=adjacency_matrix_pred_noise,
        #     dataname=dataname,
        #     noise_level=noise_dbs_for_draw[i],
        # )

        # draw_graph_toy_1(adjacency_matrix_pred_noise,
        #                          dataname = dataname, noise_level=noise_dbs_for_draw[i])

        # section ROC
        fpr, tpr, thresholds_ = roc_curve(y_true=adjacency_matrix_pred.ravel(), y_score=precision_noise.ravel() )  # todo adj_matrix  adjacency_matrix_pred
        auc = roc_auc_score(adjacency_matrix_pred.ravel(), precision_noise.ravel())  # todo adj_matrix  adjacency_matrix_pred
        TPRs.append(tpr)
        FPRs.append(fpr)
        AUCs.append(auc)

    draw_ROC(TPRs, FPRs, AUCs, dataname, GL=True)

    TPRs_all.append(TPRs)
    FPRs_all.append(FPRs)
    AUCs_all.append(AUCs)
    # plt.show()

time2 = time.time()
print("time", time2 - time1)


AUCs_all_mean = np.mean(AUCs_all, axis=0)
AUCs_all_deviations = np.max(np.abs(AUCs_all - AUCs_all_mean), axis=0)
print(AUCs_all_mean)
print(AUCs_all_deviations)
plt.show()




'''
exit()
# print(noise_dbs_for_draw[i],'db', data_id, 'auc', auc)
# print('thresholds_', thresholds_)

# TPRs_all.append(TPRs)
# FPRs_all.append(FPRs)
# AUCs_all.append(AUCs)


# print(FPRs_all)
# print(FPRs_all)
# print(AUCs_all)
AUCs_all = np.array(AUCs_all)
# print(AUCs_all)

AUCs_all_mean = np.mean(AUCs_all, axis=0)
print(AUCs_all_mean)
# exit()


def draw_ROC_all(TPRs_all, FPRs_all, AUCs_all, dataname, GL=True):
    colors = ['#4C6643', '#AFCFA6', '#F5D44B', '#D47828', '#B73508']
    noise_dbs = [100, 80, 60, 40, 20]
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.set_title('{} ROC'.format(dataname))
    for dataid in range(1):
        TPRs, FPRs, AUCs = TPRs_all[dataid], FPRs_all[dataid], AUCs_all[dataid]

        for i in range(5):
            fpr, tpr, auc, noise_db = FPRs[i], TPRs[i], AUCs[i], noise_dbs[i]
            ax.plot(
                fpr,
                tpr,
                c=colors[i],
                linestyle='--',
                label='SNR {} dB, AUC:{}'.format(noise_db, round(auc, 4)),
            )

        if GL:  # 当以truth 为 base 的时候，多一个 GL 的曲线。
            fpr, tpr, auc = FPRs[-1], TPRs[-1], AUCs[-1]
            ax.plot(fpr, tpr, c='black', linestyle='--', label='GL, auc={}'.format(round(auc, 4)))

    # ax.legend()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.subplots_adjust(left=0.1, right=0.92, top=0.92, bottom=0.1)
    plt.grid()
    # plt.show()
    # plt.savefig('toy/results/{}/{}_ROC.pdf'.format(dataname, dataname), dpi=300)


draw_ROC_all(TPRs_all, FPRs_all, AUCs_all, dataname, GL=True)
'''


