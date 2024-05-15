import random
import networkx as nx
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from utils import *
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

dataname = "cell_signal"
res_dir = './results/{}/'.format(dataname)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

def get_cell_signal_data():
    data_path = "/Users/muhaoguo/Documents/study/Paper_Projects/Datasets/Garph_Data/protein.data"
    X = np.loadtxt(data_path)
    protein_names = [
        'Raf', 'Mek', 'Plcg', 'PIP2', 'PIP3',
        'Erk', 'Akt', 'PKA', 'PKC', 'P38', 'Jnk']
    tmp = {name: name for name in protein_names}
    connection_true = [[0, 1], [0, 7], [0, 8],
                       [1, 0], [1, 5], [1, 7], [1, 8],
                       [2, 3], [2, 4], [2, 8],
                       [3, 2], [3, 4], [3, 8],
                       [4, 2], [4, 3], [4, 6],
                       [5, 1], [5, 6], [5, 7],
                       [6, 4], [6, 5], [6, 7],
                       [7, 0], [7, 1], [7, 5], [7, 6], [7, 9], [7, 10],
                       [8, 0], [8, 1], [8, 2], [8, 3], [8, 9], [8, 10],
                       [9, 7], [9, 8],
                       [10, 7], [10, 8]]
    connection_true = np.array(connection_true)
    n = len(tmp)
    adj_matrix = np.zeros((n, n), dtype=int)
    for connection in connection_true:
        i, j = connection
        adj_matrix[i][j] = 1
    adj_matrix = np.array(adj_matrix)
    return X, tmp, adj_matrix, connection_true, protein_names

def draw_graph_cell_signal(adj_matrix, dataname, noise_level=0, protein_names=None):
    # 从 adj_matrix 到 connection
    connections = []
    n = len(adj_matrix)  # Assuming the matrix is square
    for i in range(n):
        for j in range(i, n):  # Start from 'i' to avoid duplicate pairs for undirected graph
            if adj_matrix[i][j] == 1:
                connections.append([i, j])
    connections = np.array(connections)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    plt.title("{} | graph | noise: {} db".format(dataname, noise_level))

    angles = np.linspace(0, 1, len(protein_names) + 1)[:-1] * 2 * np.pi + np.pi / 2
    G, node_pos = nx.Graph(), {}
    for i, node in enumerate(protein_names):
        G.add_node(node)
        node_pos[node] = np.array([np.cos(angles[i]), np.sin(angles[i])])
    for i in range(connections.shape[0]):
        G.add_edge(protein_names[connections[i, 0]], protein_names[connections[i, 1]])

    nx.draw(G, node_pos, node_size=60, with_labels=False, ax=ax, edge_color="#4C6643", width=1, node_color="#AFCFA6", )  # node_color='#174A7E'

    description = nx.draw_networkx_labels(G, node_pos, labels=tmp, ax=ax)
    for (i, (node, t)) in enumerate(description.items()):
        t.set_position((np.cos(angles[i]), np.sin(angles[i]) + 0.08))
        t.set_fontsize(25)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    # ax.text(0, 1.18, 'True connection', fontsize=8)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig("./results/{}/{}_graph_noise_{}.pdf".format(dataname, dataname, noise_level),dpi=300)


X, tmp, adj_matrix, connection_true, protein_names = get_cell_signal_data()
DATA = X

# note Truth part ############################################################################
#section 画 adjacency_matrix
draw_adjacency_matrix(adj_matrix, dataname, noise_level = "Truth")

# section 画 graph ground Truth:
draw_graph_cell_signal(adj_matrix=adj_matrix, dataname=dataname, noise_level="Truth", protein_names=protein_names)


def one_time(seed):
    # note model part ############################################################################
    #section 预处理：标准化节点特征?
    # scaler = StandardScaler()   # 去均值和方差归一化： 对每一个特征维度来做的，而不是针对样本。
    # DATA = scaler.fit_transform(DATA)

    # seed = 1
    variance = np.var(DATA)
    L, N = DATA.shape
    print("DATA L, N:", L, N)
    print("variance of X:", variance)
    norm = False
    cross_val =False
    lambdas_exp = range(-6, 10)
    threshold = 0.0000001
    n_splits=10

    # --以下可以复制-------------------
    #section cov_matrix:
    cov_matrix = np.cov(DATA, rowvar=False)  # Calculate the sample covariance matrix
    draw_cov_matrix(cov_matrix, dataname)

    # section 直接计算得到的 precision_d，而不是 GL 得到的:
    precision_d = np.linalg.inv(cov_matrix)
    partial_corr_d = precision_to_partial_corr(precision_d)
    draw_partial_corr(abs(partial_corr_d), dataname, noise_level="direct", cmap="Oranges")

    np.fill_diagonal(precision_d, 0)
    draw_precision(abs(precision_d), dataname, noise_level="direct")

    # section cross-val 选择 lambda
    if cross_val:
        # lambdas_exp = range(1, 6)
        lambdas = [10**i for i in lambdas_exp] # this is for cd method

        SCOREs2 = cross_validation_for_lambda(lambdas, DATA, n_splits=n_splits)

        fig = plt.figure(figsize=(6, 2))
        plt.subplots_adjust(left=0.15, right=0.95, top=0.82, bottom=0.25)
        ax = fig.add_subplot(111)
        ax.set_title("Cross validation for lambda")
        ax.scatter(lambdas_exp, SCOREs2, label="mse")
        ax.set_xlabel("$\lambda$")
        ax.set_ylabel("MSE")
        ax.set_xticks(lambdas_exp, ['$10^{{{}}}$'.format(i) for i in lambdas_exp])  #  使用双大括号 {{}} 来转义大括号，使其在 format 调用中不被解释为格式化占位符
        ax.grid()
        plt.savefig("./results/{}/{}_cross_validation_lambda.png".format(dataname, dataname), dpi=300)
        plt.show()

    #section GL 计算获得 precision_
    best_rho2 = 10**4  # todo BASED on observation
    covariance_, precision_, costs, iter = cd_graphical_lasso(emp_cov = cov_matrix, alpha = best_rho2)

    partial_corr = precision_to_partial_corr(precision_)
    draw_partial_corr(abs(partial_corr), dataname, noise_level = 0, cmap="Oranges")

    np.fill_diagonal(precision_, 0)
    precision_ = np.abs(precision_)

    draw_precision(precision_, dataname, noise_level = 0)

    # section 阈值 得到 adjacency_matrix
    adjacency_matrix_pred = np.where(precision_ >= threshold, 1, 0)

    draw_adjacency_matrix(adjacency_matrix_pred, dataname, noise_level=0)

    #section 画 pred graph
    draw_graph_cell_signal(adjacency_matrix_pred, dataname, noise_level=0, protein_names=protein_names)

    plt.show()

    # note noise part ############################################################################
    if norm:
        noise_stds = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0]
    else:
        noise_stds = [np.sqrt(variance/(10**10)), np.sqrt(variance/(10**8)), np.sqrt(variance/(10**6)), np.sqrt(variance/(10**4)), np.sqrt(variance/(10**2)), 0]
    print("noise_stds:", noise_stds)
    noise_dbs_for_draw = [100, 80, 60, 40, 20, "No"]

    FPRs = []
    TPRs = []
    AUCs = []
    DATA_noises = []
    for i, noise_std in enumerate(noise_stds):
        DATA_noise = add_noise_for_time_series(DATA, noise_std_dev=noise_std, seed=seed)
        DATA_noises.append(DATA_noise)
        print("noise_std:", noise_std, noise_dbs_for_draw[i],"db", "L:", L, "N:", N)

        # section cov_matrix_noise
        cov_matrix_noise = np.cov(DATA_noise, rowvar=False)  # Calculate the sample covariance matrix

        # section cov_matrix re-fine
        term = (1 - 1 /L) * (noise_std ** 2)
        cov_matrix_noise = cov_matrix_noise - term * np.eye(N)

        draw_cov_matrix(cov_matrix_noise, dataname, noise_level=noise_dbs_for_draw[i])

        # section precision_noise calculate
        covariance_noise, precision_noise, costs, iter = cd_graphical_lasso(emp_cov=cov_matrix_noise, alpha=best_rho2)

        #section precision_ ----> 对角线0 + MAX-MIN normlize
        np.fill_diagonal(precision_noise, 0)
        precision_noise = np.abs(precision_noise)

        draw_precision(precision_noise, dataname, noise_level=noise_dbs_for_draw[i])  # todo

        # section pred adjacency_matrix_pred
        adjacency_matrix_pred_noise = np.where(precision_noise >= threshold, 1, 0)
        draw_adjacency_matrix(adjacency_matrix_pred_noise, dataname,noise_level=noise_dbs_for_draw[i])

        # section connection pred graph

        draw_graph_cell_signal(adj_matrix = adjacency_matrix_pred_noise, dataname = dataname, noise_level=noise_dbs_for_draw[i], protein_names = protein_names)

        # section ROC
        fpr, tpr, _ = roc_curve(y_true=adj_matrix.ravel(),  y_score=precision_noise.ravel())  # todo adj_matrix  adjacency_matrix_pred
        auc = roc_auc_score(adj_matrix.ravel(), precision_noise.ravel())                      # todo adj_matrix  adjacency_matrix_pred
        TPRs.append(tpr)
        FPRs.append(fpr)
        AUCs.append(auc)

    draw_ROC(TPRs, FPRs, AUCs, dataname, GL=True)
    plt.show()

    return AUCs


if __name__ == '__main__':
    time1 = time.time()
    times = 1
    AUCs_all = []
    for i in range(times):
        AUCs = one_time(seed=1)
        print("AUCs", AUCs)
        AUCs_all.append(AUCs)

    print(AUCs_all)
    print("mean", np.mean(AUCs_all, axis=0))
    print("std", np.std(AUCs_all, axis=0))

    time2 = time.time()
    print("time", time2- time1)
