import numpy as np
import pandas as pd
from utils import *
import os

dataname = 'toy_100'
res_dir = 'results/{}/'.format(dataname)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)




def generate_data_sparse(num_nodes, num_samples, num_cases=1):
    '''随机的 presicion '''

    np.random.seed(1)

    res_dir = 'data/sparse_p{}_n{}/'.format(num_nodes, num_samples)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    threshold = 1e-10

    sparsity = 0.2
    non_zeros = int(num_nodes * num_nodes * sparsity) - num_nodes  # subtract diagonal elements
    print("non_zeros", non_zeros)

    for data_id in range(num_cases):
        upper_tri_indices = np.triu_indices(num_nodes, k=1)       # List of potential off-diagonal positions
        shuffled_indices = np.random.permutation(len(upper_tri_indices[0]))[:non_zeros // 2]      # Shuffle and pick the first few indices

        precision = np.zeros((num_nodes, num_nodes))

        # Set the selected off-diagonal elements to 0.5
        for idx in shuffled_indices:
            i = upper_tri_indices[0][idx]   # 根据 idx 横向选一个 i
            j = upper_tri_indices[1][idx]   # 根据 idx 纵向选一个 j
            precision[i, j] = precision[j, i] = 0.5

        # precision = np.zeros((num_nodes, num_nodes))
        # for i in range(num_nodes):
        #     precision[i, i] = 1
        #     if i > 0:
        #         precision[i, i-1] = precision[i-1, i] = 0.5

        # # 随机交换行和列数百次
        # np.random.seed(41)  # 设置随机种子以确保结果的可重复性
        # for _ in range(100):  # 交换操作执行300次，可以根据需要调整
        #    i, j = np.random.choice(num_nodes, 2, replace=False)  # 随机选择两个不同的索引
        #    precision[[i, j], :] = precision[[j, i], :]          # 交换i和j行
        #    precision[:, [i, j]] = precision[:, [j, i]]          # 交换i和j列

        draw_precision(precision, dataname, noise_level="Truth")

        # partial_corr = precision_to_partial_corr(precision)
        # draw_partial_corr(abs(partial_corr), dataname, noise_level="Truth", cmap=None)


        covariance = np.linalg.inv(precision)
        # plt.figure()
        # plt.imshow(covariance)
        # plt.title("covariance")
        # # plt.show()

        mean = np.zeros(num_nodes)
        samples = np.random.multivariate_normal(mean, covariance, size=num_samples)
        print(samples.shape)

        # get the 'truth' adj_matrix
        np.fill_diagonal(precision, 0)
        precision = np.abs(precision)
        adj_matrix = np.where(precision >= threshold, 1, 0)

        np.save(res_dir + 'toy_adj_matrix_{}.npy'.format(data_id), adj_matrix)
        df = pd.DataFrame(samples, columns=[f'Node_{i}' for i in range(num_nodes)])
        df.to_csv(res_dir + 'toy_data_{}.csv'.format(data_id), index=False)

num_nodes = 100
num_samples = 1000

generate_data_sparse(num_nodes, num_samples, num_cases=1)

time1 = time.time()
for data_id in range(1):  # 10 次实验
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

    # section 直接计算得到的 precision_d，而不是 GL 得到的:
    precision_d = np.linalg.inv(cov_matrix)
    partial_corr_d = precision_to_partial_corr(precision_d)
    draw_partial_corr(abs(partial_corr_d), dataname, noise_level="direct", cmap="Oranges")

    np.fill_diagonal(precision_d, 0)
    draw_precision(abs(precision_d), dataname, noise_level="direct")


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

    # section 画 pred partial corr
    partial_corr_ = precision_to_partial_corr(precision_)
    draw_partial_corr(abs(partial_corr_), dataname, noise_level=0, cmap="Oranges")

    # section 画 pred precision_
    np.fill_diagonal(precision_, 0)
    precision_ = np.abs(precision_)
    draw_precision(precision_, dataname, noise_level=0)


    # section 阈值 得到 adjacency_matrix
    adjacency_matrix_pred = np.where(precision_ >= threshold, 1, 0)
    draw_adjacency_matrix(adjacency_matrix_pred, dataname, noise_level=0)

    # section 画 pred graph
    # draw_graph_toy_1(adjacency_matrix_pred, dataname, noise_level=0)

    plt.show()




