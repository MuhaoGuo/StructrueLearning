import random
import networkx as nx
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from utils import *
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

dataname = "movie_ori_rate"
res_dir = './results/{}/'.format(dataname)
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

def get_movie_data():
    data_path1 = "/Users/muhaoguo/Documents/study/Paper_Projects/Structure_learning/potientialdata/ml-1m/users.csv"
    data_path2 = "/Users/muhaoguo/Documents/study/Paper_Projects/Structure_learning/potientialdata/ml-1m/ratings.csv"
    data_path3 = "/Users/muhaoguo/Documents/study/Paper_Projects/Structure_learning/potientialdata/ml-1m/movies.csv"

    users = pd.read_csv(data_path1, sep='\t', encoding='latin-1',
                        usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

    ratings = pd.read_csv(data_path2, sep='\t', encoding='latin-1',
                          usecols=['user_id', 'movie_id', 'rating', 'timestamp'])

    movies = pd.read_csv(data_path3, sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])

    n_users = ratings.user_id.unique().shape[0]
    n_movies = ratings.movie_id.unique().shape[0]
    print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_movies))
    Ratings = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    print(Ratings)
    # exit()
    R = Ratings.values

    # user_ratings_mean = np.mean(R, axis=1)
    # Ratings_demeaned = R - user_ratings_mean.reshape(-1, 1)
    #
    # from scipy.sparse.linalg import svds
    # U, sigma, Vt = svds(Ratings_demeaned, k=50)
    # sigma = np.diag(sigma)
    # print("sigma", sigma)
    # all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    # preds = pd.DataFrame(all_user_predicted_ratings, columns=Ratings.columns)
    # # print(preds)
    # # exit()
    # data = preds.values
    return R

def draw_movie(adjacency_matrix_pred, dataname, noise_level=0):
    G = nx.from_numpy_array(np.array(adjacency_matrix_pred))
    pos = nx.kamada_kawai_layout(G)

    d = dict(G.degree)
    node_size = [300 for k in d]

    plt.figure(figsize=(8, 6))
    plt.title("{} | graph | noise: {}".format(dataname, noise_level))

    nx.draw(G, pos, width=1, with_labels=True, alpha=0.9, node_size=node_size, cmap=plt.cm.viridis, edge_color="#2a3f95", font_color='white', font_size= 9)  # "#2a3f95"

    plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
    plt.savefig("./results/{}/{}_graph_noise_{}.pdf".format(dataname, dataname, noise_level),dpi=300)



DATA = get_movie_data()

# note Truth part ############################################################################
# #section 画 adjacency_matrix
# draw_adjacency_matrix(adj_matrix, dataname, noise_level = "Truth")
#
# # section 画 graph ground Truth:
# draw_graph_cell_signal(adj_matrix=adj_matrix, dataname=dataname, noise_level="Truth", protein_names=protein_names)


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
    # precision_d = np.linalg.inv(cov_matrix)
    # partial_corr_d = precision_to_partial_corr(precision_d)
    # draw_partial_corr(abs(partial_corr_d), dataname, noise_level="direct", cmap="Oranges")
    # # plt.show()

    # np.fill_diagonal(precision_d, 0)
    # draw_precision(abs(precision_d), dataname, noise_level="direct")

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
    best_rho2 = 10 ** -0.04 # todo BASED on observation
    covariance_, precision_, costs, iter = cd_graphical_lasso(emp_cov = cov_matrix, alpha = best_rho2)

    partial_corr = precision_to_partial_corr(precision_)
    draw_partial_corr(abs(partial_corr), dataname, noise_level = 0, cmap="Oranges")

    np.fill_diagonal(precision_, 0)
    precision_ = np.abs(precision_)

    draw_precision(precision_, dataname, noise_level = 0)
    print(precision_)

    # section 阈值 得到 adjacency_matrix
    adjacency_matrix_pred = np.where(precision_ >= threshold, 1, 0)

    draw_adjacency_matrix(adjacency_matrix_pred, dataname, noise_level=0)

    #section 画 pred graph
    draw_movie(adjacency_matrix_pred, dataname, noise_level=0)


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
        DATA_noise, theta_bar = add_discrete_noise_for_time_series(DATA, noise_std_dev=noise_std, seed=seed)  # todo discrete
        DATA_noises.append(DATA_noise)
        print("noise_std:", noise_std, noise_dbs_for_draw[i],"db", "L:", L, "N:", N)

        # section cov_matrix_noise
        cov_matrix_noise = np.cov(DATA_noise, rowvar=False)  # Calculate the sample covariance matrix

        # section cov_matrix re-fine
        term = (1 - 1 /L) * (theta_bar) # todo discrete
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
        draw_movie(adjacency_matrix_pred, dataname, noise_level=noise_dbs_for_draw[i])

        # section ROC
        fpr, tpr, _ = roc_curve(y_true=adjacency_matrix_pred.ravel(),  y_score=precision_noise.ravel())  # todo adj_matrix  adjacency_matrix_pred
        auc = roc_auc_score(adjacency_matrix_pred.ravel(), precision_noise.ravel())                      # todo adj_matrix  adjacency_matrix_pred
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
