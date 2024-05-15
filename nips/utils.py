import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import GraphicalLasso
from sklearn.covariance import GraphicalLassoCV
# import gglasso
import warnings
from scipy import linalg
import sklearn.linear_model._cd_fast as cd_fast
from sklearn.utils.validation import (_is_arraylike_not_scalar, check_random_state, check_scalar,)
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans
import time
from sklearn.model_selection import KFold

def calculate_edge_proportion(adj_matrix):
    # 确保邻接矩阵是方阵
    assert adj_matrix.shape[0] == adj_matrix.shape[1], "邻接矩阵必须是方阵"
    n = adj_matrix.shape[0]  # 节点数量
    total_possible_edges = n * (n - 1) / 2  # 无向图中可能的边的总数
    # 计算实际存在的边的数量
    # 对于无向图，邻接矩阵是对称的，所以我们只考虑上三角矩阵
    actual_edges = np.sum(np.triu(adj_matrix) != 0)
    # 计算边所占的比例
    edge_proportion = actual_edges / total_possible_edges
    return edge_proportion

def calculate_objective(Theta, empirical_cov, lambda_val):
    """
    计算公式 \log\det (\Theta) - \Tr(\hat{\Sigma}\Theta) - \lambda \|\Theta\|_1 的值。
    参数:
    Theta (np.array): 正定矩阵 Theta。
    empirical_cov (np.array): 经验协方差矩阵 \hat{\Sigma}。
    lambda_val (float): 正则化参数 \lambda。
    返回:
    float: 公式的计算结果。
    """
    # 计算 log(det(Theta))
    log_det_Theta = np.log(np.linalg.det(Theta))

    # 计算 Tr(empirical_cov * Theta)
    trace_term = np.trace(empirical_cov @ Theta)

    # 计算 L1 范数 ||Theta||_1
    l1_norm = np.sum(np.abs(Theta))

    # 计算整个公式的值
    objective_value = log_det_Theta - trace_term - lambda_val * l1_norm

    return objective_value


def Kmeans_create_adjacency_matrix(precision_matrix):
    # 获取precision matrix的形状
    n = precision_matrix.shape[0]

    # 创建一个掩码数组，对于对角线元素为True，其他为False
    mask = np.eye(n, dtype=bool)

    # 将precision matrix重塑为二维数组，每个元素作为一个观测，但忽略对角线元素
    X = precision_matrix[~mask].reshape(-1, 1)

    max_value = X.max()
    min_value = X.min()
    init_centers = np.array([[min_value], [max_value]])

    # 应用KMeans聚类，分为两类
    kmeans = KMeans(n_clusters=2, init=init_centers, random_state=0, n_init='auto').fit(X)

    # 获取聚类标签
    labels = kmeans.labels_

    # 确定哪个聚类标签对应于大值
    big_value_cluster = np.argmax(kmeans.cluster_centers_.flatten())

    # 创建一个全0的邻接矩阵，准备填充
    adjacency_matrix = np.zeros_like(precision_matrix)

    # 仅对非对角线元素填充聚类结果，大值为1，小值为0
    adjacency_matrix[~mask] = (labels == big_value_cluster).astype(int)

    # 对角线元素已经是0，无需额外操作

    return adjacency_matrix

# def soft_thresholding(x, rho):
#     return np.sign(x) * np.maximum(np.abs(x) - rho, 0.0)
#
# # Graphical Lasso
# def graphical_lasso(S, rho, max_iter=100000, tol=1e-10):
#     """
#     Parameters:
#     - X: Data matrix of shape (n_samples, n_features)
#     - S: Empirical covariance matrix.
#     - W: Estimated covariance matrix.
#     - Theta: Precision matrix, the inverse of W
#     - rho: Regularization parameter.
#
#     Returns:
#     - Theta: Precision matrix, the inverse of W
#     - W: Estimated covariance matrix.
#     """
#
#     # n is the number of samples. p is the number of features.
#     p = S.shape[0]
#
#     # Initial solution for W
#     W = S + rho * np.eye(p)
#     W_old = W.copy()
#
#     for iteration in range(max_iter):
#         for j in range(p):
#             # Partition matrix
#             idx = np.arange(p) != j
#             W_11 = W[idx][:, idx]
#             s_12 = S[j, idx]
#
#             # Initialize beta
#             beta = np.linalg.solve(W_11, s_12)
#             beta_old = beta.copy()
#
#             V = W_11
#             u = s_12
#
#             for i in range(p - 1):
#                 residual = u[i] - V[i, :] @ beta + V[i, i] * beta[i]
#                 beta[i] = soft_thresholding(residual, rho) / V[i, i]
#                 # beta[i] = soft_thresholding(beta[i], rho) / V[i, i]
#                 # print(beta)
#                 # plt.figure()
#                 # plt.hist(beta, bins=30, alpha=0.75, color='blue', edgecolor='black')
#                 # plt.show()
#                 # print(beta[i])
#                 # exit()
#                 # if abs(beta[i]) < 0.05:
#                 #     beta[i] = 0
#             # Update W
#             W[j, idx] = W_11 @ beta
#             W[idx, j] = W[j, idx]
#
#         # Check for convergence
#         if np.mean(np.abs(W - W_old)) < tol:
#             break
#
#         W_old = W.copy()
#
#     return W, np.linalg.inv(W)


def draw_partial_corr(partial_corr, dataname, noise_level = 0, cmap = "Reds"):
    plt.figure(figsize=(6, 5))
    plt.imshow(partial_corr, cmap=cmap, interpolation="none")  # Greens   Greys   Oranges
    plt.colorbar()
    plt.xlabel("Node")
    plt.ylabel("Node")
    plt.title("{} | partial_corr | noise:{} db".format(dataname, noise_level))
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    plt.savefig("./results/{}/{}_partial_corr_noise_{}.png".format(dataname, dataname, noise_level), dpi=300)


def draw_cov_matrix(cov_matrix, dataname, noise_level = 0, cmap = "Greens"):
    plt.figure(figsize=(6, 5))
    plt.imshow(cov_matrix, cmap=cmap, interpolation="none")  # Greens   Greys   Oranges
    plt.colorbar()
    plt.xlabel("Node")
    plt.ylabel("Node")
    plt.title("{} | cov_matrix | noise:{} db".format(dataname, noise_level))
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    plt.savefig("./results/{}/{}_cov_matrix_noise_{}.png".format(dataname, dataname, noise_level), dpi=300)

def draw_precision(precision_, dataname, noise_level = 0, cmap = "Oranges"):
    plt.figure(figsize=(6, 5))
    plt.imshow(precision_, cmap=cmap, interpolation="none")  # Greens   Greys   Oranges
    plt.colorbar()
    plt.xlabel("Node")
    plt.ylabel("Node")
    plt.title("{} | precision | noise:{} db".format(dataname, noise_level))
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    plt.savefig("./results/{}/{}_precision_noise_{}.png".format(dataname, dataname, noise_level), dpi=300)

def draw_adjacency_matrix(adjacency_matrix_pred, dataname, noise_level = 0):
    plt.figure(figsize=(6, 5))
    plt.title("{} | adjacency_matrix | noise:{} db".format(dataname, noise_level))
    plt.imshow(adjacency_matrix_pred, cmap="Greys", interpolation="none")  # Greens   Greys   Oranges
    # plt.colorbar()
    plt.xlabel("Node")
    plt.ylabel("Node")
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    plt.savefig("./results/{}/{}_adjacency_matrix_noise_{}.png".format(dataname, dataname, noise_level), dpi=300)

def add_noise_for_time_series(data, mu=0, noise_std_dev=0, seed=3):  #3
    '''
    data : time_series: (#Time, #Node)
    '''
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(loc=mu, scale=noise_std_dev, size=data.shape)
    noisy_data = data + noise
    return noisy_data

def add_discrete_noise_for_time_series(data, mu=0, noise_std_dev=0, seed=3):  #3
    '''
    data : time_series: (#Time, #Node)
    '''
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(loc=mu, scale=noise_std_dev, size=data.shape)
    noise = np.round(noise).astype(int)

    print(noise)
    # print(data)
    noisy_data = data + noise

    # plt.figure()
    # plt.title("data")
    # plt.imshow(data)
    # plt.figure()
    # plt.title("noise")
    # plt.imshow(noise)
    # plt.figure()
    # plt.title("noisy_data")
    # plt.imshow(noisy_data)
    # plt.show()


    noise_samples = np.random.normal(loc=mu, scale=noise_std_dev, size=10000)
    noise_samples = np.round(noise_samples).astype(int)
    theta_bar = np.var(noise_samples)
    # print(noisy_data)


    return noisy_data, theta_bar

def draw_ROC(TPRs, FPRs, AUCs, dataname, GL=True):
    colors = ["#4C6643", "#AFCFA6", "#F5D44B", "#D47828", "#B73508"]
    # noise_std_db = {0.00001: 100, 0.0001: 80, 0.001: 60, 0.01: 40, 0.1: 20}
    # noise_std_db = {key * variance : noise_std_db[key] for key in noise_std_db.keys()}
    noise_dbs = [100, 80, 60, 40, 20]
    lines_width = [2.2, 1.9, 1.6, 1.3, 1]
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    ax.set_title("{} ROC".format(dataname))
    if GL:  # 当以truth 为 base 的时候，多一个 GL 的曲线。
        fpr, tpr, auc = FPRs[-1], TPRs[-1], AUCs[-1]
        ax.plot(fpr, tpr, c="black", linestyle="-", linewidth = 2.2, label="GL, AUC:{}".format(round(auc, 4)))

    for i in range(5):
        fpr, tpr, auc, noise_db = FPRs[i], TPRs[i], AUCs[i], noise_dbs[i]
        ax.plot(fpr, tpr,  c=colors[i], linestyle="-.", linewidth = lines_width[i], label="SNR {} dB, AUC:{}".format(noise_db, round(auc, 4)))

    ax.legend(framealpha =0.5)
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate',fontsize=14)
    plt.subplots_adjust(left=0.2, right=0.92, top=0.92, bottom=0.2)
    plt.grid()
    plt.savefig("./results/{}/{}_ROC.pdf".format(dataname, dataname), dpi=300)

def calculate_objective(Theta, empirical_cov, lambda_val):
    """
    计算公式 \log\det (\Theta) - \Tr(\hat{\Sigma}\Theta) - \lambda \|\Theta\|_1 的值。
    参数:
    Theta (np.array): 正定矩阵 Theta。
    empirical_cov (np.array): 经验协方差矩阵 \hat{\Sigma}。
    lambda_val (float): 正则化参数 \lambda。
    返回:
    float: 公式的计算结果。
    """
    # 计算 log(det(Theta))
    log_det_Theta = np.log(np.linalg.det(Theta))

    # 计算 Tr(empirical_cov * Theta)
    trace_term = np.trace(empirical_cov @ Theta)

    # 计算 L1 范数 ||Theta||_1
    l1_norm = np.sum(np.abs(Theta))

    # 计算整个公式的值
    objective_value = log_det_Theta - trace_term - lambda_val * l1_norm

    return objective_value


def fast_logdet(A):
    """Compute logarithm of determinant of a square matrix.

    The (natural) logarithm of the determinant of a square matrix
    is returned if det(A) is non-negative and well defined.
    If the determinant is zero or negative returns -Inf.

    Equivalent to : np.log(np.det(A)) but more robust.

    Parameters
    ----------
    A : array_like of shape (n, n)
        The square matrix.

    Returns
    -------
    logdet : float
        When det(A) is strictly positive, log(det(A)) is returned.
        When det(A) is non-positive or not defined, then -inf is returned.

    See Also
    --------
    numpy.linalg.slogdet : Compute the sign and (natural) logarithm of the determinant
        of an array.

    Examples
    --------
    # >>> import numpy as np
    # >>> from sklearn.utils.extmath import fast_logdet
    # >>> a = np.array([[5, 1], [2, 8]])
    # >>> fast_logdet(a)
    3.6375861597263857
    """
    sign, ld = np.linalg.slogdet(A)
    if not sign > 0:
        return -np.inf
    return ld

def _dual_gap(emp_cov, precision_, alpha):
    """Expression of the dual gap convergence criterion

    The specific definition is given in Duchi "Projected Subgradient Methods for Learning Sparse Gaussians".
    """
    gap = np.sum(emp_cov * precision_)
    gap -= precision_.shape[0]
    gap += alpha * (np.abs(precision_).sum() - np.abs(np.diag(precision_)).sum())
    return gap

def _objective(mle, precision_, alpha):
    """Evaluation of the graphical-lasso objective function

    the objective function is made of a shifted scaled version of the
    normalized log-likelihood (i.e. its empirical mean over the samples) and a
    penalisation term to promote sparsity
    """
    p = precision_.shape[0]
    cost = -2.0 * log_likelihood(mle, precision_) + p * np.log(2 * np.pi)
    cost += alpha * (np.abs(precision_).sum() - np.abs(np.diag(precision_)).sum())
    return cost

def log_likelihood(emp_cov, precision):
    """Compute the sample mean of the log_likelihood under a covariance model.

    Computes the empirical expected log-likelihood, allowing for universal
    comparison (beyond this software package), and accounts for normalization
    terms and scaling.

    Parameters
    ----------
    emp_cov : ndarray of shape (n_features, n_features)
        Maximum Likelihood Estimator of covariance.

    precision : ndarray of shape (n_features, n_features)
        The precision matrix of the covariance model to be tested.

    Returns
    -------
    log_likelihood_ : float
        Sample mean of the log-likelihood.
    """
    p = precision.shape[0]
    log_likelihood_ = -np.sum(emp_cov * precision) + fast_logdet(precision)
    log_likelihood_ -= p * np.log(2 * np.pi)
    log_likelihood_ /= 2.0
    return log_likelihood_

def cd_graphical_lasso(
    emp_cov, alpha,
    *, cov_init=None,mode="cd", tol=1e-4,enet_tol=1e-4,max_iter=100,verbose=False,eps=np.finfo(np.float64).eps,):
    _, n_features = emp_cov.shape
    if alpha == 0:
        precision_ = linalg.inv(emp_cov)
        cost = -2.0 * log_likelihood(emp_cov, precision_)
        cost += n_features * np.log(2 * np.pi)
        d_gap = np.sum(emp_cov * precision_) - n_features
        return emp_cov, precision_, (cost, d_gap), 0

    if cov_init is None:
        covariance_ = emp_cov.copy()
    else:
        covariance_ = cov_init.copy()
    # As a trivial regularization (Tikhonov like), we scale down the
    # off-diagonal coefficients of our starting point: This is needed, as
    # in the cross-validation the cov_init can easily be
    # ill-conditioned, and the CV loop blows. Beside, this takes
    # conservative stand-point on the initial conditions, and it tends to
    # make the convergence go faster.
    covariance_ *= 0.95
    diagonal = emp_cov.flat[:: n_features + 1]
    covariance_.flat[:: n_features + 1] = diagonal
    precision_ = linalg.pinvh(covariance_)

    indices = np.arange(n_features)
    i = 0  # initialize the counter to be robust to `max_iter=0`
    costs = list()
    # The different l1 regression solver have different numerical errors
    if mode == "cd":
        errors = dict(over="raise", invalid="ignore")
    else:
        errors = dict(invalid="raise")
    try:
        # be robust to the max_iter=0 edge case, see:
        # https://github.com/scikit-learn/scikit-learn/issues/4134
        d_gap = np.inf
        # set a sub_covariance buffer
        sub_covariance = np.copy(covariance_[1:, 1:], order="C")
        for i in range(max_iter):
            for idx in range(n_features):
                # To keep the contiguous matrix `sub_covariance` equal to
                # covariance_[indices != idx].T[indices != idx]
                # we only need to update 1 column and 1 line when idx changes
                if idx > 0:
                    di = idx - 1
                    sub_covariance[di] = covariance_[di][indices != idx]
                    sub_covariance[:, di] = covariance_[:, di][indices != idx]
                else:
                    sub_covariance[:] = covariance_[1:, 1:]
                row = emp_cov[idx, indices != idx]
                with np.errstate(**errors):
                    if mode == "cd":
                        # Use coordinate descent
                        coefs = -(precision_[indices != idx, idx] / (precision_[idx, idx] + 1000 * eps) )
                        coefs, _, _, _ = cd_fast.enet_coordinate_descent_gram(
                            coefs, alpha, 0, sub_covariance, row, row,  max_iter,enet_tol, check_random_state(None), False)  # todo None -> check_random_state(None)
                # Update the precision matrix
                precision_[idx, idx] = 1.0 / (covariance_[idx, idx] - np.dot(covariance_[indices != idx, idx], coefs) )
                precision_[indices != idx, idx] = -precision_[idx, idx] * coefs
                precision_[idx, indices != idx] = -precision_[idx, idx] * coefs
                coefs = np.dot(sub_covariance, coefs)
                covariance_[idx, indices != idx] = coefs
                covariance_[indices != idx, idx] = coefs

            if not np.isfinite(precision_.sum()):
                raise FloatingPointError(
                    "The system is too ill-conditioned for this solver"
                )
            # todo our algorithm========================
            # L = 1000
            # noise_std = 0.0001
            # term = (1 - 1 / L) * (noise_std ** 2)
            # emp_cov -= term * np.eye(len(emp_cov))
            # todo =====================================
            d_gap = _dual_gap(emp_cov, precision_, alpha)
            cost = _objective(emp_cov, precision_, alpha)

            if verbose:
                print(
                    "[graphical_lasso] Iteration % 3i, cost % 3.2e, dual gap %.3e"
                    % (i, cost, d_gap)
                )
            costs.append((cost, d_gap))
            if np.abs(d_gap) < tol:
                break
            if not np.isfinite(cost) and i > 0:
                raise FloatingPointError(
                    "Non SPD result: the system is too ill-conditioned for this solver"
                )
        else:
            warnings.warn(
                "graphical_lasso: did not converge after %i iteration: dual gap: %.3e"
                % (max_iter, d_gap),
                ConvergenceWarning,
            )
    except FloatingPointError as e:
        e.args = (e.args[0] + ". The system is too ill-conditioned for this solver",)
        raise e

    return covariance_, precision_, costs, i + 1


def cross_validation_for_lambda(lambdas, DATA, n_splits=10):
    kf = KFold(n_splits=n_splits)
    best_score = float('-inf')
    best_rho = None
    SCOREs2 = []
    for i in range(len(lambdas)):
        rho = lambdas[i]
        scores = []
        print("cross-validation - lambda:{}".format(rho))
        try:
            for train_index, test_index in kf.split(DATA):
                X_train, X_test = DATA[train_index], DATA[test_index]

                cov_matrix_train = np.cov(X_train, rowvar=False)
                cov_matrix_test = np.cov(X_test, rowvar=False)

                model = GraphicalLasso(covariance='precomputed', alpha=rho, max_iter=10000, tol=1e-4,  enet_tol=1e-4) # tol=1e-4,  enet_tol=1e-4
                model.fit(cov_matrix_train)
                precision_train = model.precision_
                covariance_train = model.covariance_

                model.fit(cov_matrix_test)
                precision_test = model.precision_
                covariance_test = model.covariance_

                # print(covariance_train)
                # print(covariance_test)

                mse = np.mean((covariance_train - covariance_test) ** 2)
                print(len(scores)+1, "-mse:", mse)
                scores.append(mse)
            average_score = np.mean(scores)
        except:
            average_score = -np.inf

        if average_score > best_score:
            best_score = average_score
            best_rho = rho
        SCOREs2.append(average_score)

    # print("method 2 | best_rho", best_rho)
    best_rho2 = best_rho # this is not the best score
    return SCOREs2

def cross_validation_for_lambda2(lambdas, DATA, n_splits=10):
    kf = KFold(n_splits=n_splits)
    best_score = float('-inf')
    best_rho = None
    SCOREs2 = []
    for i in range(len(lambdas)):
        rho = lambdas[i]
        scores = []
        print("cross-validation - lambda:{}".format(rho))
        try:
            for train_index, test_index in kf.split(DATA):
                X_train, X_test = DATA[train_index], DATA[test_index]

                # model = GraphicalLasso(covariance='precomputed', alpha=rho, max_iter=10000, tol=1e-3,
                #                        enet_tol=1e-3)  # tol=1e-4,  enet_tol=1e-4

                cov_matrix_train = np.cov(X_train, rowvar=False)
                W, precision_train = cd_graphical_lasso(cov_matrix_train, alpha=rho)

                cov_matrix_test = np.cov(X_test, rowvar=False)
                score = calculate_objective(precision_train, cov_matrix_test, rho)
                print(len(scores) + 1, "score:", score)
                scores.append(score)
            average_score = np.mean(scores)
        except:
            average_score = -np.inf

        if average_score > best_score:
            best_score = average_score
            best_rho = rho
        SCOREs2.append(average_score)

    # print("method 2 | best_rho", best_rho)
    best_rho2 = best_rho  # this is not the best score
    return SCOREs2


############ 以下是 ADMM 的实现 ############################

def objective(S, X, Z, lambda_val):
    # return np.trace(S @ X) - np.linalg.slogdet(X)[1] + lambda_val * np.linalg.norm(Z, 1)
    return np.trace(S @ X) - np.log(np.linalg.slogdet(X))+ lambda_val * np.linalg.norm(Z, 1)

def shrinkage(a, kappa):
    return np.maximum(0, a - kappa) - np.maximum(0, -a - kappa)


# admm （输入 data）
def covsel(D, lambda_val, rho, alpha):
    '''
    % Solves the following problem via ADMM: minimize  trace(S*X) - log det X + lambda*||X||_1
    % rho is the augmented Lagrangian parameter.
    % alpha is the over-relaxation parameter (typical values for alpha are between 1.0 and 1.8).
    '''
    # 开始计时
    t_start = time.time()

    # 设置参数
    QUIET = False
    MAX_ITER = 50000
    ABSTOL = 1e-4
    RELTOL = 1e-2

    # 计算经验协方差矩阵
    S = np.cov(D, rowvar=False)
    n = S.shape[0]

    # 初始化变量
    X = np.zeros((n, n))
    Z = np.zeros((n, n))
    U = np.zeros((n, n))

    history = {'objval': [], 'r_norm': [], 's_norm': [], 'eps_pri': [], 'eps_dual': []}

    for k in range(MAX_ITER):
        # print(S)
        # Q, L = la.eig(rho * (Z - U) - S)
        # Q, L = np.linalg.eig(rho * (Z - U) - S)   # eig/ eigh ??????????
        L, Q = np.linalg.eig(rho * (Z - U) - S)   # values, vectors
        L = np.diag(L)
        es = np.diag(L)
        xi = (es + np.sqrt(es**2 + 4*rho)) / (2*rho) # 逐元素操作
        # xi = (es + np.sqrt(es.T ** 2 + 4 * rho)) / (2 * rho) # todo

        X = Q @ np.diag(xi) @ Q.T

        # Z 更新，带有松弛
        Zold = Z.copy()
        X_hat = alpha * X + (1 - alpha) * Zold
        Z = shrinkage(X_hat + U, lambda_val / rho)

        # U 更新
        U += X_hat - Z

    #     # 诊断、报告、终止检查
    #     history['objval'].append(objective(S, X, Z, lambda_val))
    #     history['r_norm'].append(np.linalg.norm(X - Z, 'fro'))
    #     history['s_norm'].append(np.linalg.norm(-rho * (Z - Zold), 'fro'))
    #     history['eps_pri'].append(np.sqrt(n*n) * ABSTOL + RELTOL * max(np.linalg.norm(X, 'fro'), np.linalg.norm(Z, 'fro')))
    #     history['eps_dual'].append(np.sqrt(n*n) * ABSTOL + RELTOL * np.linalg.norm(rho * U, 'fro'))
    #
    #     if not QUIET:
    #         print(f'{k}\t{history["r_norm"][-1]:.4f}\t{history["eps_pri"][-1]:.4f}\t{history["s_norm"][-1]:.4f}\t{history["eps_dual"][-1]:.4f}\t{history["objval"][-1]:.2f}')
    #
    #     if history['r_norm'][-1] < history['eps_pri'][-1] and history['s_norm'][-1] < history['eps_dual'][-1]:
    #         break
    #
    # if not QUIET:
    #     print(f'Total time: {time.time() - t_start} seconds')

    return Z, history


# admm （输入 Covariance）
def covsel_S(S, lambda_val, rho, alpha):
    '''
    % Solves the following problem via ADMM: minimize  trace(S*X) - log det X + lambda*||X||_1
    % rho is the augmented Lagrangian parameter.
    % alpha is the over-relaxation parameter (typical values for alpha are between 1.0 and 1.8).
    '''
    # 开始计时
    t_start = time.time()

    # 设置参数
    QUIET = False
    MAX_ITER = 500
    ABSTOL = 1e-6
    RELTOL = 1e-4

    # 计算经验协方差矩阵
    # S = np.cov(D, rowvar=False)
    n = S.shape[0]

    # 初始化变量
    X = np.zeros((n, n))
    Z = np.zeros((n, n))
    U = np.zeros((n, n))

    history = {'objval': [], 'r_norm': [], 's_norm': [], 'eps_pri': [], 'eps_dual': []}

    for k in range(MAX_ITER):
        # print(S)
        # Q, L = la.eig(rho * (Z - U) - S)
        # Q, L = np.linalg.eig(rho * (Z - U) - S)   # eig/ eigh ??????????
        L, Q = np.linalg.eig(rho * (Z - U) - S)   # values, vectors
        L = np.diag(L)
        es = np.diag(L)
        xi = (es + np.sqrt(es**2 + 4*rho)) / (2*rho) # 逐元素操作
        # xi = (es + np.sqrt(es.T ** 2 + 4 * rho)) / (2 * rho) # todo

        X = Q @ np.diag(xi) @ Q.T

        # Z 更新，带有松弛
        Zold = Z.copy()
        X_hat = alpha * X + (1 - alpha) * Zold
        Z = shrinkage(X_hat + U, lambda_val / rho)

        # U 更新
        U += X_hat - Z

    #     # 诊断、报告、终止检查
    #     history['objval'].append(objective(S, X, Z, lambda_val))
    #     history['r_norm'].append(np.linalg.norm(X - Z, 'fro'))
    #     history['s_norm'].append(np.linalg.norm(-rho * (Z - Zold), 'fro'))
    #     history['eps_pri'].append(np.sqrt(n*n) * ABSTOL + RELTOL * max(np.linalg.norm(X, 'fro'), np.linalg.norm(Z, 'fro')))
    #     history['eps_dual'].append(np.sqrt(n*n) * ABSTOL + RELTOL * np.linalg.norm(rho * U, 'fro'))
    #
    #     if not QUIET:
    #         print(f'{k}\t{history["r_norm"][-1]:.4f}\t{history["eps_pri"][-1]:.4f}\t{history["s_norm"][-1]:.4f}\t{history["eps_dual"][-1]:.4f}\t{history["objval"][-1]:.2f}')
    #
    #     if history['r_norm'][-1] < history['eps_pri'][-1] and history['s_norm'][-1] < history['eps_dual'][-1]:
    #         break
    #
    # if not QUIET:
    #     print(f'Total time: {time.time() - t_start} seconds')

    return Z, history


def precision_to_partial_corr(precision_matrix):
    """
    Convert the precision matrix to the partial correlation matrix.

    Parameters:
    precision_matrix -- The precision matrix, dimensions (n, n).

    Returns:
    partial_corr_matrix -- The partial correlation matrix, dimensions (n, n).
    """

    # Initialize the partial correlation matrix
    n = precision_matrix.shape[0]
    partial_corr_matrix = np.zeros((n, n))

    # Compute the partial correlations
    for i in range(n):
        for j in range(n):
            if i != j:
                partial_corr_matrix[i, j] = -precision_matrix[i, j] / np.sqrt(
                    precision_matrix[i, i] * precision_matrix[j, j])
            else:
                # The diagonal elements are 1 since a variable is always perfectly correlated with itself
                # For convenience, we set them to 0
                partial_corr_matrix[i, j] = 0

    return partial_corr_matrix