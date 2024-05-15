import numpy as np
import time
import scipy.sparse.linalg as sla
import scipy.linalg as la
from tqdm import tqdm

def objective(S, X, Z, lambda_val):
    # return np.trace(S @ X) - np.linalg.slogdet(X)[1] + lambda_val * np.linalg.norm(Z, 1)
    return np.trace(S @ X) - np.log(np.linalg.slogdet(X))+ lambda_val * np.linalg.norm(Z, 1)


def shrinkage(a, kappa):
    return np.maximum(0, a - kappa) - np.maximum(0, -a - kappa)



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
    MAX_ITER = 5000
    ABSTOL = 1e-6
    RELTOL = 1e-4

    # 计算经验协方差矩阵
    S = np.cov(D, rowvar=False)
    n = S.shape[0]

    # 初始化变量
    X = np.zeros((n, n))
    Z = np.zeros((n, n))
    U = np.zeros((n, n))

    history = {'objval': [], 'r_norm': [], 's_norm': [], 'eps_pri': [], 'eps_dual': []}

    for k in tqdm(range(MAX_ITER), desc='ADMM Progress'):
        # print(S)
        # Q, L = la.eig(rho * (Z - U) - S)
        # Q, L = np.linalg.eig(rho * (Z - U) - S)   # eig/ eigh ??????????
        L, Q = np.linalg.eig(rho * (Z - U) - S)   # values, vectors
        #L = np.diag(L)
        #es = np.diag(L)
        es = L
        xi = (es + np.sqrt(es**2 + 4*rho)) / (2*rho) # 逐元素操作

        X = Q @ np.diag(xi) @ Q.T

        # z-update with relaxation
        Zold = Z.copy()
        X_hat = alpha * X + (1 - alpha) * Zold
        Z = shrinkage(X_hat + U, lambda_val / rho)

        # U 更新
        U += X_hat - Z

    #     # diagnostics, reporting, termination checks
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

#
