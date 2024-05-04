import osqp
import numpy as np
from scipy import sparse
from data_point import get_data

def solve_osqp():
    x_data, y_data = get_data()

    n = len(x_data)
    P = sparse.eye(n)
    q = np.zeros(n)

    # 定义矩阵A (二次，线性，常数项)
    A = sparse.vstack([
        sparse.diags(x_data**2),
        sparse.diags(x_data),
        sparse.eye(n)
    ]).T

    # 定义l（目标值）和u（目标值）
    l = y_data
    u = y_data

    # 创建OSQP对象
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, verbose=False)

    # 求解问题
    results = prob.solve()

    # 输出结果
    print("Fitted coefficients (a, b, c) using OSQP:", results.x)

    return results.x
#if __name__ == "__main__":
 #   solve_osqp()
