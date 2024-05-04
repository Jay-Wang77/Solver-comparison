import matplotlib.pyplot as plt
from osqp_solver import solve_osqp
from ipopt_solver import solve_ipopt
from data_point import get_data
import numpy as np

x_data, y_data = get_data()
coeffs_osqp = solve_osqp()
coeffs_ipopt = solve_ipopt()

# 生成拟合的曲线数据
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit_osqp = coeffs_osqp[0] * x_fit**2 + coeffs_osqp[1] * x_fit + coeffs_osqp[2]
y_fit_ipopt = coeffs_ipopt[0] * x_fit**2 + coeffs_ipopt[1] * x_fit + coeffs_ipopt[2]

# 绘图
plt.figure(figsize=(10, 5))
plt.scatter(x_data, y_data, color='red', label='Data Points')
plt.plot(x_fit, y_fit_osqp, label='OSQP Fit', linestyle='--')
plt.plot(x_fit, y_fit_ipopt, label='Ipopt Fit', linestyle='-.')
plt.title('Curve Fitting Comparison')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
