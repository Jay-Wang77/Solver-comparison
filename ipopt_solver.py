import numpy as np
import cyipopt
from data_point import get_data

class FittingProblem:
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data

    def objective(self, coeffs):
        a, b, c = coeffs
        return np.sum((self.y - (a * self.x**2 + b * self.x + c))**2)

    def gradient(self, coeffs):
        a, b, c = coeffs
        grad = np.zeros(3)
        grad[0] = -2 * np.sum((self.y - (a * self.x**2 + b * self.x + c)) * self.x**2)
        grad[1] = -2 * np.sum((self.y - (a * self.x**2 + b * self.x + c)) * self.x)
        grad[2] = -2 * np.sum(self.y - (a * self.x**2 + b * self.x + c))
        return grad

    def constraints(self, coeffs):
        return np.array([])

    def jacobian(self, coeffs):
        return np.array([])

def solve_ipopt():
    x_data, y_data = get_data()
    x0 = np.array([1, 1, 1])  # 初始猜测
    problem = FittingProblem(x_data, y_data)
    nlp = ipopt.problem(
        n=len(x0),
        m=0,
        problem_obj=problem,
        lb=[-np.inf]*3,
        ub=[np.inf]*3,
        cl=np.array([]),
        cu=np.array([])
    )
    x, info = nlp.solve(x0)
    print("Fitted coefficients (a, b, c) using Ipopt:", x)

    return x

# if __name__ == "__main__":
    #solve_ipopt()
