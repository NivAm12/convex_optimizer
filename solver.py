from scipy.optimize import minimize, LinearConstraint
import numpy as np
import io
import pickle
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime

w = 99


def solve(H: np.ndarray, y: np.ndarray):
    fun = lambda x: np.linalg.norm(H @ x - y)
    jac = lambda x: H.T @ (H @ x - y)

    constraint1 = LinearConstraint(np.ones(w).reshape((1, w)), 1, 1)
    constraint2 = LinearConstraint(np.eye(w), 0, np.infty)

    x0 = np.random.rand(w)
    x0 /= np.sum(x0)

    return minimize(fun, x0, jac=jac, constraints=(constraint1, constraint2)).x


def GD_solver(H: np.ndarray, y: np.ndarray):
    if H.shape[0] != y.shape[0]:
        return "error"

    cur_x = np.ones(H.shape[1]) / H.shape[1]
    precision = 0.001
    t = 1000
    epsilon = 1e-8
    min_gamma_threshold = 1e-30
    max_gamma_threshold = 1e-3
    gamma = max_gamma_threshold
    gamma_step_counter = 0

    def fun(x):
        return t * np.linalg.norm(H @ x - y) ** 2 - np.sum(np.log(x + epsilon))

    def step_update_required(x):
        return np.any(x < 0)

    for i in range(1000):
        prev_x = cur_x
        grad = 2 * t * (H.T @ ((H @ prev_x) - y)) - (1 / (prev_x + epsilon))

        # cur_x-= gamma * grad
        # cur_x /= np.sum(cur_x)
        if gamma_step_counter == 30:
            gamma = np.minimum(max_gamma_threshold, gamma * 1000)
            gamma_step_counter = 0

        #  check if we get out of the barrier:
        sanity_x = cur_x - gamma * grad
        while step_update_required(sanity_x) and gamma > min_gamma_threshold:
            gamma *= 0.1
            gamma_step_counter = 0
            sanity_x = cur_x - gamma * grad

        if step_update_required(sanity_x):
            sanity_x = np.maximum(0, sanity_x)

        cur_x = sanity_x
        cur_x /= np.sum(cur_x)

        gamma_step_counter += 1
        if precision >= fun(cur_x):
            break

    return cur_x


def NW_solver(H: np.ndarray, y: np.ndarray):
    if H.shape[0] != y.shape[0]:
        return "error"

    results = [1000000000000]

    precision = 0.01
    t = 100
    fun = lambda x: t * np.linalg.norm(H @ x - y) ** 2 - np.sum(np.log(x))
    barrier_grad_func = lambda xi: 1 / xi

    # use the gradient descent as the initial 
    cur_x = GD_solver(H, y)
    hessian = 2 * np.dot(H.T, H)

    for i in range(6):
        prev_x = cur_x
        grad = t * (H.T @ ((H @ prev_x) - y)) - (np.sum(np.vectorize(barrier_grad_func)(prev_x)))
        hess = t * hessian + (np.sum(np.vectorize(barrier_grad_func)(prev_x)))
        cur_x -= np.linalg.solve(hess, grad)

        # constrains
        # cur_x = np.maximum(cur_x, 0)
        cur_x /= np.sum(cur_x)
        results.append(fun(cur_x))

        if (precision >= results[i + 1] or abs((results[i + 1] - results[i])) < 0.5):
            break

    # plt.plot(range(len(results)), results)
    # plt.show()
    return cur_x


def NW_solver_v2(H: np.ndarray, y: np.ndarray):
    if H.shape[0] != y.shape[0]:
        return "error"

    results = [1000000000000]
    epsilon = 1 / 1000000000
    precision = 0.01
    t = 1000

    fun = lambda x: t * np.linalg.norm(H @ x - y) ** 2 - np.sum(np.log(x + epsilon))

    # use the gradient descent as the initial 
    cur_x = GD_solver(H, y)
    hessian = 2 * np.dot(H.T, H)

    for i in range(6):
        prev_x = cur_x
        grad = 2 * t * (H.T @ ((H @ prev_x) - y)) - (1 / (prev_x + epsilon))
        cur_x -= np.linalg.solve(hessian, grad)

        # constrains
        cur_x /= np.sum(cur_x)
        results.append(fun(cur_x))

        if (precision >= results[i + 1] or abs((results[i + 1] - results[i])) < 0.5):
            break

    return cur_x


def dual_solver(H: np.ndarray, y: np.ndarray):
    cur_mu1 = 1
    cur_mu2 = np.zeros(y.shape[0])
    min_gamma_threshold = 1e-30
    max_gamma_threshold = 1e-3
    gamma = max_gamma_threshold
    gamma_step_counter = 0
    mu2_norms = []
    grad_mu2_norms = []
    mu1_values = []
    fun_values = []

    def fun(mu1, mu2):
        return -1/4 * np.sum(mu2**2) - mu1 - mu2 @ y

    def step_update_required(mu1, mu2):
        res = mu1 * np.ones(H.shape[1]) + H.T @ mu2
        return np.any(res < 0)

    for i in range(1000):
        prev_mu1 = cur_mu1
        prev_mu2 = cur_mu2
        grad_mu1 = -1
        grad_mu2 = -1/2 * prev_mu2 - y

        if gamma_step_counter == 30:
            gamma = np.minimum(max_gamma_threshold, gamma * 1000)
            gamma_step_counter = 0

        #  check if we get out of the barrier:
        sanity_mu1 = cur_mu1 + gamma * grad_mu1
        sanity_mu2 = cur_mu2 + gamma * grad_mu2

        while step_update_required(sanity_mu1, sanity_mu2) and gamma > min_gamma_threshold:
            gamma *= 0.1
            gamma_step_counter = 0
            sanity_mu1 = cur_mu1 + gamma * grad_mu1
            sanity_mu2 = cur_mu2 + gamma * grad_mu2

        if step_update_required(sanity_mu1, sanity_mu2):
            sanity_mu1 = np.maximum(np.max(- H.T @ sanity_mu2), sanity_mu1)

        cur_mu1 = sanity_mu1
        cur_mu2 = sanity_mu2

        mu2_norms.append(np.sum(cur_mu2**2))
        grad_mu2_norms.append(np.sum(grad_mu2**2))
        mu1_values.append(cur_mu1)
        fun_values.append(fun(cur_mu1, cur_mu2))

        if i > 0 and \
                np.abs(mu2_norms[i] - mu2_norms[i-1]) < 1e-10 and \
                np.abs(fun_values[i] - fun_values[i-1]) < 1e-10 and \
                np.abs(mu1_values[i] - mu1_values[i-1]) < 1e-10 and \
                np.abs(grad_mu2_norms[i] - grad_mu2_norms[i-1]) < 1e-10:
            print(f'Stopped loop at {i} iteration')
            break
        # plot norms
        gamma_step_counter += 1

    # plot_norms(range(1000), m2_norms, grad_m2_norms, m1_values, fun_values)

    z = y - 1/2 * cur_mu2
    x = np.linalg.solve(H.T @ H, H.T @ z)

    if np.any(x < 0):
        x = np.maximum(0, x)
    if not np.isclose(np.sum(x), 1.):
        x /= np.sum(x)

    return x


def dual_solver_with_barrier(H: np.ndarray, y: np.ndarray):
    if H.shape[0] != y.shape[0]:
        return "error"

    cur_mu1 = np.ones(H.shape[1])
    cur_mu2 = np.zeros(y.shape[0])
    min_gamma_threshold = 1e-30
    max_gamma_threshold = 1e-3
    gamma = max_gamma_threshold
    gamma_step_counter = 0
    t = 100
    epsilon = 1e-8
    m2_norms = []
    grad_m2_norms = []
    m1_norms = []
    fun_values = []

    def fun(mu1, mu2):
        return t * (3/4 * np.sum(mu2**2) - mu1 - mu2 @ y) - np.log(mu1 + H.T @ mu2 + epsilon)

    def step_update_required(mu1, mu2):
        res = mu1 + H.T @ mu2
        return np.any(res < 0)

    print(f'start of for loop {datetime.datetime.now()}')
    for i in tqdm(range(1000), desc="Dual for loop"):
        prev_mu2 = cur_mu2
        prev_mu1 = cur_mu1
        grad_mu1 = -1 * t - 1 / (prev_mu1 + H.T @ prev_mu2 + epsilon)
        grad_mu2 = t * (3/2 * prev_mu2 - y) - H @ (1 / (prev_mu1 + H.T @ prev_mu2 + epsilon))

        if gamma_step_counter == 30:
            gamma = np.minimum(max_gamma_threshold, gamma * 1000)
            gamma_step_counter = 0

        #  check if we get out of the barrier:
        sanity_mu1 = cur_mu1 + gamma * grad_mu1
        sanity_mu2 = cur_mu2 + gamma * grad_mu2

        while step_update_required(sanity_mu1, sanity_mu2) and gamma > min_gamma_threshold:
            gamma *= 0.1
            gamma_step_counter = 0
            sanity_mu1 = cur_mu1 + gamma * grad_mu1
            sanity_mu2 = cur_mu2 + gamma * grad_mu2

        if step_update_required(sanity_mu1, sanity_mu2):
            sanity_mu1 = np.maximum(np.max(- H.T @ sanity_mu2), sanity_mu1)

        cur_mu1 = sanity_mu1
        cur_mu2 = sanity_mu2

        m2_norms.append(np.sum(cur_mu2**2))
        grad_m2_norms.append(np.sum(grad_mu2**2))
        m1_norms.append(np.sum(cur_mu1**2))
        fun_values.append(fun(cur_mu1, cur_mu2))
        # plot norms
        gamma_step_counter += 1

    # plot_norms(range(1000), m2_norms, grad_m2_norms, m1_values, fun_values)

    print(f'end of for loop {datetime.datetime.now()}')
    z = y - 1/2 * cur_mu2
    x = np.linalg.solve(H.T @ H, H.T @ z)
    print(f'end of solve {datetime.datetime.now()}')

    if np.any(x < 0):
        x = np.maximum(0, x)
    if not np.isclose(np.sum(x), 1.):
        x /= np.sum(x)

    return x

def plot_norms(x_axis, m2_norms, grad_m2_norms, m1_values, fun_values):
    plt.title('m2_norms')
    plt.plot(x_axis, m2_norms, label='m2_norms')
    plt.show()
    plt.title('grad_m2_norms')
    plt.plot(x_axis, grad_m2_norms, label='grad_m2_norms')
    plt.show()
    plt.title('m1_values')
    plt.plot(x_axis, m1_values, label='m1_values')
    plt.show()
    plt.title('fun_values')
    plt.plot(x_axis, fun_values, label='fun_values')
    plt.show()

class Example:
    def __init__(self, H: np.ndarray, y: np.ndarray, x: np.ndarray):
        self.H = H
        self.y = y


# start of main script

file = open('./examples/examples.pkl', 'rb')
examples = pickle.load(file)

print(f'~~~~ Start of run {datetime.datetime.now()} ~~~~')
for i, example in enumerate(examples, 0):
    H = example.H
    y = example.y

    if H.shape[0] != y.shape[0]:
        print("Error : matrix dimension doesn't correlate")
        continue

    # skip big examples for debugging manners
    # if H.shape[0] > 1000:
    #     continue

    # Mor debugging!!!!
    # if i is not 8:
    #     continue

    print(f'--- example {i} ---')
    print(H.shape)
    print(y.shape)
    # gd_x = GD_solver(H, y)
    dual_x = dual_solver(H, y)
    # dual_barrier_x = dual_solver_with_barrier(H, y)
    # nw_x = NW_solver_v2(H, y)
    # try:
    #     x_approx = solve(H, y)
    #     print('solver score:', np.linalg.norm(y - H @ x_approx))
    #     print('solver validation:', np.isclose(np.sum(x_approx), 1.) and not np.any(x_approx < 0))
    # except:
    #     pass

    # print('GD score:', np.linalg.norm(y - H @ gd_x))
    # print('GD validation:', np.isclose(np.sum(gd_x), 1.) and not np.any(gd_x < 0))

    print('Dual score:', np.linalg.norm(y - H @ dual_x))
    print('Dual validation:', np.isclose(np.sum(dual_x), 1.) and not np.any(dual_x < 0))

    # print('Dual barrier score:', np.linalg.norm(y - H @ dual_barrier_x))
    # print('Dual barrier validation:', np.isclose(np.sum(dual_barrier_x), 1.) and not np.any(dual_barrier_x < 0))

    # print('NW score:', np.linalg.norm(y - H @ nw_x))
    # print('NW validation:', np.isclose(np.sum(nw_x), 1.) and not np.any(nw_x < 0))
    print('\n\n')

print(f'~~~~ End of run {datetime.datetime.now()} ~~~~')
