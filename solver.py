from scipy.optimize import minimize, LinearConstraint
import numpy as np
import io
import pickle
import time
import matplotlib.pyplot as plt

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
    gamma = 0.001
    precision = 0.001
    t = 1000
    epsilon = 1e-8
    gamma_threshold = 1e-30
    gamma_step_counter = 0

    fun = lambda x: t * np.linalg.norm(H @ x - y) ** 2 - np.sum(np.log(x + epsilon))

    for i in range(1000):
        prev_x = cur_x
        grad = 2 * t * (H.T @ ((H @ prev_x) - y)) - (1 / (prev_x + epsilon))

        # cur_x-= gamma * grad
        # cur_x /= np.sum(cur_x)
        if gamma_step_counter == 30:
            gamma = gamma * 1000

        #  check if we get out of the barrier:
        sanity_x = cur_x - gamma * grad
        step_update_required = np.any(sanity_x < 0)
        while step_update_required and gamma > gamma_threshold:
            gamma *= 0.1
            gamma_step_counter = 0
            sanity_x = cur_x - gamma * grad
            step_update_required = np.any(sanity_x < 0)

        if step_update_required:
            sanity_x = np.maximum(0, sanity_x)

        cur_x = sanity_x
        cur_x /= np.sum(cur_x)

        gamma_step_counter += 1
        if (precision >= fun(cur_x)):
            break

    return cur_x


def NW_solver_v2(H: np.ndarray, y: np.ndarray):
    if H.shape[0] != y.shape[0]:
        return "error"

    results = [1000000000000]
    gamma = 0.001
    precision = 0.001
    t = 10000
    epsilon = 1e-8
    gamma_threshold = 1e-30
    gamma_step_counter = 0

    fun = lambda x: t * np.linalg.norm(H @ x - y) ** 2 - np.sum(np.log(x + epsilon))

    # use the gradient descent as the initial 
    cur_x = GD_solver(H, y)
    norm_hessian = 2 * t * np.dot(H.T, H)

    for i in range(6):
        prev_x = cur_x
        grad = 2 * t * (H.T @ ((H @ prev_x) - y)) - (1 / (prev_x + epsilon))
        grad_diag_v = np.diag((- 1 / (prev_x + epsilon)**2))
        hessian = norm_hessian + grad_diag_v

        #  check if we get out of the barrier:
        sanity_x = cur_x - np.linalg.solve(hessian, grad)
        step_update_required = np.any(sanity_x < 0)
        while step_update_required and gamma > gamma_threshold:
            gamma *= 0.1
            gamma_step_counter = 0
            sanity_x = cur_x - gamma * grad
            step_update_required = np.any(sanity_x < 0)

        # use relu as default
        if step_update_required:
            sanity_x = np.maximum(0, sanity_x)

        # update x
        cur_x = sanity_x
        cur_x /= np.sum(cur_x)

        gamma_step_counter += 1
        results.append(fun(cur_x))

        if (precision >= results[i + 1] or abs((results[i + 1] - results[i])) < 0.5):
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

    # use the gradient descent as the initial x
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


class Example:
    def __init__(self, H: np.ndarray, y: np.ndarray, x: np.ndarray):
        self.H = H
        self.y = y


##### start of main script
file = open('./examples/examples.pkl', 'rb')
examples = pickle.load(file)


for i, example in enumerate(examples, 0):
    H = example.H
    y = example.y

    print(f'--- example {i} ---')
    print(H.shape)
    print(y.shape)
    gd_x = GD_solver(H, y)
    nw_x = NW_solver_v2(H, y)
    try:
        x_approx = solve(H, y)
        print('solver score:', np.linalg.norm(y - H @ x_approx))
        print('solver validation:', np.isclose(np.sum(x_approx), 1.))
    except:
        pass

    print('GD score:', np.linalg.norm(y - H @ gd_x))
    print('GD validation:', np.isclose(np.sum(gd_x), 1.) and not np.any(gd_x < 0))


    print('NW score:', np.linalg.norm(y - H @ nw_x))
    print('NW validation:', np.isclose(np.sum(nw_x), 1.) and not np.any(nw_x < 0))

    print('\n\n')
