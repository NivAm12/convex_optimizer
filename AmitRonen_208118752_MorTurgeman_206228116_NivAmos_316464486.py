import numpy as np

def solver(H: np.ndarray, y: np.ndarray) -> np.ndarray:
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

    # dual problem
    fun = lambda mu1, mu2: -1/4 * np.sum(mu2**2) - mu1 - mu2 @ y

    def step_update_required(mu1, mu2):
        res = mu1 * np.ones(H.shape[1]) + H.T @ mu2
        return np.any(res < 0)

    for i in range(1000):
        # gradients
        prev_mu1 = cur_mu1
        prev_mu2 = cur_mu2
        grad_mu1 = -1
        grad_mu2 = -1/2 * prev_mu2 - y

        if gamma_step_counter == 30:
            gamma = np.minimum(max_gamma_threshold, gamma * 1000)
            gamma_step_counter = 0

        #  check if we get out of the feasible region:
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

        # check if more steps are required
        mu2_norms.append(np.sum(cur_mu2**2))
        grad_mu2_norms.append(np.sum(grad_mu2**2))
        mu1_values.append(cur_mu1)
        fun_values.append(fun(cur_mu1, cur_mu2))

        if i > 0 and \
                np.abs(mu2_norms[i] - mu2_norms[i-1]) < 1e-10 and \
                np.abs(fun_values[i] - fun_values[i-1]) < 1e-10 and \
                np.abs(mu1_values[i] - mu1_values[i-1]) < 1e-10 and \
                np.abs(grad_mu2_norms[i] - grad_mu2_norms[i-1]) < 1e-10:
            break

        gamma_step_counter += 1

    # solve the final equations
    z = y - 1/2 * cur_mu2
    x = np.linalg.solve(H.T @ H, H.T @ z)

    if np.any(x < 0):
        x = np.maximum(0, x)
    if not np.isclose(np.sum(x), 1.):
        x /= np.sum(x)

    return x

