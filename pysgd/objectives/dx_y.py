import numpy as np

# Define gradient and cost functions
def grad_fun(theta, args):
    (D,y) = args
    m, n = D.shape

    grad = [0] * len(theta)
    x = theta[:n]
    lam = theta[n:]

    for i in range(n):
        grad[i] = 2 * x[i] - np.dot(D[:, i], lam)
    for i in range(m):
        grad[i + n] = y[i] - np.dot(D[i], x)
    return np.array(grad)

def cost_fun(theta, args):
    (D, y) = args
    m,n = D.shape

    x = theta[:n]
    lam = theta[n:]

    cost = np.sum(x ** 2)
    for i in range(m):
        # print(111)
        bb = np.dot(D[i], x) - y[i]
        cost += bb * lam[i]
    return np.append(theta, cost)
