import numpy as np
from helpers import *


def softmax(s):
    """Compute softmax values for each sets of scores in s"""
    exps = np.exp(s)  # (K, n_batch)
    ones = np.ones((1, s.shape[0]))  # (1, K)
    denom = np.matmul(ones, np.exp(s))  # (1, n_batch)
    p = exps / denom  # (K, n_batch)
    return p


def relu(s):
    return np.maximum(0, s)


def BatchNorm(s, mu=None, var=None):
    n = s.shape[1]
    if mu is None:
        mu = np.mean(s, axis=1, keepdims=True)
    if var is None:
        var = np.var(s, axis=1, keepdims=False)  # False to get list for diag
    diag = np.diag(1 / (np.sqrt(var + 1e-6)))
    s_hat = np.matmul(diag, s - mu)
    return s_hat, mu, var


def BatchNormBackPass(G_batch, S, mu, var):
    n=S.shape[1]
    sigma_1 = 1 / np.sqrt(np.mean((S-mu)**2, axis=1, keepdims=True))
    sigma_2 = sigma_1 ** 3
    G_1=np.multiply(G_batch, np.repeat(sigma_1, n, axis=1))
    G_2=np.multiply(G_batch, np.repeat(sigma_2, n, axis=1))
    D=S - np.repeat(mu, n, axis = 1)
    c=np.sum(np.multiply(G_2, D), axis = 1, keepdims = True)
    G_batch=G_1
    temp=(1./n) * np.sum(G_1, axis = 1, keepdims = True)
    G_batch -= temp
    temp=(1./n) * np.multiply(D, np.repeat(c, n, axis=1))
    G_batch -= temp
    return G_batch


def evaluate_classifier(X, Ws, bs, gammas, betas, use_avg = False):
    """
    Ws = [(m, d), (K, m)]
    bs = [(m, 1), (K, 1)]
    X = (3072, n)
    p = (10, n)
    """
    n=X.shape[1]

    Hs=[np.copy(X)]
    Ss=[]
    S_hats=[]
    mus=[]
    _vars=[]
    for layer in range(n_layers - 1):
        WX=np.matmul(Ws[layer], Hs[layer])  # (m, n)
        # repeat column vector b, n times
        b_big1=np.repeat(bs[layer], n, axis = 1)
        s=WX + b_big1  # (m, n)
        if batch_norm:
            Ss.append(s)
            if use_avg:
                s_hat, mu, var=BatchNorm(
                    s, mu = mus_avg[layer], var = _vars_avg[layer])
            else:
                s_hat, mu, var=BatchNorm(s)
            S_hats.append(s_hat)
            mus.append(mu)
            _vars.append(var)
            s_tilde=np.multiply(gammas[layer], s_hat) + betas[layer]
            s=s_tilde
        h=relu(s)  # (m, n)
        Hs.append(h)

    # Last Layer
    WX=np.matmul(Ws[-1], Hs[-1])  # (K, n)
    b_big2=np.repeat(bs[-1], n, axis = 1)
    s=WX + b_big2
    p=softmax(s)
    return p, Hs, Ss, S_hats, mus, _vars


def compute_cost(X, Y, Ws, bs, gammas, betas, use_avg):
    """
    Ws = [(m, d), (K, m)]
    bs = [(m, 1), (K, 1)]
    X = (3072, n)
    p = (10, n)
    """
    n=X.shape[1]

    p, _, _, _, _, _=evaluate_classifier(
        X, Ws, bs, gammas, betas, use_avg=use_avg)
    # cross entropy for one x and one one hot y column vector
    # is -log(y^T * p) which is basically value of p[true_label]
    # Therefore we need to use the diagonal of np.matmul(Y.trasnpose(), p)
    # py = np.diag(py).reshape(1, n) # (1, n)
    # Or use the elementwise mult and then sum
    py = np.multiply(Y, p)  # (10, n)
    py = np.sum(py, axis=0).reshape(1, n)  # (1, n)
    cross_entropy = -np.log(py)  # (1, n)

    regulation = 0
    for layer in range(n_layers):
        regulation += _lambda * np.sum(Ws[layer]**2)
    J = (1/n) * np.sum(cross_entropy) + regulation  # scalar
    return J


def predict(p):
    return np.argmax(p, axis=0).reshape(1, p.shape[1])  # (1, n)


def compute_accuracy(X, y, Ws, bs, gammas, betas):
    p, _, _, _, _, _ = evaluate_classifier(
        X, Ws, bs, gammas, betas, use_avg=True)
    predicted = predict(p)

    zero_one_loss = 0
    for i in range(X.shape[1]):
        if predicted[0, i] == y[i]:
            zero_one_loss += 1

    accuracy = zero_one_loss / X.shape[1]
    return accuracy


def compute_gradients(X, Y, Ws, bs, gammas, betas):
    """
    X = (3072, n)
    Y = (10, n)
    P = (10, n)
    """
    n = X.shape[1]

    """ FORWARD PASS """
    P, Hs, Ss, S_hats, mus, _vars = evaluate_classifier(
        X, Ws, bs, gammas, betas)

    global mus_avg
    global _vars_avg
    if len(mus_avg) == 0:
        # init
        mus_avg = [mu for mu in mus]
        _vars_avg = [mu for mu in _vars]
    else:
        # update
        for layer in range(len(mus)):
            mus_avg[layer] = alpha * mus_avg[layer] + (1-alpha) * mus[layer]
            _vars_avg[layer] = alpha * \
                _vars_avg[layer] + (1-alpha) * _vars[layer]

    """ BACKWARDS PASS """
    # From lec4 slides 34
    grad_Ws = []
    grad_bs = []
    grad_gammas = []
    grad_betas = []

    G_batch = -(Y - P)  # Softmax backprop(K, n)

    # Last layer no batch norm
    grad_W = (1/n) * np.matmul(G_batch, Hs[-1].T)
    grad_W += 2 * _lambda * Ws[-1]  # Regulation term
    grad_b = (1/n) * np.sum(G_batch, axis=1, keepdims=True)

    grad_Ws.append(grad_W)
    grad_bs.append(grad_b)

    G_batch = np.matmul(Ws[-1].T, G_batch)  # (m, n)
    binary = np.zeros_like(Hs[-1])
    binary[Hs[-1] > 0] = 1
    G_batch = np.multiply(G_batch, binary)

    for layer in range(n_layers-2, -1, -1):
        if batch_norm:
            grad_gamma = (1/n) * np.sum(np.multiply(G_batch,
                                                    S_hats[layer]), axis=1, keepdims=True)
            grad_beta = (1/n) * np.sum(G_batch, axis=1, keepdims=True)
            grad_gammas.append(grad_gamma)
            grad_betas.append(grad_beta)
            
            # propagate throught scale and shift
            G_batch = np.multiply(G_batch, np.repeat(gammas[layer], n, axis=1))
            # propagate throught batch norm
            G_batch = BatchNormBackPass(
                G_batch, Ss[layer], mus[layer], _vars[layer])

        grad_W = (1/n) * np.matmul(G_batch, Hs[layer].T)
        grad_W += 2 * _lambda * Ws[layer]  # Regulation term
        grad_b = (1/n) * np.sum(G_batch, axis=1, keepdims=True)
        grad_Ws.append(grad_W)
        grad_bs.append(grad_b)

        # Porpagate through relu
        if layer == 0:
            # no need to propagate further
            break
        G_batch = np.matmul(Ws[layer].T, G_batch)  # (m, n)
        binary = np.zeros_like(Hs[layer])
        binary[Hs[layer] > 0] = 1
        G_batch = np.multiply(G_batch, binary)

    grad_Ws.reverse()
    grad_bs.reverse()
    grad_gammas.reverse()
    grad_betas.reverse()

    return grad_Ws, grad_bs, grad_gammas, grad_betas


def compare_gradients(X, Y, Ws, bs, gammas, betas):
    h = 1e-5
    n_batch = 10  # has to be > 1 for batch norm > 2 for it to work
    X, Y = X[:d, :n_batch], Y[:, :n_batch]

    grad_Ws, grad_bs, grad_gammas, grad_betas = compute_gradients(
        X, Y, Ws, bs, gammas, betas)

    num_grad_Ws, num_grad_bs, num_grad_gammas, num_grad_betas = compute_grads_num(
        X, Y, Ws, bs, gammas, betas, h, compute_cost, batch_norm)

    for layer in range(n_layers):
        print("\n\tLayer %d\\\\" % layer)
        comp_W = relative_error(
            grad_Ws[layer], num_grad_Ws[layer])
        print("gradient W%d relative error: %s\\\\\n" %
              (layer, comp_W.astype(str)))

        comp_b = relative_error(
            grad_bs[layer], num_grad_bs[layer])
        print("gradient b%d relative error: %s\\\\\n" %
              (layer, comp_b.astype(str)))

        if batch_norm and layer != n_layers - 1:
            comp_gamma = relative_error(
                grad_gammas[layer], num_grad_gammas[layer])
            print("gradient gamma%d relative error: %s\\\\\n" %
                  (layer, comp_gamma.astype(str)))

            comp_beta = relative_error(
                grad_betas[layer], num_grad_betas[layer])
            print("gradient beta%d relative error: %s\\\\\n" %
                  (layer, comp_beta.astype(str)))


def init_weights(size_in, size_out):
    xavier = 1/np.sqrt(size_in)
    he = np.sqrt(2 / size_in)
    xavier = he
    W = np.random.normal(0, xavier, size=(size_out, size_in))
    b = np.random.normal(0, xavier, size=(size_out, 1))
    if batch_norm:
        gamma = np.random.normal(0, xavier, size=(size_out, 1))
        beta = np.random.normal(0, xavier, size=(size_out, 1))
        return W, b, gamma, beta
    return W, b, None, None


def init_layers(layers):
    Ws = []
    bs = []
    gammas = []
    betas = []
    W, b, gamma, beta = init_weights(size_in=d, size_out=layers[0])
    Ws.append(W)
    bs.append(b)
    if batch_norm:
        gammas.append(gamma)
        betas.append(beta)
    for layer in range(1, n_layers):
        W, b, gamma, beta = init_weights(
            size_in=Ws[layer-1].shape[0], size_out=layers[layer])
        Ws.append(W)
        bs.append(b)
        if batch_norm and layer != n_layers - 1:  # no batch norm in last layer
            gammas.append(gamma)
            betas.append(beta)
    return Ws, bs, gammas, betas


def train_model(X, Y, y, Ws, bs, gammas, betas, X_valid, Y_valid, y_valid, X_test, y_test):
    costs_train = []
    costs_valid = []
    grads = []
    best_valid_acc = 0
    best_test_acc = 0
    eta = eta_min

    t = 0
    l = -1  # number of cycles completed
    for epoch_i in range(n_epochs):
        shuffle(X, Y)
        for X_batch, Y_batch in get_batches(n_batch, X, Y):
            grad_Ws, grad_bs, grad_gammas, grad_betas = compute_gradients(
                X_batch, Y_batch, Ws, bs, gammas, betas)

            for layer in range(n_layers):
                Ws[layer] = Ws[layer] - (eta * grad_Ws[layer])
                bs[layer] = bs[layer] - (eta * grad_bs[layer])
                if batch_norm:
                    if layer == n_layers - 1:
                        break  # no batch norm for last layer
                    gammas[layer] = gammas[layer] - (eta * grad_gammas[layer])
                    betas[layer] = betas[layer] - (eta * grad_betas[layer])

            if t % (2 * n_s) == 0:
                # this is where eta = eta_min
                l += 1
                acc_valid = compute_accuracy(
                    X_valid, y_valid, Ws, bs, gammas, betas)
                if acc_valid > best_valid_acc:
                    best_valid_acc = acc_valid
                acc_test = compute_accuracy(
                    X_test, y_test, Ws, bs, gammas, betas)
                if acc_test > best_test_acc:
                    best_test_acc = acc_test
            lower = 2 * n_s * l
            middle = (2 * l + 1) * n_s
            upper = 2 * (l + 1) * n_s
            if lower <= t and t <= middle:
                eta = eta_min + (t - 2 * l * n_s) / n_s * (eta_max - eta_min)
            elif middle <= t and t <= upper:
                eta = eta_max - (t - (2 * l + 1) * n_s) / \
                    n_s * (eta_max - eta_min)

            if save:
                if t % n_saves == 0:
                    costs_train.append(compute_cost(
                        X, Y, Ws, bs, gammas, betas, use_avg=True))
                    costs_valid.append(compute_cost(
                        X_valid, Y_valid, Ws, bs, gammas, betas, use_avg=True))
            if l == n_cycles:
                # Return after n_cycles
                return costs_train, costs_valid, grads, best_valid_acc, best_test_acc
            t += 1
        print()
        print(".... Epoch %d completed ...." % (epoch_i))
        print()

    return costs_train, costs_valid, grads, best_valid_acc, best_test_acc


def lambda_grid_search():
    global mus_avg
    global _vars_avg
    global save
    l_min = -5
    l_max = -1
    n_lambdas = 10
    lambdas = []
    for i in range(n_lambdas):
        l = l_min + (l_max - l_min) * np.random.rand()
        lambdas.append(10 ** l)
    print(lambdas)

    f = open('grid_sreach.txt', 'w+')
    count = 0
    for l in lambdas:
        print()
        print("\t--- STARTING NEW---\t%d" % count)
        print()
        count += 1

        Ws, bs, gammas, betas = init_layers(layers)
        mus_avg = []
        _vars_avg = []

        save = False
        ret = train_model(X, Y, y, Ws, bs, gammas, betas, X_valid,
                      Y_valid, y_valid, X_test, y_test)
        _, _, _, best_valid_acc, best_valid_acc = ret
        f.write('Lambda: %f    best accuracy: %f\n\n' % (l, best_acc))

    f.close()


if __name__ == '__main__':
    # X, Y, y = load_batch('data_batch_1')
    # X_valid, Y_valid, y_valid = load_batch('data_batch_2')
    # X_test, Y_test, y_test = load_batch('test_batch')
    X, Y, y, X_valid, Y_valid, y_valid, X_test, Y_test, y_test = load_batch_big(
        5000)
    # visulize_25(X)

    K = 10
    n_tot = X.shape[1]
    d = 3072

    _lambda = 0.005
    n_batch = 100
    n_epochs = 200

    eta_min = 1e-5
    eta_max = 1e-1

    # stepsize rule of thumb: n_s = k * (n_tot/n_batch) for 2 < k < 8
    n_s = 5 * np.floor(n_tot / n_batch)
    n_cycles = 2
    n_saves = round(n_s / 4)

    batch_norm = True
    alpha = 0.7

    save = True

    # out of layers, gotta have K at the end
    # layers = [50, 30, 20, 20, 10, 10, 10, K]
    layers = [50, 50, K]
    n_layers = len(layers)

    # lambda_grid_search()
    # exit()

    Ws, bs, gammas, betas = init_layers(layers)
    mus_avg = []
    _vars_avg = []

    # compare_gradients(X, Y, Ws, bs, gammas, betas)
    # exit()

    ret = train_model(X, Y, y, Ws, bs, gammas, betas, X_valid,
                      Y_valid, y_valid, X_test, y_test)
    costs_train, costs_valid, grads, best_valid_acc, best_test_acc = ret

    test_acc = compute_accuracy(X_test, y_test, Ws, bs, gammas, betas)
    print("Test accuracy: $%f$ \\\\" % (test_acc))
    print("Best test accuracy: $%f$ \\\\" % (best_test_acc))
    print("Best valid accuracy: $%f$ \\\\" % (best_valid_acc))


    if save:
        # gradient spread
        # fig, axes = plt.subplots(n_layers, 1, sharex=True, sharey=True)
        # for layer in range(n_layers):
        #     data = [np.exp(g[layer].reshape(
        #         g[layer].shape[0] * g[layer].shape[1])) for g in grads]
        #     axes[layer].boxplot(data, 0, '', showfliers=False)
        #     axes[layer].set_title("Distribution of layer %d" % (layer + 1))
        # plt.xlabel('update step')
        # plt.ylabel('exp grad')
        # plt.show()
        # training and validation cost
        plt.plot(np.arange(len(costs_train))*n_saves,
                 costs_train, 'g', label='training loss')
        plt.plot(np.arange(len(costs_valid))*n_saves,
                 costs_valid, 'r', label='validation loss')
        plt.xlabel("update step")
        plt.ylabel("cost")
        plt.legend()
        plt.show()
        # visulize_weights(Ws[0], ('Weights in layer %d' % 1))
