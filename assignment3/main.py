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


def evaluate_classifier(X, Ws, bs):
    """
    Ws = [(m, d), (K, m)]
    bs = [(m, 1), (K, 1)]
    X = (3072, n)
    p = (10, n)
    """
    n = X.shape[1]
    n_layers = len(Ws)

    Hs = [np.copy(X)]
    for layer in range(n_layers - 1):
        WX = np.matmul(Ws[layer], Hs[layer])  # (m, n)
        # repeat column vector b, n times
        b_big1 = np.repeat(bs[layer], n, axis=1)
        s = WX + b_big1  # (m, n)
        h = relu(s)  # (m, n)
        Hs.append(h)

    # Last Layer
    WX = np.matmul(Ws[-1], Hs[-1])  # (K, n)
    b_big2 = np.repeat(bs[-1], n, axis=1)  # repeat column vector b, n times
    s = WX + b_big2
    p = softmax(s)
    return p, Hs


def compute_cost(X, Y, Ws, bs, _lambda):
    """
    Ws = [(m, d), (K, m)]
    bs = [(m, 1), (K, 1)]
    X = (3072, n)
    p = (10, n)
    """
    n = X.shape[1]
    n_layers = len(Ws)

    p, _ = evaluate_classifier(X, Ws, bs)
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


def compute_accuracy(X, y, Ws, bs):
    p, _ = evaluate_classifier(X, Ws, bs)
    predicted = predict(p)

    zero_one_loss = 0
    for i in range(X.shape[1]):
        if predicted[0, i] == y[i]:
            zero_one_loss += 1

    accuracy = zero_one_loss / X.shape[1]
    return accuracy


def compute_gradients(X, Y, Ws, bs, _lambda):
    """
    X = (3072, n)
    Y = (10, n)
    P = (10, n)
    """
    n = X.shape[1]
    n_layers = len(Ws)

    """ FORWARD PASS """
    P, Hs = evaluate_classifier(X, Ws, bs)

    """ BACKWARDS PASS """
    # From lec4 slides 34
    grad_Ws = []
    grad_bs = []

    G_batch = -(Y - P)  # (K, n)

    for layer in range(n_layers-1, 0, -1):
        grad_W = (1/n) * np.matmul(G_batch, Hs[layer].T)  # (K, m)
        grad_W += 2 * _lambda * Ws[layer]  # Regulation term
        grad_b = (1/n) * np.sum(G_batch, axis=1, keepdims=True)  # (K, 1)

        grad_Ws.append(grad_W)
        grad_bs.append(grad_b)

        G_batch = np.matmul(Ws[layer].T, G_batch)  # (m, n)
        binary = np.zeros_like(Hs[layer])
        binary[Hs[layer] > 0] = 1
        G_batch = np.multiply(G_batch, binary)

    # First layer
    grad_W = (1/n) * np.matmul(G_batch, X.T)  # (m, d)
    grad_W += 2 * _lambda * Ws[0]  # Regulation term
    grad_b = (1/n) * np.sum(G_batch, axis=1, keepdims=True)  # (m, 1)

    grad_Ws.append(grad_W)
    grad_bs.append(grad_b)
    grad_Ws.reverse()
    grad_bs.reverse()

    return grad_Ws, grad_bs


def compare_gradients(X, Y, Ws, bs, _lambda):
    h = 1e-5
    n_batch = 1
    n_layers = len(Ws)
    X, Y = X[:, :n_batch], Y[:, :n_batch]

    grad_Ws, grad_bs = compute_gradients(X, Y, Ws, bs, _lambda)

    num_grad_Ws, num_grad_bs = compute_grads_num(
        X, Y, Ws, bs, _lambda, h, compute_cost)

    for layer in range(n_layers):
        comp_W = relative_error(
            grad_Ws[layer], num_grad_Ws[layer])
        print("W%d relative error: " % layer)
        print(comp_W)
        comp_b = relative_error(
            grad_bs[layer], num_grad_bs[layer])
        print("b%d relative error: " % layer)
        print(comp_b)


def init_weights(size_in, size_out):
    xavier = 1/np.sqrt(size_in)
    W = np.random.normal(0, xavier, size=(size_out, size_in))
    b = np.random.normal(0, xavier, size=(size_out, 1))
    return W, b


def init_layers(layers):
    n_layers = len(layers)
    Ws = []
    bs = []
    W, b = init_weights(size_in=d, size_out=layers[0])
    Ws.append(W)
    bs.append(b)
    for layer in range(1, n_layers):
        W, b = init_weights(
            size_in=Ws[layer-1].shape[0], size_out=layers[layer])
        Ws.append(W)
        bs.append(b)
    return Ws, bs


def train_model(X, Y, y, Ws, bs, _lambda, n_batch, eta, n_epochs, X_valid, Y_valid, y_valid, X_test, y_test, n_cycles=-1, save=False):
    costs_train = []
    costs_valid = []
    accs_train = []
    accs_valid = []
    etas = []
    grads = []
    best_acc = 0
    n_layers = len(Ws)

    t = 0
    l = -1  # number of cycles completed
    for epoch_i in range(n_epochs):
        shuffle(X, Y)
        for X_batch, Y_batch in get_batches(n_batch, X, Y):
            grad_Ws, grad_bs = compute_gradients(
                X_batch, Y_batch, Ws, bs, _lambda)

            for layer in range(n_layers):
                Ws[layer] = Ws[layer] - (eta * grad_Ws[layer])
                bs[layer] = bs[layer] - (eta * grad_bs[layer])

            if t % (2 * n_s) == 0:
                # this is where eta = eta_min
                l += 1
                acc_valid = compute_accuracy(
                    X_valid, y_valid, Ws, bs)
                if acc_valid > best_acc:
                    best_acc = acc_valid
            lower = 2 * n_s * l
            middle = (2 * l + 1) * n_s
            upper = 2 * (l + 1) * n_s
            if lower <= t and t <= middle:
                eta = eta_min + (t - 2 * l * n_s) / n_s * (eta_max - eta_min)
            elif middle <= t and t <= upper:
                eta = eta_max - (t - (2 * l + 1) * n_s) / \
                    n_s * (eta_max - eta_min)

            if save:
                etas.append(eta)
                if t % n_saves == 0:
                    costs_train.append(compute_cost(X, Y, Ws, bs, _lambda))
                    costs_valid.append(compute_cost(
                        X_valid, Y_valid, Ws, bs, _lambda))
                    accs_train.append(compute_accuracy(X, y, Ws, bs))
                    accs_valid.append(compute_accuracy(
                        X_valid, y_valid, Ws, bs))
                    grads.append(grad_Ws)
            if l == n_cycles:
                # Return after n_cycles
                return costs_train, costs_valid, accs_train, accs_valid, etas, grads, best_acc
            t += 1
        print()
        print(".... Epoch %d completed ...." % (epoch_i))
        print()

    return costs_train, costs_valid, accs_train, accs_valid, etas, grads, best_acc


def lambda_grid_search():
    l_min = -6
    l_max = -4
    n_lambdas = 6
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

        W1, b1 = init_weights(size_in=d, size_out=50)  # (50, d)
        W2, b2 = init_weights(size_in=50, size_out=K)  # (K, 50)

        Ws = [W1, W2]
        bs = [b1, b2]

        save = False
        ret = train_model(X, Y, y, Ws, bs, l, n_batch, eta, n_epochs, X_valid,
                          Y_valid, y_valid, X_test, y_test, n_cycles=n_cycles, save=save)
        _, _, _, _, _, _, best_acc = ret
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

    _lambda = 0.005  # 4.1e-5
    n_batch = 100
    n_epochs = 200

    eta_min = 1e-5
    eta_max = 1e-1
    eta = eta_min
    # stepsize rule of thumb: n_s = k * (n_tot/n_batch) for 2 < k < 8
    n_s = 5 * np.floor(n_tot / n_batch)
    n_cycles = 2
    n_saves = round(n_s / 4)

    # lambda_grid_search()
    # exit()

    # out of layers, gotta have K at the end
    layers = [50, 30, 20, 20, 10, 10, 10, K]
    # layers = [50, 50, K]
    n_layers = len(layers)

    Ws, bs = init_layers(layers)

    # compare_gradients(X, Y, Ws, bs, _lambda)
    # exit()

    save = True
    ret = train_model(X, Y, y, Ws, bs, _lambda, n_batch, eta, n_epochs, X_valid,
                      Y_valid, y_valid, X_test, y_test, n_cycles=n_cycles, save=save)
    costs_train, costs_valid, accs_train, accs_valid, etas, grads, best_acc = ret

    test_acc = compute_accuracy(X_test, y_test, Ws, bs)
    print("Test accuracy: $%f$ \\\\" % (test_acc))
    print("Train accuracy: $%f$ \\\\" % (accs_train[-1]))
    print("Best valid accuracy: $%f$ \\\\" % (best_acc))

    if save:
        # Etas
        plt.plot(etas)
        plt.show()
        # gradient spread
        fig, axes = plt.subplots(n_layers, 1, sharex=True, sharey=True)
        for layer in range(n_layers):
            data = [np.exp(g[layer].reshape(
                g[layer].shape[0] * g[layer].shape[1])) for g in grads]
            axes[layer].boxplot(data, 0, '', showfliers=False)
            axes[layer].set_title("Distribution of layer %d" % (layer + 1))
        plt.xlabel('update step')
        plt.ylabel('exp grad')
        plt.show()
        # training and validation cost
        plt.plot(np.arange(len(costs_train))*n_saves,
                 costs_train, 'g', label='training loss')
        plt.plot(np.arange(len(costs_valid))*n_saves,
                 costs_valid, 'r', label='validation loss')
        plt.xlabel("update step")
        plt.ylabel("cost")
        plt.legend()
        plt.show()
        # training and validation accuracy
        plt.plot(np.arange(len(accs_train))*n_saves,
                 accs_train, 'g', label='training accuracy')
        plt.plot(np.arange(len(accs_valid))*n_saves, accs_valid,
                 'r', label='validation accuracy')
        plt.xlabel("update step")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()
        # visulize_weights(Ws[0], ('Weights in layer %d' % 1))
