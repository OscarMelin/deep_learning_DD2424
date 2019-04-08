import pickle
import matplotlib.pyplot as plt
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_batch(batch_name):
    """  
    X = (3072, 10000) each column is an image
    Y = (10, 10000) each colum is a one hot vector
    y = labels for each column
    """
    data_dict = unpickle('./datasets/cifar-10-batches-py/' + batch_name)
    X = data_dict[b'data']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).reshape(
        10000, 3072).transpose(1, 0)  # get (rgb) not rrrgggbbb

    # center
    # X = X / 255 # between 0 and 1
    # X_mean = np.mean(X, axis=1, keepdims=True)
    # X = X - X_mean # Center with mean 0

    # normalize
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True)

    X = X - X_mean
    X = X / X_std

    y = data_dict[b'labels']
    Y = make_one_hot(y)
    return X, Y, y


def load_batch_big(n_valid):
    n_valid = 10000 - n_valid
    X, Y, y = load_batch('data_batch_1')
    X_2, Y_2, y_2 = load_batch('data_batch_2')
    X_3, Y_3, y_3 = load_batch('data_batch_3')
    X_4, Y_4, y_4 = load_batch('data_batch_4')
    X_5, Y_5, y_5 = load_batch('data_batch_5')
    X_test, Y_test, y_test = load_batch('test_batch')
    X = np.append(X, X_2, axis=1)
    Y = np.append(Y, Y_2, axis=1)
    y = np.append(y, y_2, axis=0)
    X = np.append(X, X_3, axis=1)
    Y = np.append(Y, Y_3, axis=1)
    y = np.append(y, y_3, axis=0)
    X = np.append(X, X_4, axis=1)
    Y = np.append(Y, Y_4, axis=1)
    y = np.append(y, y_4, axis=0)

    X = np.append(X, X_5[:, :n_valid], axis=1)
    Y = np.append(Y, Y_5[:, :n_valid], axis=1)
    y = np.append(y, y_5[:n_valid], axis=0)
    X_valid, Y_valid, y_valid = X_5[:,
                                    n_valid:], Y_5[:, n_valid:], y_5[n_valid:]

    return X, Y, y, X_valid, Y_valid, y_valid, X_test, Y_test, y_test


def shuffle(X, Y):
    index = np.arange(X.shape[1])
    np.random.shuffle(index)
    X = X[:, index]
    Y = Y[:, index]
    return X, Y


def get_batches(n_batch, X, Y):
    """ Return n_batch of the X 
    vector at a time
    """
    current_index = 0
    while current_index + n_batch <= X.shape[1]:
        X_batch = X[:, current_index:current_index + n_batch]
        Y_batch = Y[:, current_index:current_index + n_batch]
        current_index += n_batch
        yield X_batch, Y_batch


def compute_grads_num_slow(X, Y, W, b, _lambda, h, compute_cost):
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros_like(W)
    grad_b = np.zeros((no, 1))

    for i in range(b.shape[0]):
        b_try = np.copy(b)
        b_try[i] = b_try[i] - h
        c1 = compute_cost(X, Y, W, b_try, _lambda)
        b_try = np.copy(b)
        b_try[i] = b_try[i] + h
        c2 = compute_cost(X, Y, W, b_try, _lambda)
        grad_b[i] = (c2-c1) / (2*h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.copy(W)
            W_try[i][j] = W_try[i][j] - h
            c1 = compute_cost(X, Y, W_try, b, _lambda)
            W_try = np.copy(W)
            W_try[i][j] = W_try[i][j] + h
            c2 = compute_cost(X, Y, W_try, b, _lambda)
            grad_W[i][j] = (c2-c1) / (2*h)

    return grad_W, grad_b


def compute_grads_num(X, Y, Ws, bs, _lambda, h, compute_cost):
    no = Ws[1].shape[0]
    d = X.shape[0]

    grad_Ws = []
    grad_bs = []

    c = compute_cost(X, Y, Ws, bs, _lambda)

    for layer in range(len(Ws)):
        grad_W = np.zeros_like(Ws[layer])
        grad_b = np.zeros_like(bs[layer])
        for i in range(bs[layer].shape[0]):
            b_try = np.copy(bs[layer])
            b_try[i] = b_try[i] + h
            temp = bs[layer]
            bs[layer] = b_try
            c2 = compute_cost(X, Y, Ws, bs, _lambda)
            bs[layer] = temp
            grad_b[i] = (c2-c) / h
        print("Done with b")

        for i in range(Ws[layer].shape[0]):
            for j in range(Ws[layer].shape[1]):
                W_try = np.copy(Ws[layer])
                W_try[i][j] = W_try[i][j] + h
                temp = Ws[layer]
                Ws[layer] = W_try
                c2 = compute_cost(X, Y, Ws, bs, _lambda)
                Ws[layer] = temp
                grad_W[i][j] = (c2-c) / h
        print("Done with W")
        grad_Ws.append(grad_W)
        grad_bs.append(grad_b)
        print("Done with one layer")

    return grad_Ws, grad_bs


def relative_error(grad, num_grad):
    nominator = np.sum(np.abs(grad - num_grad))
    demonimator = max(1e-6, np.sum(np.abs(grad)) + np.sum(np.abs(num_grad)))
    return nominator / demonimator


def make_one_hot(y):
    """ create one-hot column vectors """
    one_hot = np.zeros((len(y), 10))
    for i in range(len(y)):
        one_hot[i, y[i]] = 1.
    return one_hot.transpose(1, 0)


def visulize_25(X):
    """ Show 5x5 images from X
    """
    fig, axes1 = plt.subplots(5, 5, figsize=(3, 3))
    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(X)))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(X[:, i].reshape(32, 32, 3))
    plt.show()


def visulize_weights(W, title):
    """ Show all the weight vectors as pictures
    """
    fig, axes1 = plt.subplots(6, 8, figsize=(3, 3))
    i = 0
    for j in range(6):
        for k in range(8):
            im = W[i, :].reshape(32, 32, 3)
            im = (im - np.min(im)) / (np.max(im) - np.min(im))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(im)
            i += 1
    fig.suptitle(title)
    plt.show()
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


def leaky_relu(s):
    return np.maximum(0.02*s, s)


def evaluate_classifier(X, Ws, bs):
    """
    Ws = [(m, d), (K, m)]
    bs = [(m, 1), (K, 1)]
    X = (3072, n)
    p = (10, n)
    """
    n = X.shape[1]
    # Layer 1
    WX1 = np.matmul(Ws[0], X)  # (m, n)
    b_big1 = np.repeat(bs[0], n, axis=1)  # repeat column vector b, n times
    s = WX1 + b_big1  # (m, n)
    h = leaky_relu(s)  # (m, n)
    # Layer 2
    WX2 = np.matmul(Ws[1], h)  # (K, n)
    b_big2 = np.repeat(bs[1], n, axis=1)  # repeat column vector b, n times
    s = WX2 + b_big2
    p = softmax(s)
    return p, h


def compute_cost(X, Y, Ws, bs, _lambda):
    """
    Ws = [(m, d), (K, m)]
    bs = [(m, 1), (K, 1)]
    X = (3072, n)
    p = (10, n)
    """
    n = X.shape[1]
    p, _ = evaluate_classifier(X, Ws, bs)
    # cross entropy for one x and one one hot y column vector
    # is -log(y^T * p) which is basically value of p[true_label]
    # Therefore we need to use the diagonal of np.matmul(Y.trasnpose(), p)
    # py = np.diag(py).reshape(1, n) # (1, n)
    # Or use the elementwise mult and then sum
    py = np.multiply(Y, p)  # (10, n)
    py = np.sum(py, axis=0).reshape(1, n)  # (1, n)
    cross_entropy = -np.log(py)  # (1, n)
    regulation = _lambda * (np.sum(Ws[0]**2) + np.sum(Ws[1]**2))  # scalar
    J = (1/n) * np.sum(cross_entropy) + regulation  # scalar
    return J


def predict(p):
    return np.argmax(p, axis=0).reshape(1, p.shape[1])  # (1, n)


def compute_accuracy(X, y, Ws, bs):
    p, _ = evaluate_classifier(X, Ws, bs)
    predicted = predict(p)  # (1, n)
    return zero_one_loss(X, y, predicted)


def zero_one_loss(X, y, predicted):
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
    """ FORWARD PASS """
    P, H = evaluate_classifier(X, Ws, bs)

    """ BACKWARDS PASS """
    # From lec4 slides
    n = X.shape[1]
    G_batch = -(Y - P)  # (K, n)

    grad_W2 = (1/n) * np.matmul(G_batch, H.T)  # (K, m)
    grad_W2 += 2 * _lambda * Ws[1]  # Regulation term
    grad_b2 = (1/n) * np.sum(G_batch, axis=1, keepdims=True)  # (K, 1)

    G_batch = np.matmul(Ws[1].T, G_batch)  # (m, n)
    binary = np.ones_like(H) * 0.02  # np.zeros_like(H)  #
    binary[H > 0] = 1
    G_batch = np.multiply(G_batch, binary)

    grad_W1 = (1/n) * np.matmul(G_batch, X.T)  # (m, d)
    grad_W1 += 2 * _lambda * Ws[0]  # Regulation term
    grad_b1 = (1/n) * np.sum(G_batch, axis=1, keepdims=True)  # (m, 1)

    return [grad_W1, grad_W2], [grad_b1, grad_b2]


def compare_gradients(X, Y, Ws, bs, _lambda):
    h = 1e-5
    n_batch = 1
    X, Y = X[:, :n_batch], Y[:, :n_batch]
    grad_Ws, grad_bs = compute_gradients(X, Y, Ws, bs, _lambda)

    num_grad_Ws, num_grad_bs = compute_grads_num(
        X, Y, Ws, bs, _lambda, h, compute_cost)

    comp_W1 = relative_error(grad_Ws[0], num_grad_Ws[0])
    print("W1 relative error: ")
    print(comp_W1)
    comp_b1 = relative_error(grad_bs[0], num_grad_bs[0])
    print("b1 relative error:")
    print(comp_b1)

    comp_W2 = relative_error(grad_Ws[1], num_grad_Ws[1])
    print("W2 relative error: ")
    print(comp_W2)
    comp_b2 = relative_error(grad_bs[1], num_grad_bs[1])
    print("b2 relative error:")
    print(comp_b2)


def init_weights(size_in, size_out):
    xavier = 1/np.sqrt(size_in)
    W = np.random.normal(0, xavier, size=(size_out, size_in))
    b = np.random.normal(0, xavier, size=(size_out, 1))
    return W, b


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

        W1, b1 = init_weights(size_in=d, size_out=m)  # (m, d)
        W2, b2 = init_weights(size_in=m, size_out=K)  # (K, m)

        Ws = [W1, W2]
        bs = [b1, b2]

        save = False
        ret = train_model(X, Y, y, Ws, bs, l, n_batch, eta, n_epochs, X_valid,
                          Y_valid, y_valid, X_test, y_test, n_cycles=n_cycles, save=save)
        _, _, _, _, _, _, best_acc = ret
        f.write('Lambda: %f    best accuracy: %f\n\n' % (l, best_acc))

    f.close()


def ensamble_accuracy(X, y, models):
    n = X.shape[1]
    prediction_counts = np.zeros((10, n))
    for idx, model in enumerate(models):
        p, _ = evaluate_classifier(X, model[0], model[1])
        p = predict(p)  # (1, n)
        temp = np.zeros((K, n))
        temp[p.reshape(n), np.arange(n)] = 1
        prediction_counts += temp

    predicted = predict(prediction_counts)  # (1, n)
    return zero_one_loss(X, y, predicted)


def train_model(X, Y, y, Ws, bs, _lambda, n_batch, eta, n_epochs, X_valid, Y_valid, y_valid, X_test, y_test, n_cycles=-1, save=False, ensamble=False):
    costs_train = []
    costs_valid = []
    accs_train = []
    accs_valid = []
    etas = []
    grads = []
    best_acc = 0
    models = []

    # eta = 0.01  # This is wrong to get their graphs
    t = 0
    l = -1  # number of cycles completed
    for epoch_i in range(n_epochs):
        shuffle(X, Y)
        for X_batch, Y_batch in get_batches(n_batch, X, Y):
            grad_Ws, grad_bs = compute_gradients(
                X_batch, Y_batch, Ws, bs, _lambda)

            Ws[0] = Ws[0] - (eta * grad_Ws[0])
            bs[0] = bs[0] - (eta * grad_bs[0])
            Ws[1] = Ws[1] - (eta * grad_Ws[1])
            bs[1] = bs[1] - (eta * grad_bs[1])

            if t % (2 * n_s) == 0:
                # this is where eta = eta_min
                l += 1
                acc_valid = compute_accuracy(
                    X_valid, y_valid, Ws, bs)
                if acc_valid > best_acc:
                    best_acc = acc_valid
                if t != 0 and ensamble:
                    new_model_Ws = [np.copy(Ws[0]), np.copy(Ws[1])]
                    new_model_bs = [np.copy(bs[0]), np.copy(bs[1])]
                    models.append((new_model_Ws, new_model_bs))

            lower = 2 * n_s * l
            middle = (2 * l + 1) * n_s
            upper = 2 * (l + 1) * n_s
            # if l == 0:  # to get their graphs
            #     pass
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
                return costs_train, costs_valid, accs_train, accs_valid, etas, grads, best_acc, models
            t += 1
        print()
        print(".... Epoch %d completed ...." % (epoch_i))
        print()

    return costs_train, costs_valid, accs_train, accs_valid, etas, grads, best_acc, models


if __name__ == '__main__':
    # X, Y, y = load_batch('data_batch_1')
    # X_valid, Y_valid, y_valid = load_batch('data_batch_2')
    X_test, Y_test, y_test = load_batch('test_batch')
    X, Y, y, X_valid, Y_valid, y_valid, X_test, Y_test, y_test = load_batch_big(
        1000)

    # visulize_25(X)

    K = 10
    n_tot = X.shape[1]
    m = 80
    d = 3072

    _lambda = 4.1e-3
    n_batch = 100
    n_epochs = 200

    eta_min = 1e-5
    eta_max = 1e-1
    eta = eta_min
    # stepsize rule of thumb: n_s = k * (n_tot/n_batch) for 2 < k < 8
    n_s = 2 * np.floor(n_tot / n_batch)
    n_cycles = 5
    n_saves = round(n_s / 4)

    # lambda_grid_search()
    # exit()

    W1, b1 = init_weights(size_in=d, size_out=m)  # (m, d)
    W2, b2 = init_weights(size_in=m, size_out=K)  # (K, m)

    Ws = [W1, W2]
    bs = [b1, b2]

    # compare_gradients(X, Y, Ws, bs, _lambda)
    # exit()

    save = True
    ret = train_model(X, Y, y, Ws, bs, _lambda, n_batch, eta, n_epochs, X_valid,
                      Y_valid, y_valid, X_test, y_test, n_cycles=n_cycles, save=save, ensamble=True)
    costs_train, costs_valid, accs_train, accs_valid, etas, grads, best_acc, models = ret

    test_acc = compute_accuracy(X_test[:, :], y_test[:], Ws, bs)
    test_acc_ensamble = ensamble_accuracy(X_test[:, :], y_test[:], models)
    print("Test accuracy: $%f$ \\\\" % (test_acc))
    print("Test accuracy ensamble: $%f$ \\\\" % (test_acc_ensamble))
    print("Train accuracy: $%f$ \\\\" % (accs_train[-1]))
    print("Best valid accuracy: $%f$ \\\\" % (best_acc))

    if save:
        """ ETAS """
        # plt.plot(etas)
        # plt.show()
        """ Gradient spread """
        # fig, axes = plt.subplots(2, 1)
        # for layer in range(2):
        #     data = [g[layer].reshape(
        #         g[layer].shape[0] * g[layer].shape[1]) for g in grads]
        #     axes[layer].boxplot(data, 0, '', showfliers=False)
        #     axes[layer].set_title("Distribution of layer %d" % (layer + 1))
        # plt.show()
        """ training and validation cost """
        plt.plot(np.arange(len(costs_train))*n_saves,
                 costs_train, 'g', label='training loss')
        plt.plot(np.arange(len(costs_valid))*n_saves,
                 costs_valid, 'r', label='validation loss')
        plt.xlabel("update step")
        plt.ylabel("cost")
        plt.legend()
        plt.show()
        """ training and validation accuracy """
        plt.plot(np.arange(len(accs_train))*n_saves,
                 accs_train, 'g', label='training accuracy')
        plt.plot(np.arange(len(accs_valid))*n_saves, accs_valid,
                 'r', label='validation accuracy')
        plt.xlabel("update step")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()
        # visulize_weights(Ws[0], ('Weights in layer %d' % 1))
