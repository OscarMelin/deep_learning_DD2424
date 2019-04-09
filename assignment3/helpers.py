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
