import numpy as np


def load_data():
    f = open("goblet_book.txt", "r", encoding='utf-8')
    if f.mode != 'r':
        exit()
    content = f.read()
    f.close()
    book_data = content.lower()
    # Get all unique chars
    book_chars = list(book_data)
    book_chars = set(book_chars)
    book_chars = list(book_chars)
    book_chars = sorted(book_chars)
    return book_chars, book_data


def get_mappings(book_chars):
    char_to_ind = {}
    for k, v in enumerate(book_chars):
        char_to_ind[v] = k
    ind_to_char = {}
    for k, v in enumerate(book_chars):
        ind_to_char[k] = v

    return char_to_ind, ind_to_char


def make_one_hot(y, K):
    """ create one-hot column vectors """
    one_hot = np.zeros((K, len(y)))
    for i in range(len(y)):
        one_hot[y[i], i] = 1.
    return one_hot


def relative_error(grad, num_grad):
    assert grad.shape == num_grad.shape
    nominator = np.sum(np.abs(grad - num_grad))
    demonimator = max(1e-6, np.sum(np.abs(grad)) + np.sum(np.abs(num_grad)))
    return nominator / demonimator


def numerical_gradients(RNN, X, Y, h):
    grads = {}
    print("Begin numerical gradient for U")
    grads['U'] = numerical_gradient(RNN, RNN.U, X, Y, h)
    print("Begin numerical gradient for W")
    grads['W'] = numerical_gradient(RNN, RNN.W, X, Y, h)
    print("Begin numerical gradient for V")
    grads['V'] = numerical_gradient(RNN, RNN.V, X, Y, h)
    print("Begin numerical gradient for b")
    grads['b'] = numerical_gradient(RNN, RNN.b, X, Y, h)
    print("Begin numerical gradient for c")
    grads['c'] = numerical_gradient(RNN, RNN.c, X, Y, h)
    return grads


def numerical_gradient(RNN, var, X, Y, h):
    grad = np.zeros_like(var)
    for i in range(var.shape[0]):
        for j in range(var.shape[1]):
            # normal
            normal = var[i][j]
            # back
            var[i][j] -= h
            _, _, _, c1 = RNN.forward(X, Y)
            var[i][j] = normal
            # forw
            var[i][j] += h
            _, _, _, c2 = RNN.forward(X, Y)
            var[i][j] = normal
            grad[i][j] = (c2-c1) / (2*h)
    return grad
