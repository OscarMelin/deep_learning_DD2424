import numpy as np
import json


def load_data():
    f = open("trump_2018.json", "r", encoding='utf-8')
    if f.mode != 'r':
        exit()
    book_data = json.load(f)
    f.close()

    tweets = [tweet_info['text']+'Ω' for tweet_info in book_data]

    book_data = " ".join(tweets)
    # Get all unique chars
    book_chars = list(book_data)
    book_chars = set(book_chars)
    book_chars = list(book_chars)
    book_chars = sorted(book_chars)
    return book_chars, book_data, tweets


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
    for k in RNN.weights:
        print("Begin numerical gradient for " + k)
        grads[k] = numerical_gradient(RNN, k, X, Y, h)
    return grads


def numerical_gradient(RNN, var, X, Y, h):
    grad = np.zeros_like(RNN.weights[var])
    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            # normal
            normal = RNN.weights[var][i][j]
            # back
            RNN.weights[var][i][j] -= h
            _, _, _, c1 = RNN.forward(X, Y)
            RNN.weights[var][i][j] = normal
            # forw
            RNN.weights[var][i][j] += h
            _, _, _, c2 = RNN.forward(X, Y)
            RNN.weights[var][i][j] = normal
            grad[i][j] = (c2-c1) / (2*h)
    return grad
