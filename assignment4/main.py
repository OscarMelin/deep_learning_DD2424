from helpers import *
from RNN import RNN
import numpy as np


def synthesize():
    h0 = np.zeros((m, 1))
    x0 = make_one_hot([char_to_ind['.']], K)
    test = RNN.synthesize(h0, x0, seq_length)
    text = back_to_text(test)
    print(text)


def back_to_text(synthesized):
    text = ""
    for column in synthesized.T:
        text += ind_to_char[np.argmax(column)]
    return text.encode()


def compare_gradients():
    num_grads = numerical_gradients(RNN, X_chars, Y_chars, h)
    RNN.train(X_chars, Y_chars)
    V_error = relative_error(RNN.grad_V, num_grads['V'])
    print()
    print("V error:")
    print(V_error)
    W_error = relative_error(RNN.grad_W, num_grads['W'])
    print()
    print("W error:")
    print(W_error)
    U_error = relative_error(RNN.grad_U, num_grads['U'])
    print()
    print("U error:")
    print(U_error)
    b_error = relative_error(RNN.grad_b, num_grads['b'])
    print()
    print("b error:")
    print(b_error)
    c_error = relative_error(RNN.grad_c, num_grads['c'])
    print()
    print("c error:")
    print(c_error)


def get_batch():
    e = 0
    while e + seq_length < book_data:
        X_batch = book_data[e: seq_length + e]
        Y_batch = book_data[e + 1: seq_length + e + 1]
        X_batch = [char_to_ind[c] for c in X_batch]
        Y_batch = [char_to_ind[c] for c in Y_batch]
        X_batch = make_one_hot(X_batch, K)
        Y_batch = make_one_hot(Y_batch, K)
        e += seq_length
        yield X_batch, Y_batch


if __name__ == "__main__":
    book_chars, book_data = load_data()
    char_to_ind, ind_to_char = get_mappings(book_chars)

    K = len(book_chars)  # dimention 54
    m = 100  # hidden state size
    eta = 0.1
    seq_length = 25
    h = 1e-4
    n_epoch = 1

    np.random.seed(400)  # TODO: remove

    RNN = RNN(K, m, eta, seq_length)

    # compare_gradients()
    # exit()

    for epoch in range(n_epoch):
        e = 0
        for X_seq, Y_seq in get_batch():
            if e == 0:
                RNN.h0 = np.zeros((m, 1))
            RNN.train(X_seq, Y_seq)
