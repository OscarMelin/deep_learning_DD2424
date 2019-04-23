from helpers import *
from RNN import RNN
import numpy as np
import matplotlib.pyplot as plt


def synthesize(X_seq):
    x0 = X_seq[:, 0:1]
    synth = RNN.synthesize(x0, 200)
    text = ""
    for column in synth.T:
        text += ind_to_char[np.argmax(column)]
    return text


def compare_gradients():
    X_chars, Y_chars = get_batch()
    num_grads = numerical_gradients(RNN, X_chars, Y_chars, h)
    RNN.train(X_chars, Y_chars)
    V_error = relative_error(RNN.grad_V, num_grads['V'])
    print("\nV error:")
    print(V_error)
    W_error = relative_error(RNN.grad_W, num_grads['W'])
    print("\nW error:")
    print(W_error)
    U_error = relative_error(RNN.grad_U, num_grads['U'])
    print("\nU error:")
    print(U_error)
    b_error = relative_error(RNN.grad_b, num_grads['b'])
    print("\nb error:")
    print(b_error)
    c_error = relative_error(RNN.grad_c, num_grads['c'])
    print("\nc error:")
    print(c_error)
    exit()


def get_batch():
    e = 0
    while e + seq_length < len(book_data):
        X_batch = book_data[e: seq_length + e]
        X_batch = [char_to_ind[c] for c in X_batch]
        X_batch = make_one_hot(X_batch, K)
        Y_batch = book_data[e + 1: seq_length + e + 1]
        Y_batch = [char_to_ind[c] for c in Y_batch]
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
    n_epoch = 3
    save = True

    # np.random.seed(400)  # TODO: remove

    RNN = RNN(K, m, eta, seq_length)

    compare_gradients()

    f = open('synthesized.txt', 'w+')
    smooth_loss = -1
    if save:
        smooth_loss = RNN.load()
    losses = []
    step = -1
    for epoch in range(n_epoch):
        print("\t\t---NEW EPOCH--- number: %d" % epoch)
        RNN.h0 = np.zeros((m, 1))
        for X_seq, Y_seq in get_batch():
            step += 1
            loss = RNN.train(X_seq, Y_seq)
            smooth_loss = 0.999*smooth_loss + 0.001*loss if smooth_loss != -1 else loss
            losses.append(smooth_loss)

            if step % 500 == 0:
                f.write('\n\tSynthesized text at iteration: %d with smooth loss: %f\n' % (
                    step, smooth_loss))
                text = synthesize(X_seq)
                f.write(text.encode('ascii', 'ignore').decode('ascii'))
                f.write('\n')
                f.flush()
            elif step % 100 == 0:
                print(' ... Smooth loss: %f ...' % (smooth_loss))
        if save:
            RNN.save(smooth_loss)

    plt.plot(losses, 'g', label='losses')
    plt.ylabel("loss")
    plt.legend()
    plt.show()
