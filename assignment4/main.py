from helpers import *
from RNN import RNN
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time


def synthesize(X_seq):
    x0 = X_seq[:, 0:1]
    synth = RNN.synthesize(x0, 140)
    text = ""
    for column in synth.T:
        new_char = ind_to_char[np.argmax(column)]
        if new_char != 'Ω':
            text += new_char
        else:
            text += new_char
            break
    return text


def compare_gradients():
    tRNN = RNN(K, m, eta, seq_length, init='normal')
    for X_chars, Y_chars in get_batch():
        num_grads = numerical_gradients(tRNN, X_chars, Y_chars, h)
        tRNN.train(X_chars, Y_chars, clip=False)
        for k in tRNN.weights:
            error = relative_error(tRNN.gradients[k], num_grads[k])
            print("\n%s error:" % k)
            print(error)
        exit()


def get_batch():
    for current_tweet in tweets:
        e = 0
        while e + seq_length < len(current_tweet):
            X_batch = current_tweet[e: seq_length + e]
            X_batch = [char_to_ind[c] for c in X_batch]
            X_batch = make_one_hot(X_batch, K)
            Y_batch = current_tweet[e + 1: seq_length + e + 1]
            Y_batch = [char_to_ind[c] for c in Y_batch]
            Y_batch = make_one_hot(Y_batch, K)
            e += seq_length
            yield X_batch, Y_batch, e == seq_length
        if len(current_tweet) - e > 2:
            X_batch = current_tweet[e: len(current_tweet)-1]
            X_batch = [char_to_ind[c] for c in X_batch]
            X_batch = make_one_hot(X_batch, K)
            Y_batch = current_tweet[e + 1: len(current_tweet)]
            Y_batch = [char_to_ind[c] for c in Y_batch]
            Y_batch = make_one_hot(Y_batch, K)
            yield X_batch, Y_batch, False


if __name__ == "__main__":
    book_chars, book_data, tweets = load_data()
    # Stopping characted
    book_chars.append('Ω')
    char_to_ind, ind_to_char = get_mappings(book_chars)

    K = len(book_chars)  # dimention 269
    m = 100  # hidden state size
    eta = 0.1
    seq_length = 15
    n_epoch = 3

    # np.random.seed(400)  # TODO: remove
    h = 1e-4
    # compare_gradients()

    RNN = RNN(K, m, eta, init='xavier')

    save = True
    load = True

    smooth_loss = -1
    step = -1
    last_epoch = 0
    if load:
        smooth_loss, step, last_epoch = RNN.load()
        print('last smooth_loss: %f \t last step: %d \t last epoch: %d' %
              (smooth_loss, step, last_epoch))

    text = synthesize(make_one_hot([char_to_ind['M']], K))
    print(text.encode('ascii', 'ignore').decode('ascii'))
    exit()

    losses = []
    f = open('synthesized/seq_' + str(seq_length) + '_' +
             str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')) + '.txt', 'wb+')
    for epoch in range(n_epoch):
        print("\t\t---NEW EPOCH--- number: %d/%d" %
              (epoch + last_epoch, n_epoch + last_epoch - 1))
        for X_seq, Y_seq, reset in get_batch():
            if reset:
                RNN.h0 = np.zeros((m, 1))
            step += 1
            loss = RNN.train(X_seq, Y_seq)
            smooth_loss = 0.999*smooth_loss + 0.001*loss if smooth_loss != -1 else loss
            losses.append(smooth_loss)

            if step % 1000 == 0:
                f.write(('\n\tSynthesized text at iteration: %d with smooth loss: %f\n' % (
                    step, smooth_loss)).encode('utf8'))
                text = synthesize(X_seq)
                f.write(text.encode('utf8'))
                f.write('\n'.encode('utf8'))
                f.flush()
            elif step % 499 == 0:
                print(' ... Smooth loss: %f ...' % (smooth_loss))
        if save:
            RNN.save(smooth_loss, step, epoch + last_epoch + 1)

    plt.plot(losses, 'g', label='losses')
    plt.ylabel("loss")
    plt.legend()
    plt.show()
