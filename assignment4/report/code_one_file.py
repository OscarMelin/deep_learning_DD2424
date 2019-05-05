import time
import datetime
import matplotlib.pyplot as plt
""" HELPERS FILE"""
import numpy as np


def load_data():
    f = open("goblet_book.txt", "r", encoding='utf-8')
    if f.mode != 'r':
        exit()
    book_data = f.read()
    f.close()
    # book_data = book_data.lower()
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


""" RNN FILE """


def softmax(s):
    """Compute softmax values for each sets of scores in s"""
    exps = np.exp(s)  # (K, n_batch)
    ones = np.ones((1, s.shape[0]))  # (1, K)
    denom = np.matmul(ones, np.exp(s))  # (1, n_batch)
    p = exps / denom  # (K, n_batch)
    return p


def sample_label(p):
    cp = np.cumsum(p)
    draw = np.random.rand()
    for idx, v in enumerate(cp):
        if v - draw > 0:
            return idx


class RNN:
    def __init__(self, K, m, eta, seq_length, init='normal'):
        self.K = K
        self.m = m
        self.eta = eta
        self.seq_length = seq_length
        self.h0 = np.zeros((m, 1))

        # weights
        if init == 'xavier':
            self.weights = {
                'U': np.random.normal(0, 1 / np.sqrt(K), size=(m, K)),
                'W': np.random.normal(0, 1 / np.sqrt(m), size=(m, m)),
                'V': np.random.normal(0, 1 / np.sqrt(m), size=(K, m)),
                'b': np.zeros((m, 1)),
                'c': np.zeros((K, 1))
            }
        else:
            sig = 0.01
            self.weights = {
                'U': np.random.normal(size=(m, K)) * sig,
                'W': np.random.normal(size=(m, m)) * sig,
                'V': np.random.normal(size=(K, m)) * sig,
                'b': np.zeros((m, 1)),
                'c': np.zeros((K, 1))
            }

        # gradients
        self.gradients = {
            'U': np.zeros_like(self.weights['U']),
            'W': np.zeros_like(self.weights['W']),
            'V': np.zeros_like(self.weights['V']),
            'b': np.zeros_like(self.weights['b']),
            'c': np.zeros_like(self.weights['c'])
        }

        self.ada_m = {
            'U': np.zeros_like(self.weights['U']),
            'W': np.zeros_like(self.weights['W']),
            'V': np.zeros_like(self.weights['V']),
            'b': np.zeros_like(self.weights['b']),
            'c': np.zeros_like(self.weights['c'])
        }

    def train(self, X_chars, Y_chars, clip=True):
        """
        X_chars: (K, seq_length)
        Y_chars: (K, seq_length) one before X_chars
        """
        ps, hs, a_s, loss = self.forward(X_chars, Y_chars)
        self.backwards(ps, hs, a_s, X_chars, Y_chars, clip)

        for k, v in self.ada_m.items():
            self.ada_m[k] = v + self.gradients[k]**2
        for k, v in self.gradients.items():
            self.weights[k] -= self.eta * self.gradients[k] / \
                np.sqrt(self.ada_m[k] + np.finfo(float).eps)

        return loss

    def forward(self, X_chars, Y_chars):
        n = X_chars.shape[1]
        h = self.h0
        a_s = np.empty((self.m, 0))
        hs = np.empty((self.m, 0))
        ps = np.empty((self.K, 0))
        for t in range(n):
            x_t = X_chars[:, t].reshape(self.K, 1)
            a = np.matmul(
                self.weights['W'], h) + np.matmul(self.weights['U'], x_t) + self.weights['b']
            a_s = np.append(a_s, a, axis=1)
            h = np.tanh(a)
            hs = np.append(hs, h, axis=1)
            o = np.matmul(self.weights['V'], h) + self.weights['c']
            p = softmax(o)
            ps = np.append(ps, p, axis=1)

        loss = 0
        for t in range(n):
            y_t = Y_chars[:, t].reshape(self.K, 1)
            p_t = ps[:, t].reshape(self.K, 1)
            res = np.asscalar(np.matmul(y_t.T, p_t))
            loss += np.log(res)
        loss = - loss
        return ps, hs, a_s, loss

    def backwards(self, ps, hs, a_s, X_chars, Y_chars, clip):
        n = X_chars.shape[1]
        # propagate back over sigmoid
        g = -(Y_chars - ps).T
        # Clac grad V and c
        self.gradients['V'] = np.matmul(g.T, hs.T)
        self.gradients['c'] = np.sum(g.T, axis=1, keepdims=True)

        # propagate back over h and a (backwards in time)
        last_dl_do = g[-1, :].reshape(1, self.K)
        dl_dh_tau = np.matmul(last_dl_do, self.weights['V'])  # (1, 100)
        inner = 1 - np.tanh(a_s[:, -1])**2  # (100,)
        dl_da_tau = np.matmul(dl_dh_tau, np.diag(inner))  # (1, 100)
        dl_da = np.zeros((n, self.m))  # (n, 100)
        dl_da[-1, :] = dl_da_tau
        for t in range(n-2, -1, -1):
            prev = dl_da[t+1, :].reshape(1, self.m)
            dl_dh_t = np.matmul(g[t, :].reshape(
                1, self.K), self.weights['V']) + np.matmul(prev, self.weights['W'])
            dl_da_t = np.matmul(dl_dh_t, np.diag(1 - np.tanh(a_s[:, t]**2)))
            dl_da[t, :] = dl_da_t

        g = dl_da
        # Calc grad W and b
        h0 = np.zeros((self.m, 1))
        hs = np.concatenate((h0, hs), axis=1)
        self.h0 = hs[:, -1].reshape(self.m, 1)
        hs = hs[:, :-1]
        self.gradients['W'] = np.matmul(g.T, hs.T)
        self.gradients['b'] = np.sum(g.T, axis=1, keepdims=True)
        # Calc grad U
        self.gradients['U'] = np.matmul(g.T, X_chars.T)

        if clip:
            self.clip_gradients()

    def clip_gradients(self):
        for k, v in self.gradients.items():
            self.gradients[k] = np.clip(v, -5, 5)

    def synthesize(self, x0, n):
        """
        h0: first hidden state (m, 1)
        x0: first dummy input (K, 1)
        n: length of sequence to generate
        """
        Y = np.empty((x0.shape[0], 0))
        h = self.h0
        x = x0
        for t in range(n):
            a = np.matmul(
                self.weights['W'], h) + np.matmul(self.weights['U'], x) + self.weights['b']
            h = np.tanh(a)
            o = np.matmul(self.weights['V'], h) + self.weights['c']
            p = softmax(o)

            idx = sample_label(p)

            x = make_one_hot([idx], self.K)
            Y = np.append(Y, x, axis=1)

        return Y

    def save(self, smooth_loss, step, epoch):
        for k, v in self.weights.items():
            np.save('weights/' + k, v)
        for k, v in self.ada_m.items():
            np.save('weights/ada_m_' + k, v)
        np.save('weights/training_params', [smooth_loss, step, epoch])

    def load(self):
        for k in self.weights:
            self.weights[k] = np.load('weights/' + k + '.npy')
        for k in self.ada_m:
            self.ada_m[k] = np.load('weights/ada_m_' + k + '.npy')
        tr_params = np.load('weights/training_params.npy')
        return tr_params[0], tr_params[1], tr_params[2]


""" MAIN FILE """


def synthesize(X_seq):
    x0 = X_seq[:, 0:1]
    synth = RNN.synthesize(x0, 200)
    text = ""
    for column in synth.T:
        text += ind_to_char[np.argmax(column)]
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
    n_epoch = 20

    # np.random.seed(400)  # TODO: remove
    # compare_gradients()

    RNN = RNN(K, m, eta, seq_length, init='xavier')

    save = True
    smooth_loss = -1
    step = -1
    last_epoch = 0
    if save:
        smooth_loss, step, last_epoch = RNN.load()
        print('last smooth_loss: %f \t last step: %d \t last epoch: %d' %
              (smooth_loss, step, last_epoch))

    synth = RNN.synthesize(make_one_hot([char_to_ind['.']], K), 1000)
    text = ""
    for column in synth.T:
        text += ind_to_char[np.argmax(column)]
    print(text.encode('ascii', 'ignore').decode('ascii'))
    exit()

    losses = []
    f = open('synthesized-' +
             str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')), 'w+')
    for epoch in range(n_epoch):
        print("\t\t---NEW EPOCH--- number: %d" % (epoch + last_epoch))
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
            RNN.save(smooth_loss, step, epoch + last_epoch + 1)

    plt.plot(losses, 'g', label='losses')
    plt.ylabel("loss")
    plt.legend()
    plt.show()
