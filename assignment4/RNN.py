import numpy as np
from helpers import *


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
    def __init__(self, K, m, eta, seq_length):
        self.K = K
        self.m = m
        self.eta = eta
        self.seq_length = seq_length
        self.h0 = np.zeros((m, 1))

        # weights
        sig = 0.01
        self.weights = {
            'U': np.random.normal(0, np.sqrt(2 / K), size=(m, K)),
            'W': np.random.normal(0, np.sqrt(2 / m), size=(m, m)),
            'V': np.random.normal(0, np.sqrt(2 / m), size=(K, m)),
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

    def train(self, X_chars, Y_chars):
        """
        X_chars: (K, seq_length)
        Y_chars: (K, seq_length) one before X_chars
        """
        ps, hs, a_s, loss = self.forward(X_chars, Y_chars)
        self.backwards(ps, hs, a_s, X_chars, Y_chars)

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

    def backwards(self, ps, hs, a_s, X_chars, Y_chars):
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

    def save(self, smooth_loss):
        for k, v in self.weights.items():
            np.save(k, v)
        for k, v in self.ada_m.items():
            np.save('ada_m_' + k, v)
        np.save('smooth_loss', [smooth_loss])

    def load(self):
        for k in self.weights:
            self.weights[k] = np.load(k + '.npy')
        for k in self.ada_m:
            self.ada_m[k] = np.load('ada_m_' + k + '.npy')
        return np.load('smooth_loss.npy')[0]
