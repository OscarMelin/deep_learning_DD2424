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


def compute_grads_num_slow(X, Y, Ws, bs, gammas, betas, h, compute_cost, batch_norm):
    no = Ws[1].shape[0]
    d = X.shape[0]
    n_layers = len(Ws)

    grad_Ws = []
    grad_bs = []
    grad_gammas = []
    grad_betas = []

    for layer in range(n_layers):
        grad_W = np.zeros_like(Ws[layer])
        grad_b = np.zeros_like(bs[layer])
        if batch_norm and layer != n_layers - 1:
            grad_gamma = np.zeros_like(gammas[layer])
            grad_beta = np.zeros_like(betas[layer])

        print("Starting with b for Layer %d" % layer)
        for i in range(bs[layer].shape[0]):
            # back
            b_try = np.copy(bs[layer])
            b_try[i] = b_try[i] - h
            temp = bs[layer]
            bs[layer] = b_try
            c1 = compute_cost(X, Y, Ws, bs, gammas, betas, False)
            bs[layer] = temp
            # forw
            b_try = np.copy(bs[layer])
            b_try[i] = b_try[i] + h
            temp = bs[layer]
            bs[layer] = b_try
            c2 = compute_cost(X, Y, Ws, bs, gammas, betas, False)
            bs[layer] = temp
            grad_b[i] = (c2-c1) / (2*h)

        print("Starting with W for Layer %d" % layer)
        for i in range(Ws[layer].shape[0]):
            for j in range(Ws[layer].shape[1]):
                # back
                W_try = np.copy(Ws[layer])
                W_try[i][j] = W_try[i][j] - h
                temp = Ws[layer]
                Ws[layer] = W_try
                c1 = compute_cost(X, Y, Ws, bs, gammas, betas, False)
                Ws[layer] = temp
                # forw
                W_try = np.copy(Ws[layer])
                W_try[i][j] = W_try[i][j] + h
                temp = Ws[layer]
                Ws[layer] = W_try
                c2 = compute_cost(X, Y, Ws, bs, gammas, betas, False)
                Ws[layer] = temp
                grad_W[i][j] = (c2-c1) / (2*h)

        grad_Ws.append(grad_W)
        grad_bs.append(grad_b)
        print("Done with W and bs for layer %d" % layer)

        if batch_norm and layer != n_layers - 1:  # no batch norm last layer
            print("Starting with gamma for Layer %d" % layer)
            for i in range(gammas[layer].shape[0]):
                # back
                gammas_try = np.copy(gammas[layer])
                gammas_try[i] = gammas_try[i] - h
                temp = gammas[layer]
                gammas[layer] = gammas_try
                c1 = compute_cost(X, Y, Ws, bs, gammas, betas, False)
                gammas[layer] = temp
                # forw
                gammas_try = np.copy(gammas[layer])
                gammas_try[i] = gammas_try[i] + h
                temp = gammas[layer]
                gammas[layer] = gammas_try
                c2 = compute_cost(X, Y, Ws, bs, gammas, betas, False)
                gammas[layer] = temp
                grad_gamma[i] = (c2-c1) / (2*h)

            print("Starting with beta for Layer %d" % layer)
            for i in range(betas[layer].shape[0]):
                # back
                beta_try = np.copy(betas[layer])
                beta_try[i] = beta_try[i] - h
                temp = betas[layer]
                betas[layer] = beta_try
                c1 = compute_cost(X, Y, Ws, bs, gammas, betas, False)
                betas[layer] = temp
                # forw
                beta_try = np.copy(betas[layer])
                beta_try[i] = beta_try[i] + h
                temp = betas[layer]
                betas[layer] = beta_try
                c2 = compute_cost(X, Y, Ws, bs, gammas, betas, False)
                betas[layer] = temp
                grad_beta[i] = (c2-c1) / (2*h)

            grad_gammas.append(grad_gamma)
            grad_betas.append(grad_beta)
            print("Done with gamma and beta for Layer %d" % layer)

    return grad_Ws, grad_bs, grad_gammas, grad_betas


def compute_grads_num(X, Y, Ws, bs, gammas, betas, h, compute_cost, batch_norm):
    no = Ws[1].shape[0]
    d = X.shape[0]
    n_layers = len(Ws)

    grad_Ws = []
    grad_bs = []
    grad_gammas = []
    grad_betas = []

    c = compute_cost(X, Y, Ws, bs, gammas, betas, False)

    for layer in range(n_layers):
        grad_W = np.zeros_like(Ws[layer])
        grad_b = np.zeros_like(bs[layer])
        if batch_norm and layer != n_layers - 1:
            grad_gamma = np.zeros_like(gammas[layer])
            grad_beta = np.zeros_like(betas[layer])

        print("Starting with b for Layer %d" % layer)
        for i in range(bs[layer].shape[0]):
            b_try = np.copy(bs[layer])
            b_try[i] = b_try[i] + h
            temp = bs[layer]
            bs[layer] = b_try
            c2 = compute_cost(X, Y, Ws, bs, gammas, betas, False)
            bs[layer] = temp
            grad_b[i] = (c2-c) / h

        print("Starting with W for Layer %d" % layer)
        for i in range(Ws[layer].shape[0]):
            for j in range(Ws[layer].shape[1]):
                W_try = np.copy(Ws[layer])
                W_try[i][j] = W_try[i][j] + h
                temp = Ws[layer]
                Ws[layer] = W_try
                c2 = compute_cost(X, Y, Ws, bs, gammas, betas, False)
                Ws[layer] = temp
                grad_W[i][j] = (c2-c) / h

        grad_Ws.append(grad_W)
        grad_bs.append(grad_b)
        print("Done with W and bs for layer %d" % layer)

        if batch_norm and layer != n_layers - 1:  # no batch norm last layer
            print("Starting with gamma for Layer %d" % layer)
            for i in range(gammas[layer].shape[0]):
                gammas_try = np.copy(gammas[layer])
                gammas_try[i] = gammas_try[i] + h
                temp = gammas[layer]
                gammas[layer] = gammas_try
                c2 = compute_cost(X, Y, Ws, bs, gammas, betas, False)
                gammas[layer] = temp
                grad_gamma[i] = (c2-c) / h

            print("Starting with beta for Layer %d" % layer)
            for i in range(betas[layer].shape[0]):
                beta_try = np.copy(betas[layer])
                beta_try[i] = beta_try[i] + h
                temp = betas[layer]
                betas[layer] = beta_try
                c2 = compute_cost(X, Y, Ws, bs, gammas, betas, False)
                betas[layer] = temp
                grad_beta[i] = (c2-c) / h

            grad_gammas.append(grad_gamma)
            grad_betas.append(grad_beta)
            print("Done with gamma and beta for Layer %d" % layer)

    return grad_Ws, grad_bs, grad_gammas, grad_betas


def relative_error(grad, num_grad):
    assert grad.shape == num_grad.shape
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
            im = X[:, i].reshape(32, 32, 3)
            im = (im - np.min(im)) / (np.max(im) - np.min(im))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(im)
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

def softmax(s):
    """Compute softmax values for each sets of scores in s"""
    exps = np.exp(s)  # (K, n_batch)
    ones = np.ones((1, s.shape[0]))  # (1, K)
    denom = np.matmul(ones, np.exp(s))  # (1, n_batch)
    p = exps / denom  # (K, n_batch)
    return p


def relu(s):
    return np.maximum(0, s)


def BatchNorm(s, mu=None, var=None):
    n = s.shape[1]
    if mu is None:
        mu = np.mean(s, axis=1, keepdims=True)
    if var is None:
        var = np.var(s, axis=1, keepdims=False)  # False to get list for diag
    diag = np.diag(1 / (np.sqrt(var)))
    s_hat = np.matmul(diag, s - mu)
    return s_hat, mu, var


def BatchNormBackPass(G_batch, S, mu, var):
    n=S.shape[1]
    sigma_1 = 1. / np.sqrt(np.mean((S-mu)**2, axis=1, keepdims=True))
    sigma_2 = sigma_1 ** 3
    G_1=np.multiply(G_batch, np.repeat(sigma_1, n, axis=1))
    G_2=np.multiply(G_batch, np.repeat(sigma_2, n, axis=1))
    D=S - np.repeat(mu, n, axis = 1)
    c=np.sum(np.multiply(G_2, D), axis = 1, keepdims = True)
    G_batch=G_1
    temp=(1./n) * np.sum(G_1, axis = 1, keepdims = True)
    G_batch -= temp
    temp=(1./n) * np.multiply(D, np.repeat(c, n, axis=1))
    G_batch -= temp
    return G_batch


def evaluate_classifier(X, Ws, bs, gammas, betas, use_avg = False):
    """
    Ws = [(m, d), (K, m)]
    bs = [(m, 1), (K, 1)]
    X = (3072, n)
    p = (10, n)
    """
    n=X.shape[1]

    Hs=[np.copy(X)]
    Ss=[]
    S_hats=[]
    mus=[]
    _vars=[]
    for layer in range(n_layers - 1):
        WX=np.matmul(Ws[layer], Hs[layer])  # (m, n)
        # repeat column vector b, n times
        b_big1=np.repeat(bs[layer], n, axis = 1)
        s=WX + b_big1  # (m, n)
        if batch_norm:
            Ss.append(s)
            if use_avg:
                s_hat, mu, var=BatchNorm(
                    s, mu = mus_avg[layer], var = _vars_avg[layer])
            else:
                s_hat, mu, var=BatchNorm(s)
            S_hats.append(s_hat)
            mus.append(mu)
            _vars.append(var)
            s_tilde=np.multiply(gammas[layer], s_hat) + betas[layer]
            s=s_tilde
        h=relu(s)  # (m, n)
        Hs.append(h)

    # Last Layer
    WX=np.matmul(Ws[-1], Hs[-1])  # (K, n)
    b_big2=np.repeat(bs[-1], n, axis = 1)
    s=WX + b_big2
    p=softmax(s)
    return p, Hs, Ss, S_hats, mus, _vars


def compute_cost(X, Y, Ws, bs, gammas, betas, use_avg):
    """
    Ws = [(m, d), (K, m)]
    bs = [(m, 1), (K, 1)]
    X = (3072, n)
    p = (10, n)
    """
    n=X.shape[1]

    p, _, _, _, _, _=evaluate_classifier(
        X, Ws, bs, gammas, betas, use_avg=use_avg)
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


def compute_accuracy(X, y, Ws, bs, gammas, betas):
    p, _, _, _, _, _ = evaluate_classifier(
        X, Ws, bs, gammas, betas, use_avg=True)
    predicted = predict(p)

    zero_one_loss = 0
    for i in range(X.shape[1]):
        if predicted[0, i] == y[i]:
            zero_one_loss += 1

    accuracy = zero_one_loss / X.shape[1]
    return accuracy


def compute_gradients(X, Y, Ws, bs, gammas, betas):
    """
    X = (3072, n)
    Y = (10, n)
    P = (10, n)
    """
    n = X.shape[1]

    """ FORWARD PASS """
    P, Hs, Ss, S_hats, mus, _vars = evaluate_classifier(
        X, Ws, bs, gammas, betas)

    global mus_avg
    global _vars_avg
    if len(mus_avg) == 0:
        # init
        mus_avg = [mu for mu in mus]
        _vars_avg = [mu for mu in _vars]
    else:
        # update
        for layer in range(len(mus)):
            mus_avg[layer] = alpha * mus_avg[layer] + (1-alpha) * mus[layer]
            _vars_avg[layer] = alpha * \
                _vars_avg[layer] + (1-alpha) * _vars[layer]

    """ BACKWARDS PASS """
    # From lec4 slides 34
    grad_Ws = []
    grad_bs = []
    grad_gammas = []
    grad_betas = []

    G_batch = -(Y - P)  # Softmax backprop(K, n)

    # Last layer no batch norm
    grad_W = (1/n) * np.matmul(G_batch, Hs[-1].T)
    grad_W += 2 * _lambda * Ws[-1]  # Regulation term
    grad_b = (1/n) * np.sum(G_batch, axis=1, keepdims=True)

    grad_Ws.append(grad_W)
    grad_bs.append(grad_b)

    G_batch = np.matmul(Ws[-1].T, G_batch)  # (m, n)
    binary = np.zeros_like(Hs[-1])
    binary[Hs[-1] > 0] = 1
    G_batch = np.multiply(G_batch, binary)

    for layer in range(n_layers-2, -1, -1):
        if batch_norm:
            grad_gamma = (1./n) * np.sum(np.multiply(G_batch,
                                                    S_hats[layer]), axis=1, keepdims=True)
            grad_beta = (1./n) * np.sum(G_batch, axis=1, keepdims=True)
            grad_gammas.append(grad_gamma)
            grad_betas.append(grad_beta)
            
            # propagate throught scale and shift
            G_batch = np.multiply(G_batch, np.repeat(gammas[layer], n, axis=1))
            # propagate throught batch norm
            G_batch = BatchNormBackPass(
                G_batch, Ss[layer], mus[layer], _vars[layer])

        grad_W = (1./n) * np.matmul(G_batch, Hs[layer].T)
        grad_W += 2 * _lambda * Ws[layer]  # Regulation term
        grad_b = (1./n) * np.sum(G_batch, axis=1, keepdims=True)
        grad_Ws.append(grad_W)
        grad_bs.append(grad_b)

        # Porpagate through relu
        if layer == 0:
            # no need to propagate further
            break
        G_batch = np.matmul(Ws[layer].T, G_batch)  # (m, n)
        binary = np.zeros_like(Hs[layer])
        binary[Hs[layer] > 0] = 1
        G_batch = np.multiply(G_batch, binary)

    grad_Ws.reverse()
    grad_bs.reverse()
    grad_gammas.reverse()
    grad_betas.reverse()

    return grad_Ws, grad_bs, grad_gammas, grad_betas


def compare_gradients(X, Y, Ws, bs, gammas, betas):
    h = 1e-5
    n_batch = 100  # has to be > 1 for batch norm > 2 for it to work
    X, Y = X[:d, :n_batch], Y[:, :n_batch]

    grad_Ws, grad_bs, grad_gammas, grad_betas = compute_gradients(
        X, Y, Ws, bs, gammas, betas)

    num_grad_Ws, num_grad_bs, num_grad_gammas, num_grad_betas = compute_grads_num(
        X, Y, Ws, bs, gammas, betas, h, compute_cost, batch_norm)

    for layer in range(n_layers):
        print("\n\tLayer %d\\\\" % layer)
        comp_W = relative_error(
            grad_Ws[layer], num_grad_Ws[layer])
        print("gradient W%d relative error: %s\\\\\n" %
              (layer, comp_W.astype(str)))

        comp_b = relative_error(
            grad_bs[layer], num_grad_bs[layer])
        print("gradient b%d relative error: %s\\\\\n" %
              (layer, comp_b.astype(str)))

        if batch_norm and layer != n_layers - 1:
            comp_gamma = relative_error(
                grad_gammas[layer], num_grad_gammas[layer])
            print("gradient gamma%d relative error: %s\\\\\n" %
                  (layer, comp_gamma.astype(str)))

            comp_beta = relative_error(
                grad_betas[layer], num_grad_betas[layer])
            print("gradient beta%d relative error: %s\\\\\n" %
                  (layer, comp_beta.astype(str)))


def init_weights(size_in, size_out):
    xavier = 1/np.sqrt(size_in)
    he = np.sqrt(2 / size_in)
    xavier = he
    W = np.random.normal(0, xavier, size=(size_out, size_in))
    b = np.random.normal(0, xavier, size=(size_out, 1))
    if batch_norm:
        gamma = np.random.normal(0, xavier, size=(size_out, 1))
        beta = np.random.normal(0, xavier, size=(size_out, 1))
        return W, b, gamma, beta
    return W, b, None, None


def init_layers(layers):
    Ws = []
    bs = []
    gammas = []
    betas = []
    W, b, gamma, beta = init_weights(size_in=d, size_out=layers[0])
    Ws.append(W)
    bs.append(b)
    if batch_norm:
        gammas.append(gamma)
        betas.append(beta)
    for layer in range(1, n_layers):
        W, b, gamma, beta = init_weights(
            size_in=Ws[layer-1].shape[0], size_out=layers[layer])
        Ws.append(W)
        bs.append(b)
        if batch_norm and layer != n_layers - 1:  # no batch norm in last layer
            gammas.append(gamma)
            betas.append(beta)
    return Ws, bs, gammas, betas


def train_model(X, Y, y, Ws, bs, gammas, betas, X_valid, Y_valid, y_valid, X_test, y_test):
    costs_train = []
    costs_valid = []
    grads = []
    best_valid_acc = 0
    best_test_acc = 0
    eta = eta_min

    t = 0
    l = -1  # number of cycles completed
    for epoch_i in range(n_epochs):
        shuffle(X, Y)
        for X_batch, Y_batch in get_batches(n_batch, X, Y):
            grad_Ws, grad_bs, grad_gammas, grad_betas = compute_gradients(
                X_batch, Y_batch, Ws, bs, gammas, betas)

            for layer in range(n_layers):
                Ws[layer] = Ws[layer] - (eta * grad_Ws[layer])
                bs[layer] = bs[layer] - (eta * grad_bs[layer])
                if batch_norm:
                    if layer == n_layers - 1:
                        break  # no batch norm for last layer
                    gammas[layer] = gammas[layer] - (eta * grad_gammas[layer])
                    betas[layer] = betas[layer] - (eta * grad_betas[layer])

            if t % (2 * n_s) == 0:
                # this is where eta = eta_min
                l += 1
                acc_valid = compute_accuracy(
                    X_valid, y_valid, Ws, bs, gammas, betas)
                if acc_valid > best_valid_acc:
                    best_valid_acc = acc_valid
                acc_test = compute_accuracy(
                    X_test, y_test, Ws, bs, gammas, betas)
                if acc_test > best_test_acc:
                    best_test_acc = acc_test
            lower = 2 * n_s * l
            middle = (2 * l + 1) * n_s
            upper = 2 * (l + 1) * n_s
            if lower <= t and t <= middle:
                eta = eta_min + (t - 2 * l * n_s) / n_s * (eta_max - eta_min)
            elif middle <= t and t <= upper:
                eta = eta_max - (t - (2 * l + 1) * n_s) / \
                    n_s * (eta_max - eta_min)

            if save:
                if t % n_saves == 0:
                    costs_train.append(compute_cost(
                        X, Y, Ws, bs, gammas, betas, use_avg=True))
                    costs_valid.append(compute_cost(
                        X_valid, Y_valid, Ws, bs, gammas, betas, use_avg=True))
            if l == n_cycles:
                # Return after n_cycles
                return costs_train, costs_valid, grads, best_valid_acc, best_test_acc
            t += 1
        print()
        print(".... Epoch %d completed ...." % (epoch_i))
        print()

    return costs_train, costs_valid, grads, best_valid_acc, best_test_acc


def lambda_grid_search():
    global mus_avg
    global _vars_avg
    global save
    global _lambda
    l_min = -3.9
    l_max = -4.1
    n_lambdas = 10
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
        _lambda = l
        count += 1

        Ws, bs, gammas, betas = init_layers(layers)
        mus_avg = []
        _vars_avg = []

        save = False
        ret = train_model(X, Y, y, Ws, bs, gammas, betas, X_valid,
                    Y_valid, y_valid, X_test, y_test)
        _, _, _, best_valid_acc, best_valid_acc = ret
        f.write('Lambda: %f    best accuracy: %f\n\n' % (l, best_valid_acc))

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

    _lambda = 0.005 #0.000715
    n_batch = 100
    n_epochs = 200

    eta_min = 1e-5
    eta_max = 1e-1

    # stepsize rule of thumb: n_s = k * (n_tot/n_batch) for 2 < k < 8
    n_s = 5 * np.floor(n_tot / n_batch)
    n_cycles = 2
    n_saves = round(n_s / 4)

    batch_norm = True
    alpha = 0.9

    save = True

    # out of layers, gotta have K at the end
    # layers = [50, 30, 20, 20, 10, 10, 10, 10, K]
    layers = [50, 50, K]
    n_layers = len(layers)

    # lambda_grid_search()
    # exit()

    Ws, bs, gammas, betas = init_layers(layers)
    mus_avg = []
    _vars_avg = []

    # compare_gradients(X, Y, Ws, bs, gammas, betas)
    # exit()

    ret = train_model(X, Y, y, Ws, bs, gammas, betas, X_valid,
                      Y_valid, y_valid, X_test, y_test)
    costs_train, costs_valid, grads, best_valid_acc, best_test_acc = ret

    test_acc = compute_accuracy(X_test, y_test, Ws, bs, gammas, betas)
    print("Test accuracy: $%f$ \\\\" % (test_acc))
    print("Best test accuracy: $%f$ \\\\" % (best_test_acc))
    print("Best valid accuracy: $%f$ \\\\" % (best_valid_acc))


    if save:
        # gradient spread
        # fig, axes = plt.subplots(n_layers, 1, sharex=True, sharey=True)
        # for layer in range(n_layers):
        #     data = [np.exp(g[layer].reshape(
        #         g[layer].shape[0] * g[layer].shape[1])) for g in grads]
        #     axes[layer].boxplot(data, 0, '', showfliers=False)
        #     axes[layer].set_title("Distribution of layer %d" % (layer + 1))
        # plt.xlabel('update step')
        # plt.ylabel('exp grad')
        # plt.show()
        # training and validation cost
        plt.plot(np.arange(len(costs_train))*n_saves,
                 costs_train, 'g', label='training loss')
        plt.plot(np.arange(len(costs_valid))*n_saves,
                 costs_valid, 'r', label='validation loss')
        plt.xlabel("update step")
        plt.ylabel("cost")
        plt.legend()
        plt.show()
        # visulize_weights(Ws[0], ('Weights in layer %d' % 1))
