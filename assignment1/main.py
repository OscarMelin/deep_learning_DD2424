import numpy as np
from helpers import *

def softmax(s):
    """Compute softmax values for each sets of scores in s"""
    exps = np.exp(s) # (10, n)
    ones = np.ones((1, s.shape[0])) # (1, 10)
    denom = np.matmul(ones, np.exp(s)) # (1, n)
    p = exps / denom # (10, n)
    return p

def evaluate_classifier(X, W, b):
    """
    W = (10,3072)
    X = (3072, n)
    p = (10, n)
    """
    n = X.shape[1]
    WX = np.matmul(W, X)
    # b_big = np.repeat(b, n, axis=1) # repeat column vector b, n times
    b_big = np.matmul(b, np.ones((n, 1)).transpose())
    s = WX + b_big
    p = softmax(s)
    return p

def evaluate_classifier_SVM(X, W, b):
    """
    W = (10,3072)
    X = (3072, n)
    s = (10, n)
    """
    n = X.shape[1]
    WX = np.matmul(W, X)
    b_big = np.matmul(b, np.ones((n, 1)).transpose())
    s = WX + b_big
    return s

def compute_cost(X, Y, W, b, _lambda):
    """
    W = (10,3072)
    X = (3072, n)
    p = (10, n)
    """
    n = X.shape[1]
    p = evaluate_classifier(X, W, b) #
    # cross entropy for one x and one one hot y column vector
    # is -log(y^T * p) which is basically value of p[true_label]
    # Therefore we need to use the diagonal of np.matmul(Y.trasnpose(), p)
    # py = np.diag(py).reshape(1, n) # (1, n)
    # Or use the elementwise mult and then sum
    py = np.multiply(Y, p)  # (10, n)
    py = np.sum(py, axis=0).reshape(1, n) # (1, n)
    cross_entropy = -np.log(py) # (1, n)
    regulation = _lambda * np.sum(W**2) # scalar
    J = (1/n) * np.sum(cross_entropy) + regulation # scalar
    return J

def compute_cost_SVM(X, Y, W, b, _lambda):
    """
    W = (10,3072)
    X = (3072, n)
    """
    n = X.shape[1]
    s = evaluate_classifier_SVM(X, W, b)

    sy = np.sum(np.multiply(s, Y), axis=0).reshape(1, n).repeat(10, axis=0) # (10, n)

    margins = np.maximum(0, s - sy + 1) # (10, n)
    margins[np.argmax(Y, axis=0), np.arange(n)] = 0 # where j == y, gets 0
    hinge_loss = (1/n) * np.sum(margins)# scalar. 
    regulation = _lambda * np.sum(W**2) # scalar
    J = hinge_loss + regulation
    return J, margins

def predict(p):
    return np.argmax(p, axis=0).reshape(1, p.shape[1])

def compute_accuracy(X, y, W, b, loss='cross-entropy'):
    if loss == 'SVM':
        p = evaluate_classifier_SVM(X, W, b)
    else:
        p = evaluate_classifier(X, W, b)
    predicted = predict(p)

    zero_one_loss = 0
    for i in range(X.shape[1]):
        if predicted[0, i] != y[i]:
            zero_one_loss += 1
    
    accuracy = 1 - zero_one_loss / X.shape[1]
    return accuracy

def compute_gradients(X, Y, P, W, _lambda):
    """
    X = (3072, n)
    Y = (10, n)
    P = (10, n)
    grad_W = (10, 3072)
    grad_b = (10, 1)
    """
    # From lec3 slides 
    n = X.shape[1]
    G_batch = -(Y - P) # (10, n)

    grad_W = (1/n) * np.matmul(G_batch, X.transpose())

    # Regulation term
    grad_W += 2 * _lambda * W
    
    grad_b = (1/n) * np.matmul(G_batch, np.ones((n, 1)))

    return grad_W, grad_b

def compute_gradients_SVM(X, Y, margins, W, _lambda):
    """
    X = (3072, n)
    Y = (10, n)
    margins = (10, n)
    y(Wx + b) < 1 <=> margins > 0
    because we have aleady taken
    max(0, 1 - y(Wx + b))
    grad_W = (10, 3072)
    grad_b = (10, 1)
    """
    # https://stats.stackexchange.com/questions/155088/gradient-for-hinge-loss-multiclass
    # Slide 69 on lec 2
    n = X.shape[1]
    binary = margins
    binary[margins > 0] = 1
    # Count the number of classes that did not meet margin
    count = np.sum(binary, axis=0)
    # for each x_i
    # where j == y, grad_W = -count * x_i
    # else grad_W = x_i or 0 if binary == 0
    binary[np.argmax(Y, axis=0), np.arange(n)] = -count
    grad_W = np.matmul(X, binary.T).T / n 
    # Regulation term
    grad_W += _lambda * W

    # According to slides 69 lec 2 just -y 
    grad_b = np.sum(binary, axis=1).reshape(10, 1) / n
    return grad_W, grad_b

def check_gradients(X,Y,W,b,_lambda):
    n_batch = 1
    mini_batch_X, mini_batch_Y = X[:,:n_batch], Y[:,:n_batch]
    cost = compute_cost(mini_batch_X, mini_batch_Y, W, b, _lambda)
    P = evaluate_classifier(mini_batch_X, W, b)
    grad_W, grad_b = compute_gradients(mini_batch_X, mini_batch_Y, P, W, _lambda)
    num_grad_W, num_grad_b = compute_grads_num_slow(mini_batch_X, mini_batch_Y, W, b, _lambda, h, compute_cost)
    comp_W = compare_gradients(grad_W, num_grad_W)
    print("W relative error: ")
    print(comp_W)
    comp_b = compare_gradients(grad_b, num_grad_b)
    print("b relative error:")
    print(comp_b)

def grid_search(X, Y, X_test, y_test):
    _lambdas = [0, 0.001, 0.0001]
    n_batches = [64,100, 128]
    etas = [0.01, 0.02, 0.03]

    accs = []
    vals = []
    for _lambda in _lambdas:
        for n_batch in n_batches:
            for eta in etas:
                acc = train_model(X, Y, _lambda, n_batch, eta, 40, [], [], X_test, y_test, save=False)
                accs.append(acc)
                vals.append("lambda: %f, n_batch: %d and eta: %f" % (_lambda, n_batch, eta))
                print("one done")

    max_i = np.argmax(accs)
    print(max_i)
    print(accs)
    print(vals[max_i])

def train_model(X, Y, _lambda, n_batch, eta, n_epochs, X_valid, Y_valid, X_test, y_test, save=False, loss='cross-entropy'):
    xavier = 1/np.sqrt(d)

    W = np.random.normal(0, xavier, size=(K, d))
    b = np.random.normal(0, xavier, size=(K, 1))

    costs_train = np.zeros(n_epochs)
    costs_valid = np.zeros(n_epochs)

    for epoch_i in range(n_epochs):
        shuffle(X, Y)
        for X_batch, Y_batch in get_batches(n_batch, X, Y):
            if loss == 'SVM':
                cost, margins = compute_cost_SVM(X_batch, Y_batch, W, b, _lambda)
                grad_W, grad_b = compute_gradients_SVM(X_batch, Y_batch, margins, W, _lambda)
            else:
                P = evaluate_classifier(X_batch, W, b)
                grad_W, grad_b = compute_gradients(X_batch, Y_batch, P, W, _lambda)

            W = W - (eta * grad_W)
            b = b - (eta * grad_b)
        # decay learning rate
        eta *= 0.9

        if save:
            if loss == 'SVM':
                costs_train[epoch_i] = compute_cost_SVM(X, Y, W, b, _lambda)[0]
                costs_valid[epoch_i] = compute_cost_SVM(X_valid, Y_valid, W, b, _lambda)[0]
            else:
                costs_train[epoch_i] = compute_cost(X, Y, W, b, _lambda)
                costs_valid[epoch_i] = compute_cost(X_valid, Y_valid, W, b, _lambda)
            print()
            print(".... Epoch %d completed ...." % (epoch_i))
            print()
    
    acc = compute_accuracy(X_test, y_test, W, b)
    print("The accuracy of the model: $%f$ \\\\" % ( acc))

    if save:
        plt.plot(np.arange(n_epochs), costs_train, 'g', label='training loss')
        plt.plot(np.arange(n_epochs), costs_valid, 'r', label='validation loss')
        plt.legend()
        plt.show()
        visulize_weights(W)
    return acc

if __name__ == '__main__':
    X, Y, y = load_batch('data_batch_1')
    X_valid, Y_valid, y_valid = load_batch('data_batch_2')
    X_test, Y_test, y_test = load_batch('test_batch')
    # visulize_5(X)
    K = 10
    n_tot = 10000
    d = 3072

    # grid_search(X, Y, X_test, y_test)
    # exit()

    _lambda = 0.0001
    n_batch = 64
    eta = 0.02
    n_epochs = 40

    h = 1e-6

    train_model(X, Y, _lambda, n_batch, eta, n_epochs, X_valid, Y_valid, X_test, y_test, save=True, loss='SVM')
    
