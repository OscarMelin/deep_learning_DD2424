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
    # Therefore we need to use the diagonal
    py = np.matmul(Y.transpose(), p)  # (n, n)
    py = np.diag(py).reshape(1, n) # (1, n)
    cross_entropy = -np.log(py) # (1, n)
    regulation = _lambda * np.sum(W**2) # scalar
    J = (1/n) * np.sum(cross_entropy) + regulation # scalar
    return J

def predict(p):
    return np.argmax(p, axis=0).reshape(1, p.shape[1])

def compute_accuracy(X, y, W, b):
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
    
    # grad_b = (1/n) * np.sum(G_batch, axis=1) #.reshape(10, 1) # from (10,) to (10, 1)
    ones = np.ones((n, 1))

    grad_b = (1/n) * np.matmul(G_batch, np.ones((n, 1)))  #.reshape(10, 1) # from (10,) to (10, 1)
    return grad_W, grad_b


if __name__ == '__main__':
    X, Y, y = load_batch('data_batch_1')
    # visulize_5(X)
    K = 10
    n_tot = 10000
    d = 3072

    _lambda = 0
    n_batch = 100
    eta = 0.01
    n_epochs = 40

    # np.random.seed(400)
    h = 1e-6

    W = np.random.normal(0, 0.1, size=(K, d))
    b = np.random.normal(0, 0.1, size=(K, 1))

    # mini_batch_X, mini_batch_Y = X[:,:n_batch], Y[:,:n_batch]
    # cost = compute_cost(mini_batch_X, mini_batch_Y, W, b, _lambda)
    # P = evaluate_classifier(mini_batch_X, W, b)
    # grad_W, grad_b = compute_gradients(mini_batch_X, mini_batch_Y, P, W, _lambda)
    # num_grad_W, num_grad_b = compute_grads_num(mini_batch_X, mini_batch_Y, W, b, _lambda, h, compute_cost)
    # comp_W = compare_gradients(grad_W, num_grad_W)
    # print(comp_W)
    # comp_b = compare_gradients(grad_b, num_grad_b)
    # print(comp_b)

    for epoch_i in range(n_epochs):
        shuffle(X, Y)
        for X_batch, Y_batch in get_batches(n_batch, X, Y):
            P = evaluate_classifier(X_batch, W, b)
            grad_W, grad_b = compute_gradients(X_batch, Y_batch, P, W, _lambda)
            W = W - (eta * grad_W)
            b = b - (eta * grad_b)
        
        print()
        print(".... Epoch %d completed ...." % (epoch_i))
        # print("Current cost: %f" % (compute_cost(X, Y, W, b, _lambda)))
        print()
    
    acc = compute_accuracy(X, y, W, b)
    print("The final accuracy of the model after %d epochs is: %f" % (n_epochs, acc))
    print("Current cost: %f" % (compute_cost(X, Y, W, b, _lambda)))

    visulize_weights(W)