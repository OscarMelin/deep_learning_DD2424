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
    # Therefore we need to use the diagonal of np.matmul(Y.trasnpose(), p)
    # py = np.diag(py).reshape(1, n) # (1, n)
    # Or use the elementwise mult and then sum
    py = np.multiply(Y, p)  # (10, n)
    py = np.sum(py, axis=0).reshape(1, n) # (1, n)
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
    
    grad_b = (1/n) * np.matmul(G_batch, np.ones((n, 1)))

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

def train_model(X, Y, _lambda, n_batch, eta, n_epochs, X_valid, Y_valid, X_test, y_test, save=False):
    xavier = 1/np.sqrt(d)

    W = np.random.normal(0, xavier, size=(K, d))
    b = np.random.normal(0, xavier, size=(K, 1))

    costs_train = np.zeros(n_epochs)
    costs_valid = np.zeros(n_epochs)

    for epoch_i in range(n_epochs):
        shuffle(X, Y)
        for X_batch, Y_batch in get_batches(n_batch, X, Y):
            P = evaluate_classifier(X_batch, W, b)
            grad_W, grad_b = compute_gradients(X_batch, Y_batch, P, W, _lambda)

            W = W - (eta * grad_W)
            b = b - (eta * grad_b)
        # decay learning rate
        eta *= 0.9

        

        if save:
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
    # X_2, Y_2, y_2 = load_batch('data_batch_2')
    # # X_3, Y_3, y_3 = load_batch('data_batch_3')
    # X_4, Y_4, y_4 = load_batch('data_batch_4')
    X_5, Y_5, y_5 = load_batch('data_batch_5')
    # X = np.append(X, X_2, axis=1)
    # Y = np.append(Y, Y_2, axis=1)
    # y = np.append(y, y_2, axis=0)
    # X = np.append(X, X_3, axis=1)
    # Y = np.append(Y, Y_3, axis=1)
    # y = np.append(y, y_3, axis=0)
    # X = np.append(X, X_4, axis=1)
    # Y = np.append(Y, Y_4, axis=1)
    # y = np.append(y, y_4, axis=0)
    # X = np.append(X, X_5[:,:9000], axis=1)
    # Y = np.append(Y, Y_5[:,:9000], axis=1)
    # y = np.append(y, y_5[:9000], axis=0)
    # X_valid, Y_valid, y_valid = X_5[:,:9000], Y_5[:,:9000], y[:9000]
    X_valid, Y_valid, y_valid = X_5, Y_5, y
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

    train_model(X, Y, _lambda, n_batch, eta, n_epochs, X_valid, Y_valid, X_test, y_test, save=True)
    
