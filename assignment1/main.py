import pickle
import matplotlib.pyplot as plt
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def make_one_hot(y):
    """ create one-hot column vectors """
    one_hot = np.zeros((len(y), 10))
    for i in range(len(y)):
        one_hot[i, y[i]] = 1.
    return one_hot.transpose(1,0)

def load_batch(batch_name):
    """  
    X = (3072, 10000) each column is an image
    Y = (10, 10000) each colum is a one hot vector
    y = labels for each column
    """
    data_dict = unpickle('./datasets/cifar-10-batches-py/' + batch_name)
    X = data_dict[b'data'] / 255
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).reshape(10000, 3072).transpose(1,0)
    y = data_dict[b'labels']
    Y = make_one_hot(y)
    return X, Y, y

def visulize_5(X):
    """ Show 5x5 images from X
    """
    fig, axes1 = plt.subplots(5,5,figsize=(3,3))
    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(X)))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(X[:,i].reshape(32, 32, 3))
    plt.show()

def softmax(s):
    """Compute softmax values for each sets of scores in s"""
    exps = np.exp(s)
    return exps / np.sum(exps, axis=0)

def evaluate_classifier(X, W, b):
    """
    W = (10,3072)
    X = (3072, n)
    p = (10, n)
    """
    n = X.shape[1]
    WX = np.matmul(W, X)
    b_big = np.repeat(b, n, axis=1) # repeat column vector b, n times
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

    py = np.matmul(Y.transpose(), p)  # (n, n)
    cross_entropy = -np.log(py) # (n, n)
    regulation = _lambda * np.sum(W**2) # scalar
    J = np.sum(cross_entropy) / n + regulation # scalar
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
    

if __name__ == '__main__':
    X, Y, y = load_batch('data_batch_1')
    # visulize_5(X)
    K = 10
    n_tot = 10000
    d = 3072

    W = np.random.normal(0, 0.1, size=(K, d))
    b = np.random.normal(0, 0.1, size=(K, 1))

    # cost = compute_cost(X, Y, W, b, 0.1)
    compute_accuracy(X, y, W, b)
