import pickle
import matplotlib.pyplot as plt
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def shuffle(X, Y):
    index = np.arange(X.shape[1])
    np.random.shuffle(index)
    X = X[:,index]
    Y = Y[:,index]
    return X, Y

def get_batches(n_batch, X, Y):
    """ Return n_batch of the X 
    vector at a time
    """
    current_index = 0
    while current_index + n_batch <= X.shape[1]:
        X_batch = X[:,current_index:current_index + n_batch]
        Y_batch = Y[:,current_index:current_index + n_batch]
        current_index += n_batch
        yield X_batch, Y_batch

def compute_grads_num_slow(X, Y, W, b, _lambda, h, compute_cost):
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros_like(W)
    grad_b = np.zeros((no,1))

    for i in range(b.shape[0]):
        b_try = np.copy(b)
        b_try[i] = b_try[i] - h
        c1 = compute_cost(X, Y, W, b_try, _lambda)
        b_try = np.copy(b)
        b_try[i] = b_try[i] + h
        c2 = compute_cost(X, Y, W, b_try, _lambda)
        grad_b[i] = (c2-c1) / (2*h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.copy(W)
            W_try[i][j] = W_try[i][j] - h
            c1 = compute_cost(X, Y, W_try, b, _lambda)
            W_try = np.copy(W)
            W_try[i][j] = W_try[i][j] + h
            c2 = compute_cost(X, Y, W_try, b, _lambda)
            grad_W[i][j] = (c2-c1) / (2*h)
    
    return grad_W, grad_b

def compute_grads_num(X, Y, W, b, _lambda, h, compute_cost):
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros_like(W)
    grad_b = np.zeros((no,1))

    c = compute_cost(X, Y, W, b, _lambda)
    if len(c) > 0:
        c = c[0]

    for i in range(b.shape[0]):
        b_try = np.copy(b)
        b_try[i] = b_try[i] + h
        c2 = compute_cost(X, Y, W, b_try, _lambda)
        if len(c2) > 0:
            c2 = c2[0]
        grad_b[i] = (c2-c) / h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.copy(W)
            W_try[i][j] = W_try[i][j] + h
            c2 = compute_cost(X, Y, W_try, b, _lambda)
            if len(c2) > 0:
                c2 = c2[0]
            grad_W[i][j] = (c2-c) /h
    
    return grad_W, grad_b

def compare_gradients(grad, num_grad):
    nominator = np.sum(np.abs(grad - num_grad))
    demonimator =  max(1e-6, np.sum(np.abs(grad)) + np.sum(np.abs(num_grad)))
    return nominator / demonimator

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
    X = data_dict[b'data'] / 255 # between 0 and 1
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).reshape(10000, 3072).transpose(1,0) # get (rgb) not rrrgggbbb
    X_mean = np.mean(X, axis=1, keepdims=True)
    X = X - X_mean # Center with mean 0
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

def visulize_weights(W):
    """ Show all the weight vectors as pictures
    """
    fig, axes1 = plt.subplots(2,5,figsize=(3,3))
    i = 0
    for j in range(2):
        for k in range(5):
            im = W[i,:].reshape(32, 32, 3)
            im = (im - np.min(im)) / (np.max(im) - np.min(im))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(im)
            i += 1
    plt.show()