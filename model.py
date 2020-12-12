import numpy as np

# def initialize_parameters(n_x, n_h, n_y):
#     """
#     Argument:
#     n_x -- size of the input layer
#     n_h -- size of the hidden layer
#     n_y -- size of the output layer
    
#     Returns:
#     parameters -- python dictionary containing your parameters:
#                     W1 -- weight matrix of shape (n_h, n_x)
#                     b1 -- bias vector of shape (n_h, 1)
#                     W2 -- weight matrix of shape (n_y, n_h)
#                     b2 -- bias vector of shape (n_y, 1)
#     """
#     np.random.seed(1)
#     W1 = np.random.randn(n_h, n_x) * 0.01
#     b1 = np.zeros(shape=(n_h, 1))
#     W2 = np.random.randn(n_y, n_h) * 0.01
#     b2 = np.zeros(shape=(n_y, 1))

#     assert(W1.shape == (n_h, n_x))
#     assert(b1.shape == (n_h, 1))
#     assert(W2.shape == (n_y, n_h))
#     assert(b2.shape == (n_y, 1))
    
#     parameters = {"W1": W1,
#                   "b1": b1,
#                   "W2": W2,
#                   "b2": b2}
    
#     return parameters

def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

parameters = initialize_parameters([2,2,1])

def forward(X, parameters):

    return 1

def cost(AL, y):

    return 1

def backward(AL, y, caches):

    return 1