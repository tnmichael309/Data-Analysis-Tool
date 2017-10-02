import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(tf.float32, shape=(n_x, None), name='X')
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name='Y')
    ### END CODE HERE ###
    
    return X, Y

def initialize_parameters( n_feature, n_output,
    layer_count = 3, hidder_neuron = [25, 12]):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 6 lines of code)
    W = []
    b = []

    for i in range(layer_count):
        weight_str = 'W'+str(i+1)
        bias_str = 'b'+str(i+1)

        if i == 0:
            weight = tf.get_variable(weight_str, [hidder_neuron[i], n_feature], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            bias = tf.get_variable(bias_str, [hidder_neuron[i], 1], initializer = tf.zeros_initializer())
        elif i == layer_count - 1:
            weight = tf.get_variable(weight_str, [n_output, hidder_neuron[i-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            bias = tf.get_variable(bias_str, [n_output, 1], initializer = tf.zeros_initializer())
        else:
            weight = tf.get_variable(weight_str, [hidder_neuron[i], hidder_neuron[i-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            bias = tf.get_variable(bias_str, [hidder_neuron[i], 1], initializer = tf.zeros_initializer())
            
        W.append(weight)
        b.append(bias)

    #W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    #b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
    #W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    #b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
    #W3 = tf.get_variable("W3", [6,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    #b3 = tf.get_variable("b3", [6,1], initializer = tf.zeros_initializer())
    ### END CODE HERE ###

    #parameters = {"W1": W1,
    #              "b1": b1,
    #              "W2": W2,
    #              "b2": b2,
    #              "W3": W3,
    #              "b3": b3}

    parameters = {
        'W': W,
        'b': b
    }

    return parameters
        
def forward_propagation(X, parameters, is_training=True):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    '''
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.matmul(W1, X) + b1                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2                                              # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3                                              # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###
    '''

    W = parameters['W']
    b = parameters['b']

    outputs = None
    activation = None

    for i in range(len(W)):
        weight = W[i]
        bias = b[i]
        if i == 0:
            output = tf.matmul(weight, X) + bias
            #output.set_shape([None, len(weight)])
            #output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training, scope='bn')
            activation = tf.nn.relu(output)
        elif i == len(W)-1:
            output = tf.matmul(weight, activation) + bias
        else:
            output = tf.matmul(weight, activation) + bias
            #output.set_shape([None, len(weight)])
            #output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training, scope='bn')
            activation = tf.nn.relu(output)

    return output

def compute_cost(parameters, output, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    '''
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    '''
    ### START CODE HERE ### (1 line of code)
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, output))))
    for w in parameters['W']:
        loss = loss + 0.01*tf.nn.l2_loss(w)
    ### END CODE HERE ###
    
    return loss
    
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X.iloc[:, permutation]
    shuffled_Y = Y[:,permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X.iloc[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X.iloc[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
    
def model(X_train, Y_train, learning_rate = 0.01, is_training=True,
          num_epochs = 1500, minibatch_size = 32, print_cost = True, layer_count = 3, hidder_neuron = [25, 12]):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_x, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    # n_feature, n_output,
    # layer_count = 3, hidder_neuron = [25, 12]
    parameters = initialize_parameters(n_x, n_y, layer_count = layer_count, hidder_neuron = hidder_neuron)
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    output = forward_propagation(X, parameters, is_training)
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(parameters, output, Y)
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    global_step = tf.Variable(0, trainable=False)
    online_learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                           10000, 0.95, staircase=True)
    
    # Note: when training, the moving_mean and moving_variance need to be updated. 
    # By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, 
    # so they need to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):                                       
        optimizer = tf.train.AdamOptimizer(learning_rate = online_learning_rate).minimize(cost, global_step=global_step)
    
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0 and epoch > 300:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        # parameters = sess.run(parameters)
        # print ("Parameters have been trained!")

        # Calculate the correct predictions
        rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, output))))

        # Calculate accuracy on the test set
        rmse = tf.cast(rmse, "float")

        print ("Train RMSE:", rmse.eval({X: X_train, Y: Y_train}))
        #print ("Test RMSE:", rmse.eval({X: X_test, Y: Y_test}))
        
        #return parameters