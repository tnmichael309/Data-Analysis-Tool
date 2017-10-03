import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from IPython.display import clear_output, Image, display, HTML

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
    
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

    X = tf.placeholder(tf.float32, shape=(n_x, None), name='X')
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name='Y')
    phase = tf.placeholder(tf.bool, name='phase')
    
    return X, Y, phase

def initialize_parameters( n_feature, n_output,
    layer_count = 3, hidden_neuron = [25, 12]):
       
    W = []
    b = []

    for i in range(layer_count):
        weight_str = 'W'+str(i+1)
        bias_str = 'b'+str(i+1)

        if i == 0:
            weight = tf.get_variable(weight_str, [hidden_neuron[i], n_feature], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            bias = tf.get_variable(bias_str, [hidden_neuron[i], 1], initializer = tf.zeros_initializer())
        elif i == layer_count - 1:
            weight = tf.get_variable(weight_str, [n_output, hidden_neuron[i-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            bias = tf.get_variable(bias_str, [n_output, 1], initializer = tf.zeros_initializer())
        else:
            weight = tf.get_variable(weight_str, [hidden_neuron[i], hidden_neuron[i-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
            bias = tf.get_variable(bias_str, [hidden_neuron[i], 1], initializer = tf.zeros_initializer())
            
        W.append(weight)
        b.append(bias)

    parameters = {
        'W': W,
        'b': b
    }

    return parameters

def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
    
def forward_propagation(X, parameters, is_training):

    W = parameters['W']
    b = parameters['b']
    
    for i in range(len(W)):
        weight = W[i]
        bias = b[i]
        scope = 'layer' + str(i+1)
        with tf.variable_scope(scope):
            if i == 0:
                output = tf.matmul(weight, X) + bias
                output = tf.transpose(output, name='transpose_to')
                output = batch_norm(output, output.get_shape()[1], is_training)
                output = tf.transpose(output, name='transpose_back')
                activation = tf.nn.relu(output)
            elif i == len(W)-1:
                output = tf.matmul(weight, activation) + bias
            else:
                output = tf.matmul(weight, activation) + bias
                output = tf.transpose(output, name='transpose_to')
                output = batch_norm(output, output.get_shape()[1], is_training)
                output = tf.transpose(output, name='transpose_back')
                activation = tf.nn.relu(output)

    return output

def compute_cost(parameters, output, Y):

    with tf.name_scope('loss'):
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, output))))
        for w in parameters['W']:
            loss = loss + 0.01*tf.nn.l2_loss(w)
        
        return loss
        
    raise ValueError('Cannot enter scope \"loss \"')
    
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
    
def model(X_train, Y_train, 
        X_valid, Y_valid,
        X_test,
        learning_rate = 0.01,
        num_epochs = 1500, 
        minibatch_size = 32, 
        print_cost = True, 
        layer_count = 3, 
        hidden_neuron = [25, 12]):
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
    X, Y, phase = create_placeholders(n_x, n_y)

    # Initialize parameters
    # n_feature, n_output,
    # layer_count = 3, hidder_neuron = [25, 12]
    parameters = initialize_parameters(n_x, n_y, layer_count = layer_count, hidden_neuron = hidden_neuron)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    output = forward_propagation(X, parameters, phase)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(parameters, output, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
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

    # show graph
    show_graph(tf.get_default_graph().as_graph_def())
    
    # Calculate the correct predictions
    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, output))))

    # Calculate accuracy on the test set
    rmse = tf.cast(rmse, "float")
    
    # Start the session to compute the tensorflow graph
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        
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
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, phase:True})
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                print ("Trainning set rmse:", rmse.eval({X: X_train, Y: Y_train, phase:False}))
                print ("Validation set rmse:", rmse.eval({X: X_valid, Y: Y_valid, phase:False}))
                
            if print_cost == True and epoch % 5 == 0 and epoch > 300:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        final_output = sess.run(output, feed_dict={X: X_test, phase: False})
        submission = pd.DataFrame({
            "Id": X_test["Id"],
            "SalePrice": final_output
        })
        submission.to_csv("submission.csv", encoding='utf-8', index=False)
        
        return parameters