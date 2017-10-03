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

    X = tf.placeholder(tf.float32, shape=(None, n_x), name='X')
    Y = tf.placeholder(tf.float32, shape=(None, n_y), name='Y')
    phase = tf.placeholder(tf.bool, name='phase')
    
    return X, Y, phase

def dense(x, size, scope):
    return tf.contrib.layers.fully_connected(x, size, 
                                             activation_fn=None,
                                             scope=scope)

                                             
def dense_relu(x, size, scope):
    with tf.variable_scope(scope):
        h1 = dense(x, size, 'dense')
        return tf.nn.relu(h1, 'relu')

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
    
def dense_batch_relu(x, size, phase, scope):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, size, activation_fn=None, scope='dense')
        #h2 = batch_norm(h1, h1.get_shape()[1], phase)
        #h2 = tf.contrib.layers.batch_norm(h1, 
        #                                  center=True, scale=True, 
        #                                  is_training=phase,
        #                                  scope='bn')
            
        return tf.nn.relu(h1, 'relu')

def forward_propagation(X, phase, layer_count = 3, hidden_neuron = [25, 12]):

    for i in range(layer_count-1):
        layer_str = 'layer' + str(i+1)
        print(layer_str, ' initializing...')
        if i == 0:
            h1 = dense_batch_relu(X, hidden_neuron[i], phase, layer_str)
        else:
            h1 = dense_batch_relu(h1, hidden_neuron[i], phase, layer_str)

            
    output = dense(h1, 1, 'output')
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    
    #for v in tf.trainable_variables():
    #    if 'weights' in v.name:
    #        weights = tf.get_variable(
    #            name=v.name,
    #            regularizer=regularizer
    #        )
            #tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, v)
            
    #print(tf.trainable_variables())
    return output, regularizer

def compute_cost(output, Y, regularizer):

    #print(output.shape)
    #print(Y.shape)
    with tf.name_scope('loss'):
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, output))))
    #reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    #loss = loss + reg_term
    
        return loss
    
    raise ValueError('Cannot enter scope \"loss \"')
    
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X.iloc[permutation, :]
    shuffled_Y = Y[permutation,:].reshape((m, Y.shape[1]))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X.iloc[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X.iloc[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
    
def model(X_train, Y_train, learning_rate = 0.01, is_training=True,
          num_epochs = 1500, minibatch_size = 32, print_cost = True, layer_count = 3, hidden_neuron = [25, 12]):
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (m, n_x) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[1]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y, phase = create_placeholders(n_x, n_y)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    output,regularizer = forward_propagation(X, phase, layer_count, hidden_neuron)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(output, Y, regularizer)
    
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

    # Calculate the final cost
    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, output))))

    # Calculate accuracy on the test set
    rmse = tf.cast(rmse, "float")
    
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
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, phase: is_training})
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("\nCost after epoch %i: %f" % (epoch, epoch_cost))
                print ("Train set rmse:", rmse.eval({X: X_train, Y: Y_train, phase: False}))
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

        #print ("Test RMSE:", rmse.eval({X: X_test, Y: Y_test}))
        
        #return parameters