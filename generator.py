import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def new_variables(z_dim, t_dim, X_dim = 2):
    h_dim_1,h_dim_2,h_dim_3,h_dim_4,h_dim_5=10,10,10,10,10


    W1 = tf.Variable(xavier_init([z_dim+t_dim, h_dim_1]))
    b1 = tf.Variable(tf.zeros(shape=[h_dim_1]))

    W2 = tf.Variable(xavier_init([h_dim_1, h_dim_2]))
    b2 = tf.Variable(tf.zeros(shape=[h_dim_2]))

    W3 = tf.Variable(xavier_init([h_dim_2, h_dim_3]))
    b3 = tf.Variable(tf.zeros(shape=[h_dim_3]))

    W4 = tf.Variable(xavier_init([h_dim_3, h_dim_4]))
    b4 = tf.Variable(tf.zeros(shape=[h_dim_4]))

    W5 = tf.Variable(xavier_init([h_dim_4, h_dim_5]))
    b5 = tf.Variable(tf.zeros(shape=[h_dim_5]))

    W6 = tf.Variable(xavier_init([h_dim_5, X_dim]))
    b6 = tf.Variable(tf.zeros(shape=[X_dim]))

    WB = {}
    WB['W1'], WB['W2'], WB['W3'], WB['W4'], WB['W5'], WB['W6'] = W1, W2, W3, W4, W5, W6
    WB['b1'], WB['b2'], WB['b3'], WB['b4'], WB['b5'], WB['b6'] = b1, b2, b3, b4, b5, b6
    return WB

def build_generator_new(z_input,t_input, WB):

    W1, W2, W3, W4, W5, W6 = WB['W1'], WB['W2'], WB['W3'], WB['W4'], WB['W5'], WB['W6']
    b1, b2, b3, b4, b5, b6 = WB['b1'], WB['b2'], WB['b3'], WB['b4'], WB['b5'], WB['b6']

    h1 = tf.nn.relu(tf.matmul(tf.concat([z_input,t_input],1), W1) + b1)
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
    h4 = tf.nn.relu(tf.matmul(h3, W4) + b4)
    h5 = tf.nn.relu(tf.matmul(h4, W5) + b5)
    x_output = tf.nn.sigmoid(tf.matmul(h5, W6) + b6)

    return x_output

def build_generator_new_new(z_input,t_input, WB):
    '''x,y position and state'''

    W1, W2, W3, W4, W5, W6 = WB['W1'], WB['W2'], WB['W3'], WB['W4'], WB['W5'], WB['W6']
    b1, b2, b3, b4, b5, b6 = WB['b1'], WB['b2'], WB['b3'], WB['b4'], WB['b5'], WB['b6']

    h1 = tf.nn.relu(tf.matmul(tf.concat([z_input,t_input],1), W1) + b1)
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
    h4 = tf.nn.relu(tf.matmul(h3, W4) + b4)
    h5 = tf.nn.relu(tf.matmul(h4, W5) + b5)
    x_output = tf.nn.sigmoid(tf.matmul(h5, W6) + b6)[:,:2]
    open_flg = tf.nn.sigmoid(tf.matmul(h5, W6) + b6)[:,2:]

    return tf.concat([x_output, open_flg],1)

def build_generator(z_input,t_input):
    h_dim_1,h_dim_2,h_dim_3,h_dim_4,h_dim_5=10,10,10,10,10
    X_dim = 2
    r=4
    z_dim = z_input.get_shape().as_list()[1]
    t_dim = t_input.get_shape().as_list()[1]

    W1 = tf.Variable(xavier_init([z_dim+t_dim, h_dim_1]))
    b1 = tf.Variable(tf.zeros(shape=[h_dim_1]))

    W2 = tf.Variable(xavier_init([h_dim_1, h_dim_2]))
    b2 = tf.Variable(tf.zeros(shape=[h_dim_2]))

    W3 = tf.Variable(xavier_init([h_dim_2, h_dim_3]))
    b3 = tf.Variable(tf.zeros(shape=[h_dim_3]))

    W4 = tf.Variable(xavier_init([h_dim_3, h_dim_4]))
    b4 = tf.Variable(tf.zeros(shape=[h_dim_4]))

    W5 = tf.Variable(xavier_init([h_dim_4, h_dim_5]))
    b5 = tf.Variable(tf.zeros(shape=[h_dim_5]))

    W6 = tf.Variable(xavier_init([h_dim_5, X_dim]))
    b6 = tf.Variable(tf.zeros(shape=[X_dim]))

    # W1=tf.Variable(sio.loadmat('weight/W1_initial.mat')['W'])
    # W2=tf.Variable(sio.loadmat('weight/W2_initial.mat')['W'])
    # W3=tf.Variable(sio.loadmat('weight/W3_initial.mat')['W'])
    # W4=tf.Variable(sio.loadmat('weight/W4_initial.mat')['W'])
    # W5=tf.Variable(sio.loadmat('weight/W5_initial.mat')['W'])
    # W6=tf.Variable(sio.loadmat('weight/W6_initial.mat')['W'])

    # b1=tf.Variable(sio.loadmat('weight/b1_initial.mat')['b'])
    # b2=tf.Variable(sio.loadmat('weight/b2_initial.mat')['b'])
    # b3=tf.Variable(sio.loadmat('weight/b3_initial.mat')['b'])
    # b4=tf.Variable(sio.loadmat('weight/b4_initial.mat')['b'])
    # b5=tf.Variable(sio.loadmat('weight/b5_initial.mat')['b'])
    # b6=tf.Variable(sio.loadmat('weight/b6_initial.mat')['b'])

    h1 = tf.nn.relu(tf.matmul(tf.concat([z_input,t_input],1), W1) + b1)
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
    h4 = tf.nn.relu(tf.matmul(h3, W4) + b4)
    h5 = tf.nn.relu(tf.matmul(h4, W5) + b5)
    x_output = tf.nn.sigmoid(tf.matmul(h5, W6) + b6)

    return x_output



if __name__ == '__main__':

    batch_size=16
    z_dim = 1
    t_dim = 1
    z_input = tf.placeholder(tf.float32, shape=([batch_size, z_dim]))
    t_input = tf.placeholder(tf.float32, shape=([batch_size, t_dim]))
    x_output = build_generator(z_input,t_input)

    ## training starts ###
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)

    x_output_val = []
    for t in np.linspace(0.,1.,100):
        feed_dict = {z_input: np.asarray(16*[[0.,]],dtype='float32'), t_input: np.asarray(16*[[t,]],dtype='float32')}
        x_output_val += [sess.run(x_output, feed_dict)]



    import matplotlib.pyplot as plt
    plt.plot(np.asarray(x_output_val)[:, 0, 0], np.asarray(x_output_val)[:, 0, 1], 'b','.-')
    print('done')