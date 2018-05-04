from rl.pg_reinforce import PolicyGradientREINFORCE
import tensorflow as tf
from collections import deque

from generator import build_generator_new_new, new_variables
from losses_new_new import *
import numpy as np

num_nodes = 16
batch_size = 1
z_dim = 1
t_dim = 1

def generator(z_input, act_function=tf.identity):
    # z_input = tf.placeholder(tf.float32, shape=([batch_size, z_dim]))
    t_input = np.tile(np.linspace(0., 1., num_nodes), (batch_size, 1))#tf.placeholder(tf.float32, shape=([batch_size, num_nodes*t_dim]))
    sig_w = 1.#tf.placeholder(tf.float32, shape=([]))

    WB = new_variables(z_dim, t_dim, X_dim=3)
    data_i = build_generator_new_new(z_input, t_input[:,0:1], WB, sig_w, act_function)
    data = tf.expand_dims(data_i,1)
    for i in range(1,num_nodes,1):
        data_i = build_generator_new_new(z_input, t_input[:,i:i+1], WB, sig_w, act_function)
        data = tf.concat([data,tf.expand_dims(data_i,1)],1)
    return tf.reshape(data,(batch_size,num_nodes*3))#x_output, open_flg

def get_total_loss(sess,data):
    state = tf.reshape(data,(batch_size,num_nodes,3))
    x_output = state[:, :, :2]
    open_flg = state[:, :, 2]
    top_height = 0.8
    loss = {}
    loss['range'] = tf.constant(0.)
    loss['height'] = tf.constant(0.)
    loss['bottom'] = tf.constant(0.)
    loss['area'] = tf.constant(0.)
    loss['open_flg'] = tf.constant(0.)
    # loss['edge'] = tf.constant(0.)
    loss['stability'] = tf.constant(0.)
    for i in range(batch_size):
        loss['height'] += get_loss_at_top(x_output[i], top_height) / batch_size
        loss['bottom'] += get_loss_flat_bottom(x_output[i]) / batch_size
        # sig_w = tf.placeholder(tf.float32, shape=([]))
        sig_w = 1.
        loss_area, area_list = get_loss_area(x_output[i], open_flg, sig_w)
        loss['area'] += loss_area / batch_size
        loss['range'] += get_loss_range(x_output[i], open_flg) / batch_size
        # loss['curvature'] += get_loss_curvature_vs_length(x_output[i]) /batch_size
        # loss['edge'] += get_loss_sharp_edge(x_output[i]) /batch_size
        loss_stability, center_gravity = get_loss_stability(x_output[i])
        loss['stability'] += loss_stability / batch_size
        loss['open_flg'] += 0.  # tf.reduce_mean(0.5-tf.abs(open_flg-0.5))

    # w_input = tf.placeholder(tf.float32, shape=([len(loss)]))
    w_input = [30, 10, 0, 0., 0., 0.]  # height, bottom, area, range, open, stability
    total_loss = w_input[0] * loss['height'] \
                 + w_input[1] * loss['bottom'] \
                 + w_input[2] * loss['area'] \
                 + w_input[3] * loss['range'] \
                 + w_input[5] * loss['stability']
    # + w_input[4] * loss['open_flg']\

    # w_input_val = [30, 10, 0, 0., 0., 0.]  # height, bottom, area, range, open, stability
    # sig_w_val = 1.
    # feed_dict = {w_input:w_input_val, sig_w:sig_w_val}
    return sess.run(total_loss)

z_val = tf.placeholder(tf.float32, shape=([batch_size, z_dim]))
data = generator(z_val, act_function=tf.sigmoid)
x_output = tf.reshape(data,(batch_size,num_nodes,3))
position, open_flg = x_output[:,:,:2], x_output[:,:,2]

def plot_cup(position, open_flg):
    idx = 0
    x_pos, y_pos = position[idx, :, 0], position[idx, :, 1]
    open_flg_val = open_flg[idx]
    import matplotlib.pyplot as plt
    import seaborn
    plt.close('all')
    plt.figure()
    # idx = -1
    bs_i = 0
    for i, open_flg_i in enumerate(open_flg_val):
        if open_flg_i < 0.5:
            plt.plot(x_pos, y_pos, 'r.', linewidth=0.5)
    plt.plot(x_pos, y_pos, linewidth=0.5)
    plt.scatter(x_pos[0], y_pos[0])
    axes = plt.gca()
    axes.set_xlim([-0.1, 1.1])
    axes.set_ylim([-0.1, 1.1])

## training starts ###
FLAGS = tf.app.flags.FLAGS
tfconfig = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True,
)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)

#### main
pg_reinforce = PolicyGradientREINFORCE(sess,
                                       optimizer,
                                       generator,
                                       z_dim,
                                       num_nodes*3,
                                       batch_size=batch_size)
MAX_EPISODES = 1000
MAX_STEPS    = 1
episode_history = deque(maxlen=MAX_EPISODES)
for i_episode in range(MAX_EPISODES):
  # initialize
  state = np.random.randn(batch_size, z_dim)
  total_rewards = 0
  for t in range(MAX_STEPS):
    action = pg_reinforce.sampleAction(state)
    reward = get_total_loss(sess, action)
    total_rewards = reward
    pg_reinforce.storeRollout(state, action, reward)
  pg_reinforce.updateModel()
  episode_history.append(total_rewards)
  mean_rewards = np.mean(episode_history)

print('done')
import matplotlib.pyplot as plt
import seaborn
plt.plot(episode_history)

feed_dict = {z_val: np.random.randn(batch_size, z_dim)}
position_val, open_flg_val = sess.run([position, open_flg], feed_dict)
plot_cup(position_val, open_flg_val)