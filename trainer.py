import tensorflow as tf
from generator import build_generator
from losses import *
import numpy as np

batch_size = 8
z_dim = 1
t_dim = 1
z_input = tf.placeholder(tf.float32, shape=([batch_size, z_dim]))
t_input = tf.placeholder(tf.float32, shape=([batch_size, t_dim]))
x_output = build_generator(z_input, t_input)

top_height = 0.8
loss = {}
loss['height'] = get_loss_at_top(x_output, top_height)
loss['bottom'] = get_loss_flat_bottom(x_output)
loss['area'] = get_loss_area(x_output)
loss['curvature'] = get_loss_curvature_vs_length(x_output)
loss['stability'], center_gravity = get_loss_stability(x_output)
num_loss = len(loss)
w_input = tf.placeholder(tf.float32, shape=([num_loss]))
total_loss = w_input[0]*loss['height'] \
             + w_input[1]*loss['bottom']\
             + w_input[2]*loss['area']\
             + w_input[3]*loss['curvature']\
             + w_input[4] * loss['stability']

opt = tf.train.AdamOptimizer(0.001)
grads_g = opt.compute_gradients(total_loss)
apply_gradient_op = opt.apply_gradients(grads_g)

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

total_loss_val_hist = []
loss_curvature_val_hist = []
loss_height_val_hist = []
loss_bottom_val_hist = []
loss_area_val_hist = []
loss_stability_val_hist = []
center_gravity_val_hist = []
x_output_val_hist = []
t_input_val = np.linspace(0., 1., batch_size)
max_epoch = 100000
cnt = 0
z_input_val = np.tile(np.random.randn(1, 1),(batch_size,1))
for ep_i in np.linspace(0., 1., max_epoch):
    cnt += 1
    if cnt%4000 == 0:
        z_input_val = np.random.randn(batch_size, 1)
    np.asarray(16 * [[0., ]], dtype='float32')
    if cnt/1000 < 0.1:
        w_input_val = [3, 10, 3, 0., 0.]  # height, bottom, area, curvature, stability
    elif cnt/1000 < 0.4:
        w_input_val = [3, 10, 3, 1.5, 0.]  # height, bottom, area, curvature, stability
    elif cnt/1000 < 0.8:
        w_input_val = [3, 10, 3, 1.5, 0.5]  # height, bottom, area, curvature, stability
    else:
        w_input_val = [3, 10, 3, 1.5, 0.5]  # height, bottom, area, curvature, stability
    feed_dict = {z_input: z_input_val,
                 t_input: t_input_val.reshape(batch_size,1),
                 w_input: w_input_val}
    x_output_val, loss_val, _ = sess.run([x_output, total_loss, apply_gradient_op], feed_dict)
    total_loss_val_hist += [loss_val]
    x_output_val_hist += [x_output_val]

    loss_curvature_val, loss_height_val, loss_bottom_val, loss_area_val, loss_stability_val, center_gravity_val\
        = sess.run([loss['curvature'],loss['height'],loss['bottom'],loss['area'],loss['stability'], center_gravity], feed_dict)
    loss_curvature_val_hist += [loss_curvature_val]
    loss_height_val_hist += [loss_height_val]
    loss_bottom_val_hist += [loss_bottom_val]
    loss_area_val_hist += [loss_area_val]
    loss_stability_val_hist += [loss_stability_val]
    center_gravity_val_hist += [center_gravity_val]

print('done')


import matplotlib.pyplot as plt
import seaborn
plt.figure()
idx = -1
plt.plot(np.asarray(x_output_val_hist)[idx, :, 0], np.asarray(x_output_val_hist)[idx, :, 1],linewidth=0.5)
plt.scatter(np.asarray(x_output_val_hist)[idx, 0, 0], np.asarray(x_output_val_hist)[idx, 0, 1])
plt.scatter(np.asarray(x_output_val_hist)[idx, -1, 0], np.asarray(x_output_val_hist)[idx, -1, 1])
axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([0,1])
#################
fig, ax1 = plt.subplots()
ax1.plot(loss_bottom_val_hist,'b')
ax1.plot(loss_area_val_hist,'g')
ax1.plot(loss_height_val_hist,'r')
ax1.set_ylabel('bottom_area_height', color='r')
ax1.tick_params('y', colors='r')
ax1.grid('off')
ax2 = ax1.twinx()
ax2.plot(loss_curvature_val_hist,'k')
ax2.plot(loss_stability_val_hist,'y')
ax2.set_ylabel('curvature_stability', color='y')
ax2.tick_params('y', colors='y')
ax2.grid('off')