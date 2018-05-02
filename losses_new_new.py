import tensorflow as tf


def get_loss_at_top(input_tensor, hight):
    loss_left = (hight - input_tensor[0,1])*(hight - input_tensor[0,1])
    loss_right = (hight - input_tensor[-1,1])*(hight - input_tensor[-1,1])
    # loss_mid = tf.nn.relu(-input_tensor[0,0] + 0.5) + tf.nn.relu(-0.5 + input_tensor[-1,0])
    return loss_left + loss_right #+ loss_mid

def get_loss_curvature(input_tensor):
    v1 = input_tensor[1:-1] - input_tensor[:-2]
    v2 = input_tensor[2:] - input_tensor[1:-1]
    cos_dist = tf.reduce_sum(v1*v2,1) / tf.norm(v1,axis=1) / tf.norm(v2,axis=1)
    loss_curvature = tf.reduce_max(tf.square(1-cos_dist))
    return loss_curvature

def get_loss_flat_bottom(input_tensor):
    length = input_tensor.get_shape().as_list()[0]
    bottom_y = input_tensor[int(length * 0.35):int(length * 0.65), 1]
    loss_flat_bottom = tf.reduce_max( tf.abs(bottom_y - 0.2))
    loss_bottom = tf.reduce_max(tf.nn.relu( 0.2 - input_tensor[:,1]))
    return loss_flat_bottom + loss_bottom

def get_loss_area(input_tensor):
    bs = input_tensor.get_shape().as_list()[0]
    import numpy as np
    center = np.asarray(bs * [[0.5, 0.8]])
    padding = tf.zeros((bs,1))
    v = tf.concat([center,padding],1) - tf.concat([input_tensor,padding],1)
    area = tf.reduce_sum(tf.cross(v[1:], v[:-1])[:,2] / 2.)
    loss_area = tf.abs(area-0.3)
    return loss_area

def get_loss_curvature_vs_length(input_tensor):
    '''larger curvature region needs shorter lines'''
    v1 = input_tensor[1:-1] - input_tensor[:-2]
    v2 = input_tensor[2:] - input_tensor[1:-1]
    cos_dist = tf.reduce_sum(v1*v2,1) / tf.norm(v1,axis=1) / tf.norm(v2,axis=1)
    loss_curvature = tf.reduce_max(tf.square( (1-cos_dist) * (tf.norm(v1,axis=1)+tf.norm(v2,axis=1)) ))
    return loss_curvature

def get_loss_sharp_edge(input_tensor):
    import math as m
    v1 = input_tensor[1:-1] - input_tensor[:-2]
    v2 = input_tensor[2:] - input_tensor[1:-1]
    cos_dist = tf.reduce_sum(v1 * v2, 1) / tf.norm(v1, axis=1) / tf.norm(v2, axis=1)
    max_curvature = tf.reduce_max(tf.square((1 - cos_dist)))
    loss_edge = tf.nn.relu(max_curvature - (1 - tf.cos(-m.pi / 4.)))
    return loss_edge

def get_loss_stability(input_tensor):
    bs = input_tensor.get_shape().as_list()[0]
    import numpy as np
    center = np.asarray(bs * [[0.5, 0.8]])
    padding = tf.zeros((bs,1))
    v = tf.concat([center,padding],1) - tf.concat([input_tensor,padding],1)
    area_path = tf.cross(v[1:], v[:-1])[:,2] / 2.
    center_patch = (input_tensor[:-1,0]+input_tensor[1:,0]+0.5 )/3.
    center_gravity = tf.reduce_mean(center_patch * area_path) / tf.reduce_mean(area_path)

    length = input_tensor.get_shape().as_list()[0]
    bottom_left = tf.reduce_mean(input_tensor[int(length * 0.35):int(length * 0.45), 0])
    bottom_right = tf.reduce_mean(input_tensor[int(length * 0.55):int(length * 0.65), 0])
    loss_stability_case1 = tf.reduce_mean(tf.nn.relu(bottom_left-center_gravity) + tf.nn.relu(center_gravity-bottom_right))
    loss_stability_case2 = tf.reduce_mean(tf.nn.relu(bottom_right-center_gravity) + tf.nn.relu(center_gravity-bottom_left))
    loss_stability = tf.reduce_min([loss_stability_case1, loss_stability_case2])
    log_res = {}
    log_res['center'], log_res['left'], log_res['right'] = center_gravity, bottom_left, bottom_right
    return loss_stability, log_res


def get_loss_collision(input_tensor):
    v = input_tensor[1:] - input_tensor[:-1]
    v1 = tf.expand_dims(v,0)
    v2 = tf.expand_dims(v,1)
    loss_collision = v1-v2
    return loss_collision