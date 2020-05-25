import sys
import numpy as np
sys.path.append("..")
import SCNN
import tensorflow as tf
import Cifar10
import os
from keras.utils import to_categorical



TRAINING_BATCH = 28



lr = tf.placeholder(tf.float32)
input_real = tf.placeholder(tf.float32)
output_real = tf.placeholder(tf.float32)

global_step = tf.Variable(1, dtype=tf.int64)
step_inc_op = tf.assign(global_step, global_step + 1)
'''
Here is a reshape, because TensorFlow DO NOT SUPPORT tf.extract_image_patches gradients operation for VARIABLE SIZE inputs
'''
input_real_resize = tf.reshape(tf.exp(input_real), [TRAINING_BATCH, 32, 32, 3])

try:
    w1 = np.load('weight_CFAR_F21.npy')
    w2 = np.load('weight_CFAR_F22.npy')
    w3 = np.load('weight_CFAR_F23.npy')
    w4 = np.load('weight_CFAR_F24.npy')
    w5 = np.load('weight_CFAR_F25.npy')
    print('Done')
    layer1 = SCNN.SCNNLayer(kernel_size=5, in_channel=3, out_channel=64, strides=2, n_layer_new=1, w=w1)
    layer2 = SCNN.SCNNLayer(kernel_size=5, in_channel=64, out_channel=32, strides=2, n_layer_new=2, w=w2)
    layer3 = SCNN.SCNNLayer(kernel_size=5, in_channel=32, out_channel=16, strides=2, n_layer_new=3, w=w3)
    layer4 = SCNN.SNNLayer(in_size=256, out_size=64, n_layer=4, w=w4)
    layer5 = SCNN.SNNLayer(in_size=64, out_size=10, n_layer=5, w=w5)
    print('Weight loaded!')
except:
    layer1 = SCNN.SCNNLayer(kernel_size=5, in_channel=3, out_channel=64, strides=2, n_layer_new=1)
    layer2 = SCNN.SCNNLayer(kernel_size=5, in_channel=64, out_channel=32, strides=2, n_layer_new=2)
    layer3 = SCNN.SCNNLayer(kernel_size=5, in_channel=32, out_channel=16, strides=2, n_layer_new=3)
    layer4 = SCNN.SNNLayer(in_size=256, out_size=64, n_layer=4)
    layer5 = SCNN.SNNLayer(in_size=64, out_size=10, n_layer=5)
    print('No weight file found, use random weight')

layerout1 = layer1.forward(input_real_resize)
print(layerout1.shape)
layerout2 = layer2.forward(layerout1)
print(layerout2.shape)
layerout3 = layer3.forward(layerout2)
layerout4 = layer4.forward(tf.reshape(layerout3,[TRAINING_BATCH,256]))
layerout5 = layer5.forward(layerout4)


nnout = tf.log(layerout5)



config = tf.ConfigProto(
    device_count={'GPU': 0}
)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


print("Testing started")


scale = 3

BATCH_SIZE = 28
SAVE_PATH = os.getcwd() + '/weight_scnn'
cfar10 = Cifar10.CFAR10()

accuracy = []
time_cal = []
num_cal = []
while(True):
    xs, ys = cfar10.next_batch(batch_size=BATCH_SIZE, shuffle=False)
    xs = scale * xs
    xs = np.reshape(xs, [-1, 32, 32, 3])
    xs_new = np.reshape(xs, [-1, 3072])
    [lo, out] = sess.run([layerout5, nnout], {input_real: xs, output_real: ys})
    min_out = np.min(out, axis=1)
    layerout = np.argmin(lo, axis=1)
    layerout = to_categorical(layerout, 10)
    accurate = 0
    for i in range(len(ys)):
        xs_1 = np.extract(xs[i, :, :, :] > 0, xs[i, :, :, :])
        min_input = np.min(xs_1)
        xs_2 = xs_new[i, :]
        num = [j for j in xs_2 if 0 < j < min_out[i]]
        spike_num = len(num)
        num_cal.append(spike_num)
        time = min_out[i] - min_input
        time_cal.append(time)
        if (layerout[i] == ys[i]).all():
            accurate = accurate + 1
    accurate = accurate / 28.
    # print(accurate)
    accuracy.append(accurate)
    step = sess.run(step_inc_op)
    if step % 10 == 0:
        accurate1 = tf.reduce_mean(accuracy)
        print('Step: ' + repr(step) + ', ' + 'Accurate: ' + repr(sess.run(accurate1)))

    if step == 357:
        break

file = open('time_kitti.txt', 'w')
for k in time_cal:
    file.write(str(k))
    file.write('\n')
file.close()

file = open('Number_kitti.txt', 'w')
for t in num_cal:
    file.write(str(t))
    file.write('\n')
file.close()




