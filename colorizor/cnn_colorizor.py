import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
from skimage import color
from skimage.transform import resize
import numpy as np
import cv2

def _variable(name, shape, initializer):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        wd = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', wd)
    return var

class Net(object):
    def __init__(self):
        tf.reset_default_graph()
        pass
    
    def conv2d(self, scope, input, kernel_size, stride=1, dilation=1, relu=True, wd=0.0):
        with tf.variable_scope(scope) as scope:
            kernel = _variable_with_weight_decay('weights', shape=kernel_size, stddev=5e-2, wd=wd)
            if (dilation == 1):
                conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding='SAME')
            else:
                conv = tf.nn.atrous_conv2d(input, kernel, dilation, padding='SAME')
            biases = _variable('biases', kernel_size[3:], tf.constant_initializer(0.))
            bias = tf.nn.bias_add(conv,biases)
            if relu:
                conv1 = tf.nn.relu(bias)
            else:
                conv1 = bias
        return conv1
    
    def deconv2d(self, scope, input, kernel_size, stride=1, wd=0.0):
        pad_size = int((kernel_size[0] - 1) / 2)
        batch_size, height, width, in_channel = [int(i) for i in input.get_shape()]
        out_channel = kernel_size[3]
        kernel_size = [kernel_size[0], kernel_size[1], kernel_size[3], kernel_size[2]]
        output_shape = [batch_size, height * stride, width * stride, out_channel]
        with tf.variable_scope(scope) as scope:
            kernel = _variable_with_weight_decay('weights', shape=kernel_size, stddev=5e-2, wd=wd)
            deconv = tf.nn.conv2d_transpose(input, kernel, output_shape, [1, stride, stride, 1], padding='SAME')
            biases = _variable('biases', (out_channel), tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(deconv, biases)
            deconv1 = tf.nn.relu(bias)
        return deconv1

    def batch_norm(self, scope, x):
        return tf.keras.layers.BatchNormalization(name=scope, center=True, scale=True, trainable=False)(x)

    def inference(self, input):
        # model 1
        temp_conv = self.conv2d('conv1', input, [3, 3, 1, 64], stride=1)
        temp_conv = self.conv2d('conv2', temp_conv, [3, 3, 64, 64], stride=2)
        temp_conv = self.batch_norm('bn_1', temp_conv)

        # model_2
        temp_conv = self.conv2d('conv3', temp_conv, [3, 3, 64, 128], stride=1)
        temp_conv = self.conv2d('conv4', temp_conv, [3, 3, 128, 128], stride=2)
        temp_conv = self.batch_norm('bn_2', temp_conv)

        # model_3
        temp_conv = self.conv2d('conv5', temp_conv, [3, 3, 128, 256], stride=1)
        temp_conv = self.conv2d('conv6', temp_conv, [3, 3, 256, 256], stride=1)
        temp_conv = self.conv2d('conv7', temp_conv, [3, 3, 256, 256], stride=2)
        temp_conv = self.batch_norm('bn_3', temp_conv)

        # model_4
        temp_conv = self.conv2d('conv8', temp_conv, [3, 3, 256, 512], stride=1)
        temp_conv = self.conv2d('conv9', temp_conv, [3, 3, 512, 512], stride=1)
        temp_conv = self.conv2d('conv10', temp_conv, [3, 3, 512, 512], stride=1)
        temp_conv = self.batch_norm('bn_4', temp_conv)

        # model_5
        temp_conv = self.conv2d('conv11', temp_conv, [3, 3, 512, 512], stride=1, dilation=2)
        temp_conv = self.conv2d('conv12', temp_conv, [3, 3, 512, 512], stride=1, dilation=2)
        temp_conv = self.conv2d('conv13', temp_conv, [3, 3, 512, 512], stride=1, dilation=2)
        temp_conv = self.batch_norm('bn_5', temp_conv)

        # model_6
        temp_conv = self.conv2d('conv14', temp_conv, [3, 3, 512, 512], stride=1, dilation=2)
        temp_conv = self.conv2d('conv15', temp_conv, [3, 3, 512, 512], stride=1, dilation=2)
        temp_conv = self.conv2d('conv16', temp_conv, [3, 3, 512, 512], stride=1, dilation=2)
        temp_conv = self.batch_norm('bn_6', temp_conv)    

        # model_7
        temp_conv = self.conv2d('conv17', temp_conv, [3, 3, 512, 512], stride=1)
        temp_conv = self.conv2d('conv18', temp_conv, [3, 3, 512, 512], stride=1)
        temp_conv = self.conv2d('conv19', temp_conv, [3, 3, 512, 512], stride=1)
        temp_conv = self.batch_norm('bn_7', temp_conv)

        # model_8
        temp_conv = self.deconv2d('conv20', temp_conv, [4, 4, 512, 256], stride=2)
        temp_conv = self.conv2d('conv21', temp_conv, [3, 3, 256, 256], stride=1)
        temp_conv = self.conv2d('conv22', temp_conv, [3, 3, 256, 256], stride=1)

        # out
        temp_conv = self.conv2d('conv23', temp_conv, [1, 1, 256, 313], stride=1, relu=False)

        conv8_313 = temp_conv
        return conv8_313

def colorize(img, ckpt, pts):
    if (len(img.shape) == 3):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[None, :, :, None]
    input = (img.astype(dtype=np.float32)) / 255. * 100. - 50.

    net = Net()
    output = net.inference(input)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        output = sess.run(output)

    input = input + 50
    _, h, w, _ = input.shape
    input = input[0, :, :, :]

    output = output[0, :, :, :]
    output = output * 2.63
    
    def softmax(x):
        e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
        return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1)
    
    output = softmax(output)
    cc = np.load(pts)
    
    output_ab = np.dot(output, cc)
    output_ab = resize(output_ab, (h, w))
    img = np.concatenate((input, output_ab), axis=-1)
    img = color.lab2rgb(img)
    img = img[:, :, ::-1]
    img = img * 255
    img = img.astype(np.uint8)

    return img
