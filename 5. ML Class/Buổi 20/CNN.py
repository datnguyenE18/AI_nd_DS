import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import os
import numpy as np
import random
import sys
from os.path import isfile, join
import scipy

# augmentation #############################################
def randomCrop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])
    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad, mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    return new_batch

def randomFlipLeftRight(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

def randomRotation(batch, max_angle):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            angle = random.uniform(-max_angle, max_angle)
            batch[i] = scipy.ndimage.interpolation.rotate(batch[i], angle, reshape=False)
    return batch

def augmentation(batch, img_size):
    batch = randomCrop(batch, (img_size, img_size), 10)
    batch = randomFlipLeftRight(batch)
    batch = randomRotation(batch, 10)

    return batch
###############################################

IMG_SIZE = 96
NUM_CLASS = -1
SAVE_FOLDER = "./CNN/SAVE/"

class DataSet():
    def __init__(self, batchSize):
        self.imgSet = [];
        self.labels = [];
        self.labelMap = {};
        self.batchSize = batchSize;

    def loadFolder(self, folder):
        subs = [x[0] for x in os.walk(folder) if x[0]!=folder]
        idx = 0;
        for subDir in subs:
            filelist = [file for file in os.listdir(subDir) if file.endswith('.jpg')]

            self.labels.append(idx);
            self.labelMap[idx] = subDir;
            print(idx, subDir)
            idx = idx+1
            imgs = []
            for filename in filelist:
                img = cv2.imread(join(subDir, filename))
                if img is not None:
                    imgs.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
            self.imgSet.append(imgs)

    def getTrainData(self):
        trainData = []
        for i in range(len(self.imgSet)):
            for j in range(int(len(self.imgSet[i])*0.8)):
                trainData.append([self.imgSet[i][j], self.labels[i]])
        return trainData

    def getTestData(self):
        trainData = []
        for i in range(len(self.imgSet)):
            for j in range(int(len(self.imgSet[i])*0.8)+1, len(self.imgSet[i])):
                trainData.append([self.imgSet[i][j], self.labels[i]])
        return trainData


def cnn(images, num_classes):
    with slim.arg_scope([slim.max_pool2d], kernel_size=[3, 3], stride=2):
        net = slim.conv2d(images, 32, [3, 3])
        net = slim.max_pool2d(net)
        net = slim.conv2d(net, 64, [3, 3])
        net = slim.max_pool2d(net)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 256)
        net = slim.fully_connected(net, num_classes)

        return net


def one_hot(index, num_classes):
    assert index < num_classes and index >= 0
    tmp = np.zeros(num_classes, dtype=np.float32)
    tmp[index] = 1.0
    return tmp

def test():
    db = DataSet(128);
    global_step = tf.contrib.framework.get_or_create_global_step()
    has_trainning = True
    has_testing = True

    db.loadFolder(r"objects")
    NUM_CLASS = len(db.labels)
    print("NUM_CLASS " + str(NUM_CLASS))
    X = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])
    Y = tf.placeholder(tf.int32, [None, NUM_CLASS])
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
    logits = cnn(X, NUM_CLASS)

    l2_loss = 1e-4 * tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    total_loss = tf.add(l2_loss, cross_entropy, name='loss')

    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss, global_step)

    prediction = tf.argmax(logits, 1)
    correct_pred = tf.equal(prediction, tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    sess = tf.InteractiveSession()
    init_op = tf.global_variables_initializer();
    writer = tf.summary.FileWriter('./CNN/summary')

    writer.add_graph(sess.graph)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('acc', accuracy)
    merge_summary = tf.summary.merge_all()

    saver = tf.train.Saver()
    sess.run(init_op)

    if not os.path.isfile(SAVE_FOLDER + 'model.ckpt.index'):
        print('Create new model')
        print('OK')
    else:
        print('Restoring existed model')
        saver.restore(sess, SAVE_FOLDER + 'model.ckpt')
        print('OK')

    data_train = db.getTrainData()
    data_test = db.getTestData()

    batch_size = db.batchSize;
    num_batch = int(len(data_train) / batch_size)
    cur_epoch = int(global_step.eval() / num_batch)

    num_batch_test = int(len(data_test) / batch_size)

    for epoch in range(cur_epoch+1, 1000):
        sys.stdout.write('\nEpoch:' + str(epoch))
        # TRAIN
        if has_trainning is True:
            np.random.shuffle(data_train)
            train_data = []
            train_label = []
            for i in range(len(data_train)):
                train_data.append(data_train[i][0]/255.0)
                train_label.append(one_hot(data_train[i][1], NUM_CLASS))

            mean_loss = []
            mean_acc = []
            for batch in range(num_batch):
                top = batch*batch_size;
                bot = min((batch+1)*batch_size, len(train_data))

                batch_img = np.asarray(augmentation(np.asarray(train_data[top:bot]), IMG_SIZE))
                batch_label = np.asarray(train_label[top:bot])


                ttl, _, acc, s = sess.run([total_loss, train_step, accuracy, merge_summary],
                                          feed_dict={X: batch_img, Y: batch_label, learning_rate: 0.001})
                writer.add_summary(s, int(global_step.eval()))
                mean_loss.append(ttl)
                mean_acc.append(acc)

            mean_loss = np.mean(mean_loss)
            mean_acc = np.mean(mean_acc)
            sys.stdout.write(' | TrainLoss:'+str(mean_loss))
            sys.stdout.write(' | TrainAcc:' + str(mean_acc))
            saver.save(sess, SAVE_FOLDER + 'model.ckpt')

        # TEST
        if has_testing is True:
            np.random.shuffle(data_test)
            test_data = []
            test_label = []
            for i in range(len(data_test)):
                test_data.append(data_test[i][0]/255.0)
                test_label.append(one_hot(data_test[i][1], NUM_CLASS))

            mean_loss = []
            mean_acc = []
            for batch in range(num_batch_test):
                top = batch * batch_size;
                bot = min((batch + 1) * batch_size, len(test_data))

                batch_img = np.asarray(augmentation(np.asarray(test_data[top:bot]), IMG_SIZE))
                batch_label = np.asarray(test_label[top:bot])

                ttl, acc, s = sess.run([total_loss, accuracy, merge_summary], feed_dict={X: batch_img, Y: batch_label})

                mean_loss.append(ttl)
                mean_acc.append(acc)

            mean_loss = np.mean(mean_loss)
            mean_acc = np.mean(mean_acc)
            sys.stdout.write(' | TestLoss:'+str(mean_loss))
            sys.stdout.write(' | TestAcc:' + str(mean_acc))


