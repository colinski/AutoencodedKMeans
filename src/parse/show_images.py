#!/usr/bin/env python
from __future__ import division
import numpy as np
from sklearn.cluster import MiniBatchKMeans as KMeans
from mnist import MNIST
from matplotlib import pyplot as plt


np.random.seed(0)

images = np.load('data/emnist/emnist_train_images.npy')
labels = np.load('data/emnist/emnist_train_labels.npy')
print(images.shape)
quit()

images, labels = MNIST('./data/MNIST_data').load_training()
images = np.array(images)
labels = np.array(labels)
s = plt.subplot(1,11,1)
s.set_title(r'$\sigma^2$=0')
sp = images[1]
plt.imshow(sp.reshape(28,28))
plt.gray()
plt.xticks(())
plt.yticks(())
for i in range(1,11):
    noise = np.random.normal(0,i*11,images.shape)
    s = plt.subplot(1,11,i+1)
    s.set_title(str(i*10))
    sp = (noise + images)[1]
    plt.imshow(sp.reshape(28,28))
    plt.gray()
    plt.xticks(())
    plt.yticks(())

plt.savefig("noisy.png", bbox_inches="tight")
quit()
'''
images = np.load('data/emnist/emnist_train_images.npy')
labels = np.load('data/emnist/emnist_train_labels.npy')

for i in range(10):
        plt.subplot(1,10,i+1)
        sp = images[i]
        plt.imshow(sp.reshape(28,28))
        plt.gray()
        plt.xticks(())
        plt.yticks(())


plt.savefig("emnist_row.png", bbox_inches="tight")
quit()
'''


images, labels = MNIST('./data/MNIST_data').load_training()
images = np.array(images)
labels = np.array(labels)
noise = np.random.normal(0,.1,images.shape)

for i in range(10):
    for j in range(len(labels)):
        if labels[j] == i:
            plt.subplot(1,10,i+1)
            sp = images[j]
            plt.imshow(sp.reshape(28,28))
            plt.gray()
            plt.xticks(())
            plt.yticks(())
            break


plt.savefig("mnist_row.png", bbox_inches="tight")
