from __future__ import division
import numpy as np
import tensorflow as tf
from mnist import MNIST
from sklearn.metrics import adjusted_rand_score as ars
from sklearn.metrics import adjusted_mutual_info_score as mis
from sklearn.metrics import v_measure_score as v_score
from sklearn.cluster import MiniBatchKMeans as KMeans

class GradientKMeans(object):
    def __init__(self, data, labels, num_clusters, epochs, batch_size=100):
        self.raw_images = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_clusters = num_clusters
        self.data_dim = data.shape[1]
        self.labels = labels
        train_images = tf.train.slice_input_producer([data], num_epochs=self.epochs)
        train_images = tf.cast(tf.train.batch([train_images], batch_size=self.batch_size), tf.float32)
        
        self.test_images = tf.placeholder(tf.float32, shape=[None, self.data_dim])
        
        centers = tf.get_variable(name='cluster_centers', initializer=tf.eye(self.num_clusters, self.data_dim))
        centers = tf.expand_dims(centers, 0)
       
        self.norm = tf.norm(centers, ord='fro', axis=[-2,-1])
        
        #train graph
        dist = tf.reduce_sum((train_images - centers)**2, 2)
        mins = tf.reduce_min(dist, 1)
        self.loss = tf.reduce_sum(0.5*mins)
        self.opt = tf.train.AdamOptimizer(learning_rate=1).minimize(self.loss)

        #test graph, put on cpu because training fills up the gpu memory
        with tf.device('/cpu:0'):
            expanded_images = tf.expand_dims(self.test_images, 1)
            test_dist = tf.reduce_sum((expanded_images - centers)**2, 2)
            self.assignments = tf.argmin(test_dist, 1)
       
        #init 
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.coord  = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def train(self):
        step = 0
        sum = 0
        try:
            while not self.coord.should_stop():
                step += 1
                loss, _, norm = self.sess.run([self.loss, self.opt, self.norm])
                sum += loss
                if step % 500 == 0: 
                    assignments, = self.sess.run([self.assignments], feed_dict={self.test_images : self.raw_images})
                    print step, ars(self.labels, assignments), mis(self.labels, assignments), v_score(self.labels, assignments), purity_score(self.labels, assignments), norm, sum / step
                   
        except tf.errors.OutOfRangeError:
            print 'Done training'
        finally:
            self.coord.request_stop()
        
        self.coord.join(self.threads)
        return

def purity_score(true_labels, cluster_labels):
    L = np.column_stack((true_labels, cluster_labels))
    total = 0
    for i in np.unique(L[:,1]):
        labeled_i = L[L[:,1] == i, 0]
        counts = np.bincount(labeled_i)
        mode = np.argmax(counts)
        total += counts[mode]
    return total / len(L)

def main():
    np.random.seed(0)
    data = MNIST('./data/MNIST_data')
    images, labels = data.load_training()
    images = np.asarray(images)
    labels = np.asarray(labels)
    kmeans = KMeans(n_clusters=100)
    kmeans.fit(images)

if __name__ == '__main__':
    main()
