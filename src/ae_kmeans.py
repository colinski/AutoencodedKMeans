from __future__ import division
import numpy as np
import tensorflow as tf
from mnist import MNIST
from sklearn.metrics import adjusted_rand_score as ars
from sklearn.metrics import adjusted_mutual_info_score as mis
from sklearn.metrics import v_measure_score as v_score
import sys

class AutoencodedKMeans(object):
    def __init__(self, data, labels, num_clusters, embed_dim, epochs, batch_size=100, nonlinear=tf.nn.tanh):
        self.raw_images = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_clusters = num_clusters
        self.data_dim = data.shape[1]
        self.embed_dim = embed_dim
        self.nonlinear = nonlinear
        self.labels = labels
        train_images = tf.train.slice_input_producer([data], num_epochs=self.epochs)
        train_images = tf.squeeze(tf.train.batch([train_images], batch_size=self.batch_size))
        train_images = tf.cast(train_images, tf.float32)
        self.test_images = tf.placeholder(tf.float32, shape=[None, self.data_dim])
        
        seed = 1111        
        
        centers = tf.get_variable(name='cluster_centers', initializer=tf.eye(self.num_clusters, self.embed_dim))
        centers = tf.expand_dims(centers, 0)
        
        input_weights = tf.get_variable(name='input_weights',
                shape=[self.data_dim, self.embed_dim],
                initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        
        input_bias = tf.get_variable(name='input_bias',
                shape=[self.embed_dim],
                initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        
        output_weights = tf.get_variable(name='output_weights',
                shape=[self.embed_dim, self.data_dim],
                initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        
        output_bias = tf.get_variable(name='output_bias',
                shape=[self.data_dim],
                initializer=tf.contrib.layers.xavier_initializer(seed=seed))

        self.norm = tf.norm(centers, ord='fro', axis=[-2,-1])
        
        #train graph
        #autoencoder
        embed_x = self.nonlinear(tf.matmul(train_images, input_weights) + input_bias)
        recon_x = tf.matmul(embed_x, output_weights) + output_bias
        self.recon_loss = tf.reduce_mean(tf.reduce_mean((train_images - recon_x)**2, 1))
        
        #kmeans
        expanded_embed_x = tf.expand_dims(embed_x, 1)
        dists = tf.reduce_sum((expanded_embed_x - centers)**2, 2)
        mins = tf.reduce_min(dists, 1) 
        self.kmeans_loss = tf.reduce_mean(0.5*mins)

        #optimization
        self.loss = self.kmeans_loss + self.recon_loss
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

        #test graph
        #put on cpu because training fills up the gpu memory
        with tf.device('/cpu:0'):
            test_embed_x = self.nonlinear(tf.matmul(self.test_images, input_weights) + input_bias)
            expanded_test_embed_x = tf.expand_dims(test_embed_x, 1)
            test_dist = tf.reduce_sum((expanded_test_embed_x - centers)**2, 2)
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
                    print('%s\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f' 
                            % (step, ars(self.labels, assignments), mis(self.labels, assignments), v_score(self.labels, assignments), self.purity_score(self.labels, assignments), norm, sum / step))
        except tf.errors.OutOfRangeError:
            print 'Done training'
        finally:
            self.coord.request_stop()
        self.coord.join(self.threads)

    def purity_score(self, true_labels, cluster_labels):
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
    images = np.load('data/emnist_train_images.npy')    
    labels = np.load('data/emnist_train_labels.npy')    
    dim = int(sys.argv[1])
    print('\nDIM=%s' % dim)
    kmeans = AutoencodedKMeans(data=images, labels=labels, num_clusters=260, embed_dim=dim, epochs=200, batch_size=100, nonlinear=tf.nn.tanh)
    kmeans.train()

if __name__ == '__main__':
    main()
