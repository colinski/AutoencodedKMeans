#!/usr/bin/env python
import sys
from matplotlib import pyplot as plt
from parse_log import parse_file

def plot(data_set):
    rands = []
    mis = []
    v_scores = []
    scale = [2,64,128,200,256,300,400]
    for size in ['002','064',128,200,256,300,400]:
        plot_vals = parse_file('../../logs/%s/%s_ae_kmeans_%s.log' % (data_set, data_set,str(size)))
        rands.append(plot_vals[0])
        mis.append(plot_vals[1])
        v_scores.append(plot_vals[2])
   
    vanilla_vals = parse_file('../../logs/%s/%s_vanilla_kmeans.log' % (data_set, data_set))

    plt.xlim(0,400)
    plt.ylim(0,0.6)
    plt.gca().set_autoscale_on(False)
    
    plt.plot(scale, v_scores, marker='^', linestyle='-', label=r'AkM V-Measure', color='b') 
    plt.axhline(y=vanilla_vals[2], color='b', linestyle='-.', label=r'  kM V-Measure')
    
    plt.plot(scale, mis, marker='s', linestyle='-', label=r'AkM NMI', color='r')
    plt.axhline(y=vanilla_vals[1], color='r', linestyle=':', label=r'  kM NMI')
    
    plt.plot(scale, rands, marker='o', linestyle='-', label=r'AkM Rand Score', color='g') 
    plt.axhline(y=vanilla_vals[0], color='g', linestyle='--', label=r'  kM Rand Score')
    
    plt.xlabel('Representation Dimension')
    plt.ylabel('Score Value')
    plt.title(data_set.upper())
    
    l=plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1), fancybox=True, shadow=True, ncol=3)
    plt.savefig('e_plot.png', bbox_extra_artists=(l,), bbox_inches='tight')

if __name__ == '__main__':
    plot('emnist')
