#!/usr/bin/env python
import sys
from matplotlib import pyplot as plt

def parse_file(path, delim):
    lines = open(path).readlines()
    rs, mi, v, p = 0, 0, 0, 0

    for line in lines:
        tokens = line.strip().split(delim)
        if len(tokens) > 3:
            rs = max(rs, float(tokens[1]))
            mi = max(mi, float(tokens[2]))
            v = max(v, float(tokens[3]))
            p = max(p, float(tokens[4]))

    print 'rs: %s mi: %s v_score: %s purity: %s' % (rs, mi, v, p)
    return [rs, mi, v, p]


def plot(data_set):
    if data_set == 'mnist':
        ae_rands = [0.4145]
        ae_mis = [0.507]
        ae_v_scores = [0.5533]
        k_rands = [0.3376]
        k_mis = [0.4406]
        k_v_scores = [0.5534]
    else:
        ae_rands = [0.2145]
        ae_mis = [0.3759]
        ae_v_scores = [0.4168]
        k_rands = [0.1848]
        k_mis = [0.3722]
        k_v_scores = [0.4686]

  
    scale = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for size in scale:
        if size == 0:
            continue
        plot_vals = parse_file('../../logs/%s/%s_denoise_ae_kmeans_%s.log' % (data_set, data_set, str(size)),delim='\t')
        ae_rands.append(plot_vals[0])
        ae_mis.append(plot_vals[1])
        ae_v_scores.append(plot_vals[2])
        plot_vals = parse_file('../../logs/%s/%s_noisy_vanilla_kmeans_%s.log' % (data_set, data_set, str(size)), delim='\t')
        k_rands.append(plot_vals[0])
        k_mis.append(plot_vals[1])
        k_v_scores.append(plot_vals[2])

    
    plt.xlim(0,100)
    plt.ylim(0,0.6)
    plt.gca().set_autoscale_on(False)
    
    plt.plot(scale, ae_v_scores, marker='^', linestyle='-', label='Denoised V-Measure', color='b')
    plt.plot(scale, k_v_scores, label=r'kM V-Measure', color='b', linestyle='-.')
    
    plt.plot(scale, ae_mis, marker='s', linestyle='-', label='Denoised NMI', color='r')
    plt.plot(scale, k_mis, label=r'kM NMI', color='r', linestyle=':')

    plt.plot(scale, ae_rands, marker='o', linestyle='-', label='Denosied Rand Score', color='g') 
    plt.plot(scale, k_rands, label=r'kM Rand Score', color='g', linestyle='--')

    plt.title(data_set.upper()) 
    plt.xlabel('Variance of Noise Distribution')
    plt.ylabel('Score Value')
    l=plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1), fancybox=True, shadow=True, ncol=3)
    plt.savefig('noise_plot.png', bbox_extra_artists=(l,), bbox_inches='tight')

plot('emnist')
