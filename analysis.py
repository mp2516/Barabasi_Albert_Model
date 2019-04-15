from matplotlib import rc
import random
import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy.stats as sp
from BarabasiAlbertNetwork.ba_model.log_bin_CN_2016 import logbin

plotting_colours = ['blue', 'green', 'red', 'fuchsia', 'black', 'pink', 'orange']
plotting_shapes = ['>', '<', '^', 'v', 's', 'p', '1', '2', '3', '4']
rc('font', **{'family': 'serif', 'weight': 'bold', 'size': 16})
rc('text', usetex=True)

def probability(degrees):
    """
    Counts the number of each degree and normalises to produce the degree frequency
    :return: self.deg_prob, self.deg
    """
    deg_freq = np.bincount(degrees)
    deg_prob = deg_freq / float(np.sum(deg_freq))
    deg = np.array(range(len(deg_freq)))
    return deg_freq, deg_prob, deg


def remove_zeros(deg_prob, deg):
    """
    Removes all zeroes from the bins to ensure that a bin does not contain zero entries
    :return: self.degprob, self.deg
    """
    deg_prob_arg_non0 = np.argwhere(deg_prob > 0)
    deg_prob = deg_prob[deg_prob_arg_non0]
    deg = deg[deg_prob_arg_non0]
    return deg_prob, deg


def log_bin(deg_float, a):
    deg_bin_0, deg_prob_bin_0 = logbin(deg_float, scale=a)
    deg_prob_arg_non0 = np.argwhere(deg_prob_bin_0 > 0)
    deg_prob_bin_non0 = list(np.array(deg_prob_bin_0)[deg_prob_arg_non0].flatten())
    deg_bin_non0 = list(np.array(deg_bin_0)[deg_prob_arg_non0].flatten())
    return deg_prob_bin_non0, deg_bin_non0


def theoretical_degree_probability(k_axis, graph_type, m):
    prob_theory = []
    for k in k_axis:
        if graph_type == "Pure Random":
            try:
                prob_theory.append((m **(k - m))/((m + 1)**(k - m + 1)))
            except:
                prob_theory.append(np.nan)
        else:
            prob_theory.append(float(2 * m * (m + 1)) / (k * (k + 1) * (k + 2)))
    return prob_theory

def theoretical_max_degree(N, m, graph_type):
    if graph_type == "Pure Random":
        return m - (np.log(N) / (np.log(m) - np.log(m + 1)))
    else:
        return (-1 + np.sqrt(1 + (4 * N * m * (m + 1)))) / 2

# file_name = "probability_distribution_k_varying_m"
# file_name = "probability_distribution_k_varying_N"
file_name = "probability_distribution_k_varying_N_many"
#file_name = "probability_distribution_k_varying_m_ra"
# file_name = "probability_distribution_k_varying_m_ra_v2"
# file_name = "probability_distribution_k_varying_m_ra_v3"
# file_name = "probability_distribution_k_varying_N_ra"
# file_name = "probability_distribution_k_varying_N_many_ra_v2"
file_path = "C:\\Users\\18072\\PycharmProjects\\Complexity_Networks\\BarabasiAlbertNetwork\\Data\\" + file_name
pickle_in = open(file_path,"rb")
data = pickle.load(pickle_in)

degree_probability_plot = False
statistics_test_plot = False
data_collapse = False
maximum_degree = True

fig, ax = plt.subplots()
N_all, m_all, max_deg_mean_all, max_deg_std_all, theoretical_max_deg_all = [], [], [], [], []
max_deg_max_all, max_deg_min_all = [], []
data = data[::-1]
for num, sim in enumerate(data[:]):
    n0 = sim.n0
    m = sim.m
    N = sim.N
    graph_type = sim.graphtype
    print(graph_type)
    deg_float = sim.degrees
    deg = deg_float.astype(int)
    max_deg = sim.maxdeg
    repeats = int(len(deg) / N)
    nodes = range(len(deg))

    # remove zeroes and find probability distribution
    deg_freq, deg_prob, deg = probability(deg)
    deg_prob, deg = remove_zeros(deg_prob, deg)
    k_range_log = np.logspace(np.log10(min(deg)), np.log10(max(deg)), 1000)
    k_axis = [k for k in np.linspace(min(deg), max(deg), 1000)]
    prob_theory = theoretical_degree_probability(k_axis, graph_type, m)
    k_range = [int(k) for k in deg]
    prob_theory_arr = np.array(theoretical_degree_probability(k_range, graph_type, m))

    deg_prob_bin, deg_bin = log_bin(deg_float, a=1.1)

    if degree_probability_plot:
        # ax.scatter(deg, deg_prob, color=plotting_colours[num], alpha=0.4, s=2)
        ax.plot(deg_bin[1:], deg_prob_bin[1:], label=r'$N = $'+str(N), color=plotting_colours[num], linestyle='-', linewidth=1)
        ax.plot(k_axis, prob_theory, '--', color=plotting_colours[num], linewidth=1)
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$p(k)$')
        # ax.set_xlim(2, 200)
        ax.set_xscale('log')
        ax.set_yscale('log')
    elif statistics_test_plot:
        ks_range = [sp.ks_2samp(deg_prob.flatten()[:k] * N, prob_theory_arr[:k] * N).pvalue for k in
                    range(1, len(deg))]
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'P Value (k)')
        ax.plot(deg[1:], ks_range, label=r'$N = $'+str(N), color=plotting_colours[num])
        # ad_range = [sp.anderson_ksamp(samples=(deg_prob.flatten()[:k-5] * N, prob_theory_arr[:k-5] * N)).critical_values for k in
        #             range(5, len(deg))]
        # ax.plot(deg[1:], ad_range, label=r'$m = $' + str(m), color=plotting_colours[num])
        # ax.set_xscale('log')
    elif data_collapse:
        prob_theory_bin = np.array(theoretical_degree_probability(deg_bin[1:], graph_type, m))
        ax.plot(np.array(deg_bin[1:]) / np.mean(max_deg), np.array(deg_prob_bin[1:]) / prob_theory_bin, '-', label=r'$N = $'+str(N), color=plotting_colours[num])
        ax.set_xlabel(r'$k/k_1$')
        ax.set_ylabel(r'$p(k)/p_\infty(k)$')
        # ax.set_xlim(0.3, 1.2)
        # ax.set_ylim(0.6, 2)
        ax.set_xscale('log')
        ax.set_yscale('log')
    elif maximum_degree:
        max_deg_max = max(max_deg)
        max_deg_min = min(max_deg)
        max_deg_mean = np.mean(max_deg)
        max_deg_mean_all.append(max_deg_mean)
        max_deg_max_all.append(max_deg_max)
        max_deg_min_all.append(max_deg_min)
        max_deg_std = (np.std(max_deg))
        max_deg_std_all.append(max_deg_std)
        N_all.append(N)
        m_all.append(m)
        theoretical_max_deg_all.append(theoretical_max_degree(N, m, graph_type))
        ax.set_xlabel(r'$N$')
        ax.set_ylabel(r'$k_1$')

if maximum_degree:
    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'$k_1$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.plot(N_all, max_deg_max_all, '-.', color='blue')
    # ax.plot(N_all, max_deg_min_all, '-.', color='blue')
    plt.plot(N_all, theoretical_max_deg_all, '--', label="Theoretical", color='red')
    plt.errorbar(x=N_all, y=max_deg_mean_all, yerr=max_deg_std_all, label="Numerical", color='black', fmt='.', markersize=10)
    ax.fill_between(N_all, max_deg_max_all, max_deg_min_all, color='grey', alpha=0.05, label='Numerical Max-Min')
    max_deg_grad, max_deg_intercept = sp.linregress(np.log10(N_all), np.log10(max_deg_mean_all))[:2]
    print(max_deg_grad)
    print(max_deg_intercept)

plt.legend(loc='best', fontsize=14)
plt.show()