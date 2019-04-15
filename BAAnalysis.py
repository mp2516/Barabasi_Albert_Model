import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import BarabasiAlbertNetwork.ba_model.log_bin_CN_2016 as lb
import scipy.stats as sp
from matplotlib import rc
import matplotlib

class BAModelAnalysis:
    def __init__(self, edges=[], edges_flatten=[], degrees = [], n0=0, m=0, N=0, graphtype="None"):
        """Used to analysis the growing networks
        inputs:
            either: edges, edges_flatten, degrees
            n0 - the number of nodes in the initial graph
            m - the number of edges that a new vertex connects with
            N - the total number of vertices
            graphtype - the model used"""
        self.n0 = n0
        self.m = m
        self.N = N
        self.graphtype = graphtype

        rc('font', **{'family': 'serif', 'weight': 'bold', 'size': 22})
        rc('text', usetex=True)

        if degrees != []:
            self.degrees_float = degrees
            self.degrees = self.degrees_float.astype(int)
            self.R = int(len(self.degrees)/self.N)
        else:
            if edges != []:
                if type(edges) != np.ndarray: edges = np.array(edges)
                self.edges = edges
                self.edges_flattened = self.edges.flatten().astype(int)
            elif edges_flatten != []:
                self.edges_flattened = edges_flatten.astype(int)
            
            self.degrees = np.bincount(self.edges_flattened)
            self.degrees_float = self.degrees
            self.degrees = self.degrees_float.astype(int)
        
        self.nodes = range(len(self.degrees))
        
        self.probability()
        self.strip_zeros()
        
        self.label = r'$N=10^{%s}$, $n_0=%s$, $m=%s$, $R=%s$' % (int(np.log10(self.N)), self.n0, self.m, self.R)
        
    def probability(self):
        """
        Counts the number of each degree and normalises to produce the degree frequency
        :return: self.deg_prob, self.deg
        """
        degfreq = np.bincount(self.degrees)
        self.degprob = degfreq/float(np.sum(degfreq))
        self.deg = np.array(range(len(degfreq)))
        
    def strip_zeros(self):
        """
        Removes all zeroes from the bins to ensure that a bin does not contain zero entries
        :return: self.degprob, self.deg
        """
        degprob_arg_notzero = np.argwhere(self.degprob > 0)
        self.degprob = self.degprob[degprob_arg_notzero]
        self.deg = self.deg[degprob_arg_notzero]
        
    def plot(self):
        """
        Plot the networkx interpretation of the graph.
        :return:
        """
        plt.figure('NodeEdgePlot')
        G = nx.Graph()
        G.add_edges_from(self.edges)
        nx.draw(G, node_size=20)

    def colour_plot(self, ncols):
        plt.figure('ColourNodePlot')
        pos = np.array([(i % ncols, (len(self.nodes)-i-1)//ncols) for i in self.nodes])
        node_color = np.array([self.degrees[n] for n in self.nodes])
        plt.scatter(pos[:,0], pos[:,1], c=node_color, s=node_color)
        plt.title('The nodes in order of addition \n showing number of degrees')
        plt.colorbar()
        
    def log_bin(self, a=1.5):
        self.deg_bin_original, self.degprob_bin_original = lb.logbin(self.degrees_float, scale=a)
        degprob_arg_notzero = np.argwhere(self.degprob_bin_original > 0)
        self.degprob_bin = list(np.array(self.degprob_bin_original)[degprob_arg_notzero].flatten())
        self.deg_bin = list(np.array(self.deg_bin_original)[degprob_arg_notzero].flatten())

    def degree_probability_raw_plot(self):
        plt.figure('DegreeProbabilityRawVsLogBinPlot')
        plt.scatter(self.deg, self.degprob, c=[random.random(),random.random(),random.random()], label=self.label)
        plt.title('The probability against the degree \n for the %s model' % self.graphtype)
        plt.xlabel(r'$k$')
        plt.ylabel(r'$p(k)$')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.grid(True)
        
    def degree_probability_raw_vs_logbinned_plot(self):
        plt.figure('DegreeProbabilityRawVsLogBinPlot')
        plt.scatter(self.deg, self.degprob, c=[random.random(),random.random(),random.random()], label=self.label)
        plt.plot(self.deg_bin[1:], self.degprob_bin[1:])
        plt.title('The probability against the degree \n for the %s model' % self.graphtype)
        plt.xlabel(r'$k$')
        plt.ylabel(r'$p(k)$')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.grid(True)

    def gradient_degree_probability(self):
        self.degprob_loggrad, self.degprob_logintercept = sp.linregress(np.log10(self.deg_bin[3:]), np.log10(self.degprob_bin[3:]))[:2]
        print(self.label + ": Degree probability log gradient=" + str(self.degprob_loggrad))
        
    def degree_probability_plot(self, fig, ax):
        ax.plot(self.deg_bin[1:], self.degprob_bin[1:],  label=self.label)
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$p(k)$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='best')
        # ax.grid(True)
        return fig, ax
        
    def theoretical_degree_probability(self, k):
        if self.graphtype == "Pure Random":
            try:
                return 10**((k-self.m)*np.log10(self.m) - (1+k-self.m)*np.log10(1+self.m))
            except:
                return np.nan
        else:
            return float(2*self.m*(self.m+1))/(k*(k+1)*(k+2))
        
    def degree_probability_theoretical_plot(self, fig, ax):
        k_range = np.logspace(np.log10(min(self.deg)), np.log10(max(self.deg)), 100)
        self.probtheory = [self.theoretical_degree_probability(k) for k in k_range]
        ax.plot(k_range, self.probtheory, '--')
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$p(k)$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        return fig, ax
        
    def cum_distribution_function(self):
        self.cdf = np.cumsum(self.degprob)[::-1]
        plt.figure('CDFPlot')
        plt.title('The CDF for the degree distribution \n for the %s model' % self.graphtype)
        plt.xlabel(r'$k$')
        plt.ylabel(r'$cdf(k)$')
        plt.plot(self.deg, self.cdf,  label='Theoretical: m='+str(self.m))
        plt.legend(loc='best')
        plt.grid(True)
        
    def numerical_cdf(self, k):
        return np.cumsum(self.degprob)[k:]
        
    def theoretical_cdf(self, k):
        degprob_theoretical = np.array([self.theoretical_degree_probability(i) for i in self.deg])
        return np.cumsum(degprob_theoretical)[k:]
        
    def p_value(self):
        self.probtheory = np.array([self.theoretical_degree_probability(int(k)) for k in self.deg])
        self.KS_range = [sp.ks_2samp(self.degprob.flatten()[:k]*self.N, self.probtheory[:k]*self.N).pvalue for k in range(1, len(self.deg))]
        plt.figure('PValuePlot')
        plt.title('The P Value of the KS test against the degree \n for the %s model' % self.graphtype)
        plt.xlabel(r'$k$')
        plt.ylabel(r'$P Value(k)$')
        plt.plot(self.deg[1:], self.KS_range,  label=self.label)
        plt.legend(loc='best')
        plt.grid(True)
        
    def test_statistics(self):
        degprob_theoretical = np.array([self.theoretical_degree_probability(k) for k in self.deg])
        chisquare = sp.chisquare(self.degprob, degprob_theoretical).statistic[0]
        KS = sp.ks_2samp(self.degprob.flatten(), degprob_theoretical.flatten()).statistic
        print(self.label + ": Chi-square=", chisquare, ", KS=", KS)
        
    def degree_probability_collapsed_plot(self, k1):
        plt.figure('DegreeProbabilityCollapsedPlot')
        y_theo = np.array([self.theoretical_degree_probability(i) for i in self.deg_bin[1:]])
        x=np.array(self.deg_bin[1:])/np.mean(k1)
        y=np.array(self.degprob_bin[1:])/y_theo
        plt.plot(x, y,  label=self.label)
        plt.title('The collapsed probability against the degree \n for the %s model' % self.graphtype)
        plt.xlabel(r'$k/k_1$')
        plt.ylabel(r'$p(k)/p_\infty(k)$')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.grid(True)
        
    def degree_probability_collapsed_plot_v2(self, tau_k, D):
        plt.figure('DegreeProbabilityCollapsedPlot')
        x=np.array(self.deg_bin[1:])/float(self.N**D)
        y=np.array(self.degprob_bin[1:])*np.array(self.deg_bin[1:])**tau_k
        plt.plot(np.log10(x), np.log10(y),  label=self.label)
        plt.title('The collapsed probability against the degree \n for the %s model' % self.graphtype)
        plt.xlabel(r'$k/N^D$')
        plt.ylabel(r'$p(k)s^{-\tau_k}$')
        #plt.xscale('log')
        #plt.yscale('log')
        plt.legend(loc='best')
        plt.grid(True)