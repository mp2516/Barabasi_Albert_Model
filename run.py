import pickle
import BarabasiAlbertNetwork.ba_model.BAModelCPP as bam
from tqdm import tqdm, trange
import numpy as np

def collect_data(repeats=1000, graph="BA", vary_N=False, vary_m=False):
    data = []
    if vary_N_many:
        i_range = np.linspace(0, 5, 12)
    else:
        i_range = np.arange(1, 6, 1)
    for i in tqdm(i_range):
        if vary_m:
            N = 10 ** 5
            m = 2 ** i
            n0 = 3 ** i + 1
        elif vary_N or vary_N_many: # vary_N = True
            print(i)
            N = int(10 ** (i + 1))
            m = 3
            n0 = 4
        else:
            raise Exception("Need to set one of the parameters to true")
        sim = bam.BAModel(N=N, n0=n0, m=m, initiate=2, repeats=repeats, graph=graph)
        sim.Iterate()
        data.append(sim)
    return data

repeats = 500
graph = "Pure Random"
vary_N=False
vary_m=True
vary_N_many = False
data = collect_data(repeats=repeats, graph=graph, vary_N=vary_N, vary_m=vary_m)
# file_name = "probability_distribution_k_varying_m"
# file_name = "probability_distribution_k_varying_N"
# file_name = "probability_distribution_k_varying_N_many"
# file_name = "probability_distribution_k_varying_m_ra_v2"
file_name = "probability_distribution_k_varying_m_ra_v3"
# file_name = "probability_distribution_k_varying_N_ra"
# file_name = "probability_distribution_k_varying_N_many_ra_v2"
file_path = "C:\\Users\\18072\\PycharmProjects\\Complexity_Networks\\BarabasiAlbertNetwork\\Data\\" + file_name
pickle_out = open(file_path, "wb")
pickle.dump(data, pickle_out)
pickle_out.close()
