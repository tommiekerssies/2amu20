#%%
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.special import logsumexp
import numpy as np
import csv
#%%
with open('data/nltcs/nltcs.train.data', "r") as file:
    reader = csv.reader(file, delimiter=',')
    data = np.array(list(reader)).astype(float)
print(data)
#%%
class BinaryCLT :
    def __init__ (self, data, root=None, alpha=0.01):
        # Construct tree
        num_samples = len(data)
        num_rv = len(data[0])
        joint_distribution = np.empty((num_rv, num_rv, 2, 2))
        for rv_i in range(num_rv):
            for rv_j in range(num_rv):
                rv_i_1_rv_j_0 = 0
                rv_i_0_rv_j_1 = 0
                rv_i_1_rv_j_1 = 0
                rv_i_0_rv_j_0 = 0
                for sample in data:
                    if sample[rv_i] == 1:
                        if sample[rv_j] == 1:
                            rv_i_1_rv_j_1 += 1
                        else:
                            rv_i_1_rv_j_0 += 1
                    else:
                        if sample[rv_j] == 1:
                            rv_i_0_rv_j_1 += 1
                        else:
                            rv_i_0_rv_j_0 += 1
                joint_distribution[rv_i][rv_j][0][0] = rv_i_0_rv_j_0 / num_samples
                joint_distribution[rv_i][rv_j][0][1] = rv_i_0_rv_j_1 / num_samples
                joint_distribution[rv_i][rv_j][1][0] = rv_i_1_rv_j_0 / num_samples
                joint_distribution[rv_i][rv_j][1][1] = rv_i_1_rv_j_1 / num_samples
        print(CPT)
        pass

    def get_tree(self):
        pass

    def get_log_params(self):
        pass

    def log_prob(self, x, exhaustive=False):
        pass

    def sample(self, n_samples):
        pass

clt = BinaryCLT(data)