#%%
from numpy.lib.function_base import _percentile_dispatcher
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.special import logsumexp
import random
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
 
        mutual_information_table = np.empty((num_rv, num_rv))
        for i in range(num_rv): 
            for j in range(num_rv):
                jbt = joint_distribution[i][j]
                if i == j: # else nan values because of log 0, think MI should be 0: https://stats.stackexchange.com/questions/161429/why-would-perfectly-similar-data-have-0-mutual-information/423640#:~:text=Intuitively%2C%20mutual%20information%20measures%20the,reduces%20uncertainty%20about%20the%20other.&text=As%20a%20result%2C%20in%20this,of%20Y%20(or%20X).
                    mutual_information_table[i][j] = 0
                else:
                    mutual_information_table[i][j] =jbt[0][0]*np.log(jbt[0][0]/((jbt[0][0]+jbt[1][0])*(jbt[0][0]+jbt[0][1]))) + \
                                                    jbt[0][1]*np.log(jbt[0][1]/((jbt[0][1]+jbt[1][1])*(jbt[0][0]+jbt[0][1]))) + \
                                                    jbt[1][0]*np.log(jbt[1][0]/((jbt[0][0]+jbt[1][0])*(jbt[1][0]+jbt[1][1]))) + \
                                                    jbt[1][1]*np.log(jbt[1][1]/((jbt[0][1]+jbt[1][1])*(jbt[1][0]+jbt[1][1])))
        #print(mutual_information_table)
        tree = minimum_spanning_tree(-mutual_information_table)
        print(tree)
        self.dir_tree = breadth_first_order(tree, root if root is not None else random.randint(0, num_rv)) 
        print(self.dir_tree)
        pass

    def get_tree(self):
        # klopt nog geen reet van maar wilde vast pushen
        predecessors = self.dir_tree
        print(predecessors[1])
        for i, val in predecessors[1]:
            if val == predecessors[0][0]: # root of the tree
                val = -1
        return predecessors[1]

    def get_log_params(self):
        pass

    def log_prob(self, x, exhaustive=False):
        pass

    def sample(self, n_samples):
        pass

clt = BinaryCLT(data, 8)
predecessors = clt.get_tree()   
print(predecessors)
# %%
