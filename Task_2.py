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
# TODO put computations in log domain
# TODO validate computations
class BinaryCLT :
    def __init__ (self, data, root=None, alpha=0.01):
        # Construct tree
        num_samples = len(data) + 4 * alpha
        self.num_rv = len(data[0])
        self.probabilities = np.empty((self.num_rv, 2))
        self.joint_distribution = np.empty((self.num_rv, self.num_rv, 2, 2))
        for rv_i in range(self.num_rv):
            # first compute simple probability of 0 or 1
            times_rv_i_is_1 = 0
            for sample in data:
                if sample[rv_i] == 1:
                    times_rv_i_is_1 += 1
            prob_rv_i_is_1 = (2 * alpha + times_rv_i_is_1) / num_samples
            self.probabilities[rv_i][0] = 1 - prob_rv_i_is_1
            self.probabilities[rv_i][1] = prob_rv_i_is_1

            # then compute joint probability for this rv with all other rv's
            for rv_j in range(self.num_rv):
                rv_i_1_rv_j_0, rv_i_0_rv_j_1, rv_i_1_rv_j_1, rv_i_0_rv_j_0 = alpha, alpha, alpha, alpha
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
                self.joint_distribution[rv_i][rv_j][0][0] = rv_i_0_rv_j_0 / num_samples
                self.joint_distribution[rv_i][rv_j][0][1] = rv_i_0_rv_j_1 / num_samples
                self.joint_distribution[rv_i][rv_j][1][0] = rv_i_1_rv_j_0 / num_samples
                self.joint_distribution[rv_i][rv_j][1][1] = rv_i_1_rv_j_1 / num_samples
 
        mutual_information_table = np.empty((self.num_rv, self.num_rv))
        for i in range(self.num_rv): 
            for j in range(self.num_rv):
                jdt = self.joint_distribution[i][j]
                if i == j: # else nan values because of log 0, think MI should be 0: https://stats.stackexchange.com/questions/161429/why-would-perfectly-similar-data-have-0-mutual-information/423640#:~:text=Intuitively%2C%20mutual%20information%20measures%20the,reduces%20uncertainty%20about%20the%20other.&text=As%20a%20result%2C%20in%20this,of%20Y%20(or%20X).
                    mutual_information_table[i][j] = 0
                else:
                    mutual_information_table[i][j] =jdt[0][0]*np.log(jdt[0][0]/((jdt[0][0]+jdt[1][0])*(jdt[0][0]+jdt[0][1]))) + \
                                                    jdt[0][1]*np.log(jdt[0][1]/((jdt[0][1]+jdt[1][1])*(jdt[0][0]+jdt[0][1]))) + \
                                                    jdt[1][0]*np.log(jdt[1][0]/((jdt[0][0]+jdt[1][0])*(jdt[1][0]+jdt[1][1]))) + \
                                                    jdt[1][1]*np.log(jdt[1][1]/((jdt[0][1]+jdt[1][1])*(jdt[1][0]+jdt[1][1])))

        tree = minimum_spanning_tree(-mutual_information_table)
        self.dir_tree = breadth_first_order(tree, root if root is not None else random.randint(0, self.num_rv), directed=False) 
        pass

clt = BinaryCLT(data, 8)
# %%
# TODO move into object
def get_tree(self):
    predecessors = self.dir_tree
    for i, val in enumerate(predecessors[1]):
        if val == -9999: # root of the tree
            predecessors[1][i] = -1
            break
    return predecessors[1]

def get_log_params(self):
    cpt = np.empty((self.num_rv, 2, 2))
    for rv in range(self.num_rv):
        predecessor = get_tree(self)[rv]
        if predecessor == -1:
            cpt[rv][0], cpt[rv][1] = self.probabilities[rv], self.probabilities[rv]
        else:
            jdt = self.joint_distribution[rv][predecessor]
            probabilities_predecessor = self.probabilities[predecessor]
            cpt[rv][0][0] = jdt[0][0] / probabilities_predecessor[0]
            cpt[rv][0][1] = jdt[0][1] / probabilities_predecessor[1]
            cpt[rv][1][0] = jdt[1][0] / probabilities_predecessor[0]
            cpt[rv][1][1] = jdt[1][1] / probabilities_predecessor[1]
    return np.log(cpt)

def log_prob(self, x, exhaustive=False):
    pass

def sample(self, n_samples):
    pass
# %%
predecessors = get_tree(clt)   
print(predecessors)
# %%
print(np.exp(get_log_params(clt)))
# %%
print(clt.probabilities)
# %%
