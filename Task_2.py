#%%
from numpy.lib.function_base import _percentile_dispatcher
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.special import logsumexp
import random
import numpy as np
import csv
import itertools

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
        num_samples = len(data) 
        self.num_rv = len(data[0])
        self.probabilities = np.empty((self.num_rv, 2))
        self.joint_distribution = np.empty((self.num_rv, self.num_rv, 2, 2))
        for rv_i in range(self.num_rv):
            # first compute simple probability of 0 or 1, marginal probabilities
            times_rv_i_is_1 = 0
            for sample in data:
                if sample[rv_i] == 1:
                    times_rv_i_is_1 += 1
            prob_rv_i_is_1 = (2 * alpha + times_rv_i_is_1) / (num_samples + 4 * alpha)
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
                self.joint_distribution[rv_i][rv_j][0][0] = rv_i_0_rv_j_0 / (num_samples + 4 * alpha)
                self.joint_distribution[rv_i][rv_j][0][1] = rv_i_0_rv_j_1 / (num_samples + 4 * alpha)
                self.joint_distribution[rv_i][rv_j][1][0] = rv_i_1_rv_j_0 / (num_samples + 4 * alpha)
                self.joint_distribution[rv_i][rv_j][1][1] = rv_i_1_rv_j_1 / (num_samples + 4 * alpha)
 
        mutual_information_table = np.empty((self.num_rv, self.num_rv))
        for i in range(self.num_rv): 
            for j in range(self.num_rv):
                jdt = self.joint_distribution[i][j]
                if i == j: # else nan values because of log 0, think MI should be 0: https://stats.stackexchange.com/questions/161429/why-would-perfectly-similar-data-have-0-mutual-information/423640#:~:text=Intuitively%2C%20mutual%20information%20measures%20the,reduces%20uncertainty%20about%20the%20other.&text=As%20a%20result%2C%20in%20this,of%20Y%20(or%20X).
                    mutual_information_table[i][j] = 0
                else:
                    # TODO replace multiplications with log additions
                    # and additions with log-sum-exp operations
                    mutual_information_table[i][j] =jdt[0][0]*np.log(jdt[0][0]/((jdt[0][0]+jdt[1][0])*(jdt[0][0]+jdt[0][1]))) + \
                                                    jdt[0][1]*np.log(jdt[0][1]/((jdt[0][1]+jdt[1][1])*(jdt[0][0]+jdt[0][1]))) + \
                                                    jdt[1][0]*np.log(jdt[1][0]/((jdt[0][0]+jdt[1][0])*(jdt[1][0]+jdt[1][1]))) + \
                                                    jdt[1][1]*np.log(jdt[1][1]/((jdt[0][1]+jdt[1][1])*(jdt[1][0]+jdt[1][1])))

        tree = minimum_spanning_tree(-mutual_information_table)
        self.dir_tree = breadth_first_order(tree, root if root is not None else random.randint(0, self.num_rv), directed=False) 

clt = BinaryCLT(data, 8)
# %%
# TODO move below functions into object
# TODO remove print statements

def get_tree(self):
    predecessors = self.dir_tree
    for i, val in enumerate(predecessors[1]):
        if val == -9999: # root of the tree
            predecessors[1][i] = -1
            break
    return predecessors[1]

# TODO weghalen en doen volgens assignment
# def _get_params(self):
#     if not hasattr(self, 'cpt') or self.cpt is None or self.cpt.size == 0:
#         self.cpt = np.empty((self.num_rv, 2, 2))
#         for rv in range(self.num_rv):
#             predecessor = get_tree(self)[rv]
#             if predecessor == -1:
#                 self.cpt[rv][0], self.cpt[rv][1] = self.probabilities[rv], self.probabilities[rv]
#             else:
#                 jdt = self.joint_distribution[rv][predecessor]
#                 probabilities_predecessor = self.probabilities[predecessor]
#                 self.cpt[rv][0][0] = jdt[0][0] / probabilities_predecessor[0]
#                 self.cpt[rv][0][1] = jdt[0][1] / probabilities_predecessor[1]
#                 self.cpt[rv][1][0] = jdt[1][0] / probabilities_predecessor[0]
#                 self.cpt[rv][1][1] = jdt[1][1] / probabilities_predecessor[1]
#     else:
#         print('CPT already created')
#     return self.cpt 

# def get_log_params(self):
#     # TODO reshape output to adhere to assignment description (switch last two dimensions)
#     # return np.log(self._get_params())
#     return np.log(_get_params(self)) 

def get_log_params(self):
    if not hasattr(self, 'cpt') or self.cpt is None or self.cpt.size == 0:
        self.cpt = np.empty((self.num_rv, 2, 2))
        for rv in range(self.num_rv):
            predecessor = get_tree(self)[rv]
            if predecessor == -1:
                self.cpt[rv][0], self.cpt[rv][1] = self.probabilities[rv], self.probabilities[rv]
            else:
                jdt = self.joint_distribution[rv][predecessor]
                probabilities_predecessor = self.probabilities[predecessor]
                self.cpt[rv][0][0] = jdt[0][0] / probabilities_predecessor[0]
                self.cpt[rv][0][1] = jdt[0][1] / probabilities_predecessor[1]
                self.cpt[rv][1][0] = jdt[1][0] / probabilities_predecessor[0]
                self.cpt[rv][1][1] = jdt[1][1] / probabilities_predecessor[1]
    else:
        # TODO remove print
        print('CPT already created')
    return np.log(self.cpt)

def log_prob(self, x, exhaustive=False):
    # cpt = self._get_params()
    # tree = self.get_tree()
    #cpt = _get_params(self)
    cpt = get_log_params(self)
    tree = get_tree(self)
    result = np.empty((len(x), 1))
    if exhaustive:
        combinations = list(itertools.product([0, 1], repeat=16))
        if not hasattr(self, 'jpmf') or self.jpmf is None or self.jpmf == {}:    
            self.jpmf = {}
            for combination in combinations:
                combination_prob = 1
                for rv, value in enumerate(combination):
                    parent_rv = tree[rv]
                    if parent_rv == -1:
                        value_conditional = value
                    else:
                        value_conditional = combination[parent_rv]
                    combination_prob *= cpt[rv][value][value_conditional]
                self.jpmf[combination] = combination_prob
        for q_i, query in enumerate(x):
            combinations_to_sum_probs_for = []
            for combination in combinations:
                include_combination = True
                for q_rv, q_value in enumerate(query):
                    if not np.isnan(q_value) and combination[q_rv] != q_value:
                        include_combination = False
                    if not include_combination:
                        break
                if include_combination:
                    combinations_to_sum_probs_for.append(combination)
            probability = 0
            for combination in combinations_to_sum_probs_for:
                probability += self.jpmf[combination] 
            result[q_i][0] = probability
            print(q_i / len(x))
    else:
        # TODO implement efficient algorithm
        pass
    return result
    # TODO return np log
        
def sample(self, n_samples):
    samples = np.empty((n_samples, self.num_rv), dtype=int)
    ordering = self.dir_tree[0]
    cpt = _get_params(self)
    tree = get_tree(self)
    for sample in samples:
        for i, rv in enumerate(ordering):
            prob_rv_is_1 = None
            if i == 0:
                prob_rv_is_1 = self.probabilities[rv][1]
            else:
                parent_rv = tree[rv]
                conditional_value = sample[parent_rv]
                prob_rv_is_1 = cpt[rv][1][conditional_value]
            random_int = random.random()
            if random_int <= prob_rv_is_1:
                sample[rv] = 1
            else:
                sample[rv] = 0
    return samples

#print(sample(clt, 5))
# %%
predecessors = get_tree(clt)   
print(predecessors)
# %%
print(np.exp(get_log_params(clt)))
# %%
print(get_log_params(clt))
# %%
import itertools
print(len(itertools.product([0, 1], repeat=16)))
# %%
print(sum(log_prob(clt, np.array(list(itertools.product([0, 1], repeat=16))), exhaustive=True)))

#%%
log_prob(clt, [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], exhaustive=True)
# %%
