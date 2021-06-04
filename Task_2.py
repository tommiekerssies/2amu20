#%%
from numpy.lib.function_base import _percentile_dispatcher
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.special import logsumexp
import random
import numpy as np
import csv
import itertools
import math
#%%
NUM_RV = 16

with open('data/nltcs/nltcs.train.data', "r") as file:
    reader = csv.reader(file, delimiter=',')
    list = list(reader)
    data = np.array([list[i][0:NUM_RV] for i in range(len(list))]).astype(float)
print(data)
#%%
# TODO put computations in log domain
# TODO validate computations
class BinaryCLT:
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

    def log_prob(self, x, exhaustive=False):
        cpt = np.exp(self.get_log_params())
        tree = self.get_tree()
        result = np.empty((len(x), 1))
        
        if exhaustive:
            combinations = list(itertools.product([0, 1], repeat=self.num_rv))
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
                
        else:
            for q_i, query in enumerate(x):
                nan_rvs = []
                
                for q_rv, q_value in enumerate(query):
                    if np.isnan(q_value):
                        nan_rvs.append(q_rv)

                # TODO can we remove this?
                # if len(nan_rvs) == 0:
                #     # TODO result[q_i][0] = self.log_prob...
                #     result[q_i] = log_prob(self, [query], exhaustive=True)[0]
                #     continue

                big_cpts = {}
                for nan_rv in nan_rvs:
                    nan_rv_cpt = cpt[nan_rv]
                    not_nan_parent = [parent for child, parent in enumerate(tree) if child == nan_rv and parent not in nan_rvs]
                    not_nan_children = [child for child, parent in enumerate(tree) if parent == nan_rv and child not in nan_rvs]
                    child_cpts = [cpt[child] for child in not_nan_children]
                    parent_rv = tree[nan_rv]
                    #[(1, [0,1], 0.5)] rv, children, prob
                    big_cpt = []
                    if len(not_nan_children) == 0:
                        if parent_rv in nan_rvs or parent_rv == -1:
                            continue
                        for parent_val in [0,1]:
                            marginalized_prob = 0
                            for nan_rv_value in [0,1]:
                                probability = 1
                                if parent_rv == [rv for rv, parent in enumerate(tree) if parent == -1][0]:
                                    probability *= self.probabilities[parent_rv][parent_val]
                                probability *= nan_rv_cpt[nan_rv_value][parent_val]
                                marginalized_prob += probability
                            big_cpt.append((parent_val, (), marginalized_prob))
                    else:
                        for child_value_combination in itertools.product([0, 1], repeat=len(not_nan_children)):
                            if parent_rv == -1 or parent_rv in nan_rvs:
                                range = [None]
                            else:
                                range = [0,1]
                            
                            for parent_val in range:
                                marginalized_prob = 0
                                for nan_rv_value in [0,1]:
                                    if parent_val is None:
                                        if parent_rv in nan_rvs:
                                            probability = self.probabilities[nan_rv][nan_rv_value]
                                        else:
                                            probability = nan_rv_cpt[0][nan_rv_value]
                                    else:
                                        probability = nan_rv_cpt[nan_rv_value][parent_val]

                                    for child_idx, child_value in enumerate(child_value_combination):
                                        probability *= child_cpts[child_idx][child_value][nan_rv_value]

                                    marginalized_prob += probability
                                
                                big_cpt.append((parent_val, child_value_combination, marginalized_prob))

                    big_cpts[nan_rv] = big_cpt

                # # big cpts have not been marginalized on nan parents yet, so do that below:
                # for nan_rv, big_cpt in big_cpts.items():
                #     parent_rv = tree[nan_rv]
                #     if parent_rv in nan_rvs:
                #         new_big_cpt = []
                #         for i1, big_cpt_tuple in enumerate(big_cpt):
                #             for i2, big_cpt_tuple2 in enumerate(big_cpt):
                #                 # check if we already marginalized these two tuples, if so, then skip
                #                 skip = False
                #                 for new_row in new_big_cpt:
                #                     if new_row[1] == big_cpt_tuple[1]:
                #                         skip = True
                #                         break
                                
                #                 if not skip and i1 != i2 and big_cpt_tuple[1] == big_cpt_tuple2[1]:
                #                     new_big_cpt.append(
                #                         (None, big_cpt_tuple[1], big_cpt_tuple[2] + big_cpt_tuple2[2])
                #                     )
                #         big_cpts[nan_rv] = new_big_cpt

                used_tuples = []
                probability = 1
                for nan_rv, big_cpt in big_cpts.items():
                    parent_nan_rv = tree[nan_rv]
                    not_nan_children = [child for child, parent in enumerate(tree) if parent == nan_rv and child not in nan_rvs]
                    for big_cpt_tuple in big_cpt:
                        if big_cpt_tuple[0] is None or big_cpt_tuple[0] == query[parent_nan_rv]:
                            children_values_same = True
                            if len(big_cpt_tuple[1]) > 0:
                                for child_idx, child in enumerate(not_nan_children):
                                    if big_cpt_tuple[1][child_idx] != query[child]:
                                        children_values_same = False
                                        break
                            if children_values_same:
                                used = False
                                for used_tuple in used_tuples:
                                    if big_cpt_tuple[0] == used_tuple[0] and big_cpt_tuple[1] == used_tuple[1] and math.isclose(big_cpt_tuple[2], used_tuple[2]):
                                        used = True
                                        break
                                if not used:
                                    probability *= big_cpt_tuple[2]
                                    used_tuples.append(big_cpt_tuple)
                                    break

                for q_rv, q_value in enumerate(query):
                    parent_rv = tree[q_rv]
                    if not np.isnan(q_value) and not np.isnan(query[parent_rv]):
                        probability *= cpt[q_rv][q_value][query[parent_rv]]

                result[q_i][0] = probability

        return np.log(result)

    def get_tree(self):
        predecessors = self.dir_tree
        for i, val in enumerate(predecessors[1]):
            if val == -9999: # root of the tree
                predecessors[1][i] = -1
                break
        return predecessors[1]

    def get_log_params(self):
        if not hasattr(self, 'cpt') or self.cpt is None or self.cpt.size == 0:
            self.cpt = np.empty((self.num_rv, 2, 2))
            for rv in range(self.num_rv):
                predecessor = self.get_tree()[rv]
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
            print('CPT already created')
        return np.log(self.cpt)

    def sample(self, n_samples):
        samples = np.empty((n_samples, self.num_rv), dtype=int)
        ordering = self.dir_tree[0]
        cpt = np.exp(self.get_log_params())
        tree = self.get_tree()
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

clt = BinaryCLT(data, 3)
# %%
# TODO remove print statements

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


# input = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,1,np.nan,
#         np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
# print(log_prob(clt, [input], exhaustive=False))
# print(log_prob(clt, [input], exhaustive=True))

# combinations = list(itertools.product([0, 1, np.nan], repeat=clt.num_rv))
# for combination in combinations:
#     exhaustive = clt.log_prob([combination], exhaustive=False)
#     non_exhaustive = clt.log_prob([combination], exhaustive=True)
#     if exhaustive == non_exhaustive:
#         print('pass')
#     else:
#         print(exhaustive - non_exhaustive, exhaustive, non_exhaustive, combination)
# %%
cpt = np.exp(clt.get_log_params())
print(cpt[13])
# %%
predecessors = clt.get_tree()   
print(predecessors)
# %%
print(np.exp(clt.get_log_params()))
# %%
print(clt.get_log_params())
# %%
del list
# %%
print(np.sum(np.exp(clt.log_prob(np.array(list(itertools.product([0, 1], repeat=clt.num_rv))), exhaustive=False))))
#%%
import time
start = time.time()
clt.log_prob([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, np.nan, 1, 1, 1]], exhaustive=True)
print(time.time() - start)

# %%
# TODO test sample function
# %%
# tree = clt.get_tree() 
# nan_rv = 4
# children = [child for child, parent in enumerate(tree) if parent == nan_rv]
# print(children)
# # %%
# for i in [0,1]:
#     print(i)
# %%
