import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch import optim
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

import src.settings as settings
from src.data_utils.helper_fns import gen_batch

PRECISION = 1e-16

def get_prob(list_of_words):
    
    counts = Counter(list_of_words)
    total_elements = len(list_of_words)
    probabilities = {key: count / total_elements for key, count in counts.items()}

    prob = np.array(list(probabilities.values()))
    
    return prob


def xlogx(v):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(v > PRECISION, v * np.log2(v), 0)


def H(p, axis=None):
    """ Entropy """
    return -xlogx(p).sum(axis=axis)


def MI(list1, list2):
    
    # joint distribution
    joint_counts = Counter(zip(list1, list2))
    total_counts = len(list1)  
    joint_probabilities = {k: v / total_counts for k, v in joint_counts.items()}

    # Marginal probabilities for each list
    marginal_prob_list1 = Counter(list1)
    marginal_prob_list2 = Counter(list2)
    marginal_prob_list1 = {k: v / total_counts for k, v in marginal_prob_list1.items()}
    marginal_prob_list2 = {k: v / total_counts for k, v in marginal_prob_list2.items()}

    # mutual information
    mi = 0
    for (x, y), joint_p in joint_probabilities.items():
        mi += joint_p * math.log2(joint_p / (marginal_prob_list1[x] * marginal_prob_list2[y]))

    return mi


def NID(U, V):
    I = MI(U, V)
    HU = H(get_prob(U))
    HV = H(get_prob(V))
    score = 1 - I / (np.max([HU, HV]))
    return score


def euclidean_distance(tensor1, tensor2):
    return np.linalg.norm(tensor1 - tensor2)


def find_synonyms(prototypes, epsilon):
    
    distances = squareform(pdist(prototypes, 'euclidean'))
    # dbscan clustering
    dbscan = DBSCAN(eps=epsilon, min_samples=1, metric='precomputed')
    clusters = dbscan.fit_predict(distances)

    cluster_dict = {}
    for idx, cluster in enumerate(clusters):
        if cluster in cluster_dict:
            cluster_dict[cluster].append(idx)
        else:
            cluster_dict[cluster] = [idx]
    
    # we organize a dictionary that maps each cluster to one vector that represents the cluster
    synonyms = {}
    for k,v in cluster_dict.items():
        synonyms[v[0]] = v
    
    assert len(synonyms) == len(cluster_dict), "Mismatch!"
    
    return synonyms
    
    
def get_NID(model, dataset, batch_size, field, glove_data=None, vae=None):
   
    # remove prototypes that are synonyms to avoid noise
    prototypes = model.speaker.vq_layer.prototypes.detach().cpu()
    proto_synonyms = find_synonyms(prototypes, 0.1)
    print("prototypes' clusters found:", len(proto_synonyms))   
#    if len(proto_synonyms.keys()) < 2:
#        subset_prototypes = find_synonyms(prototypes, 0.05)

    human_names = []
    model_names = []

    for targ_idx in list(dataset.index.values):
        
        speaker_obs, _, _, _ = gen_batch(dataset, 1, "topname", p_notseedist=1, glove_data=glove_data, vae=vae, preset_targ_idx=targ_idx)
        
        if speaker_obs != None: # i.e. we have the glove embeds    
            human_names.append(dataset[field][targ_idx])
            
            # we repeat the input to get a sense of the topname
            speaker_obs = speaker_obs.repeat(100, 1, 1)
            
            with torch.no_grad(): 
                likelihood = model.speaker.get_token_dist(speaker_obs)
                index_of_one = torch.argmax(torch.tensor(likelihood)).item()
             
                # we substitute the prototype with the cluster representative (if no synonym was found, we use the prototype itself)
                for k,v in proto_synonyms.items():
                    if index_of_one in v:
                        model_names.append(k)
                
        else:
            pass
    
    assert len(human_names) == len(model_names), "Mismatch!"
    
    word_count = len(proto_synonyms)
    
    return NID(human_names, model_names), word_count 
         
 
