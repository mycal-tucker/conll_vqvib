import os
import ast
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import random

import src.settings as settings
from src.data_utils.helper_fns import gen_batch, get_glove_embedding, get_unique_labels, get_entry_for_labels, get_unique_by_field
from src.data_utils.read_data import get_feature_data
from src.data_utils.read_data import load_cleaned_results
from src.models.decoder import Decoder
from src.models.listener_pragmatics import ListenerPragmatics, ListenerPragmaticsCosines
from src.models.team import Team
from src.models.vae import VAE
from src.models.vq import VQ
from src.models.vqvib2 import VQVIB2
from src.models.mlp import MLP
from src.models.proto import ProtoNetwork
from src.utils.mine_pragmatics import get_info_humans, get_info_lexsem, get_cond_info
from src.utils.NID import find_synonyms
from src.utils.plotting import plot_metrics, plot_naming, plot_scatter
from src.utils.performance_metrics import PerformanceMetrics

import time

from src.data_utils.read_data import get_glove_vectors



def p_W_I(data, model, vae=None, glove_data=None):

    human_names = []
    for i in list(data['responses']):
        for j in i.keys():
            human_names.append(j)
    human_names = set(human_names)

    human_probs = {key: [] for key in human_names}
    model_probs = {key: [] for key in range(settings.num_protos)}

    for targ_idx in list(data.index.values):

        speaker_obs, _, _, _ = gen_batch(data, 1, "topname", p_notseedist=1, glove_data=glove_data, vae=vae, preset_targ_idx=targ_idx)

        if speaker_obs != None: # i.e. we have the glove embeds
            responses = data['responses'][targ_idx]
            total = sum(list(responses.values()))
            normalized_responses = {key: value/total for key, value in responses.items()}

            for k,v in human_probs.items():
                if k in normalized_responses.keys():
                    human_probs[k].append(v)
                else:
                    human_probs[k].append(0.0)

            # we repeat the input to get a sense of the topname
            speaker_obs = speaker_obs.repeat(100, 1, 1)
            
            with torch.no_grad():
                likelihood = model.speaker.get_token_dist(speaker_obs)
                for idx in range(len(likelihood)):
                    model_probs[idx].append(likelihood[idx])


                # we substitute the prototype with the cluster representative (if no synonym was found, we use the prototype itself)
                #for k,v in proto_synonyms.items():
                #    if index_of_one in v:
                #        model_names.append(k)

        else:
            pass
    
    M = len(model_probs[0])
    W = settings.num_protos
    model_matrix = np.empty((M, W))
    for i in range(W):
        model_matrix[:, i] = model_probs[i]
    
    return M, W, model_matrix, human_probs



def p_W(M, W, model, model_matrix):
    
    p_image = 1/M

    # remove prototypes that are synonyms to avoid noise
    prototypes = model.speaker.vq_layer.prototypes.detach().cpu()
    proto_synonyms = find_synonyms(prototypes, 0.1)
    print("prototypes' clusters found:", len(proto_synonyms))
    
    p_words = []
    for i in range(W):
        p_words.append(p_image * sum(model_matrix[:, i]))
    p_words = np.array(p_words).reshape(1, W)

    return p_words, p_image


def p_I_W(model_matrix, p_word, p_image):
    
    PRECISION = 1e-9
    p_word = p_word + PRECISION
    
    return (model_matrix * p_image) / p_word 




def run():
    if settings.see_distractors_pragmatics:
        num_imgs = 3 if settings.with_ctx_representation else 2
    else:
        num_imgs = 1 if not settings.see_distractor else (num_distractors + 1)

    data = get_feature_data(t_features_filename, excluded_ids=excluded_ids)
    data = data.sample(frac=1, random_state=seed) # Shuffle the data.

    
    # TO TEST ON ALL DATA

    print("Len data:", len(data))
    print(len(set(data['topname'])))

    # re-swap target and distractor to judge with the correct human name
    mask = data["swapped_t_d"] == 1
    data.loc[mask, ["t_features", "d_features"]] = data.loc[mask, ["d_features", "t_features"]].values
    
 
    print("context:", settings.with_ctx_representation)
    folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"


    speaker = VQ(feature_len, c_dim, num_layers=3, num_protos=settings.num_protos, num_simultaneous_tokens=1, variational=variational, num_imgs=num_imgs)
    listener = ListenerPragmaticsCosines(feature_len)
    decoder = Decoder(c_dim, feature_len, num_layers=3, num_imgs=num_imgs)
    model = Team(speaker, listener, decoder)
    model.to(settings.device)

    complexity = 1.0
    informativeness = 200
    utility = 20
    
    folder_utility = "utility" + str(utility) + "/"
    folder_alpha = "alpha" + str(informativeness) + "/"
    folder_complexity = "compl" + str(complexity) + "/"

    model_to_eval_path = 'src/saved_models/' + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight' + str(complexity) + '/' + 'seed0/' + folder_utility + folder_alpha + "4999/"
    model.load_state_dict(torch.load(model_to_eval_path + '/model.pt'))
    model.to(settings.device)
    model.eval()

    M, W, p_word_image, human_probs = p_W_I(data, model, vae=vae, glove_data=glove_data)
    print("p_W_I:", p_word_image, p_word_image.shape)
    p_word, p_image = p_W(M, W, model, p_word_image)
    print("p_word:", p_word, p_word.shape)
    p_image_word = p_I_W(p_word_image, p_word, p_image)
    print("p_I_W:", p_image_word, p_image_word.shape)


    column_sums = np.sum(p_image_word, axis=0)
    print(column_sums)


    most_prob_words = np.argsort(p_word)[-10:]
    most_prob_images_per_word = []
    for i in most_prob_words:
        most_prob_images_per_word.append(np.argsort(p_image_word[:, i])[-5:])

    print(most_prob_words)
    print(most_prob_images_per_word)



if __name__ == '__main__':
    feature_len = 512
    settings.see_distractor = False
    settings.see_distractors_pragmatics = True
    if settings.see_distractors_pragmatics:
        settings.see_distractor = False # Mycal's one
    
    settings.with_ctx_representation = False
    settings.dropout = False
    settings.see_probabilities = True
    
    settings.eval_someRE = False
    
    num_distractors = 1
    settings.num_distractors = num_distractors
    c_dim = 128
    variational = True
    settings.num_protos = 3000 # 442 is the number of topnames in MN 

    settings.random_init = False
    random_init_dir = "random_init/" if settings.random_init else "anneal/"

    # Measuring complexity takes a lot of time. For debugging other features, set to false.
    settings.entropy_weight = 0.0 
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.learned_marginal = False
    settings.max_num_align_data = 1
    settings.distinct_words = False  # FIXME
    with_bbox = False

    t_features_filename = 'src/data/t_features.csv'
    settings.d_features_filename = 'src/data/d_features.csv'
    settings.d_bboxes_filename = 'src/data/d_xyxy.tsv'
    settings.ctx_features_filename = 'src/data/ctx_features.csv'
    manynames = load_cleaned_results(filename="src/data/manynames.tsv")
    someRE = pd.read_csv("src/data/someRE.csv", sep = ";")
    merged_tmp = pd.merge(manynames, someRE, on=['link_vg'])
    excluded_ids = [str(i) for i in merged_tmp['vg_image_id']] 
    print("excluded ids:", len(excluded_ids))

    vae = VAE(512, 32)
    vae.load_state_dict(torch.load('src/saved_models/vae0.001.pt'))
    vae.to(settings.device)
    settings.embedding_cache = {}
    settings.sample_first = True
    speaker_type = 'vq'  # Options are 'vq', 'cont', or 'onehot'
    
    seed = 0 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    glove_data = get_glove_vectors(32)
    run()

