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
from src.utils.NID import get_NID
from src.utils.plotting import plot_metrics, plot_naming, plot_scatter
from src.utils.performance_metrics import PerformanceMetrics

import time

from src.data_utils.read_data import get_glove_vectors



def normalize_and_adjust(values):
    total = sum(values)
    normalized = [round(value / total, 2) for value in values]
    rounding_error = 1 - sum(normalized)
    # adjust the largest value(s) by the rounding error
    if rounding_error != 0:
        max_value = max(normalized)
        indexes_of_max = [i for i, v in enumerate(normalized) if v == max_value]
        error_per_value = rounding_error / len(indexes_of_max)
        for index in indexes_of_max:
            normalized[index] += error_per_value
    return [round(i,3) for i in normalized]


def run():
    if settings.see_distractors_pragmatics:
        num_imgs = 3 if settings.with_ctx_representation else 2
    else:
        num_imgs = 1 if not settings.see_distractor else (num_distractors + 1)

    data = get_feature_data(t_features_filename, excluded_ids=excluded_ids)
    data = data.sample(frac=1, random_state=seed) # Shuffle the data.

    #data = data.loc[data['swapped_t_d'] == 0] # SCIL 
    #data = data.reset_index(drop=True) # SCIL

    # TO TEST ON ALL DATA

    print("Len data:", len(data))
    print(len(set(data['topname'])))

    # re-swap target and distractor to judge with the correct human name
    mask = data["swapped_t_d"] == 1
    data.loc[mask, ["t_features", "d_features"]] = data.loc[mask, ["d_features", "t_features"]].values
    
    
    #train_data, test_data, val_data = np.split(data.sample(frac=1, random_state=seed), 
    #                                    [int(.7*len(data)), int(.9*len(data))])
    #train_data, test_data, val_data = train_data.reset_index(), test_data.reset_index(), val_data.reset_index()
    #train_data, test_data, val_data = train_data.sample(frac=1, random_state=seed).reset_index(), test_data.sample(frac=1, random_state=seed).reset_index(), val_data.sample(frac=1, random_state=seed).reset_index() 
    #print("Len train set:",len(train_data), "Len val set:", len(val_data), "Len test set:", len(test_data))
    
    # TO TEST ONLY UNSEEN DATA
    
    #unseen_train_data = train_data.loc[train_data['swapped_t_d'] == 1]
    #unseen_data = pd.concat([unseen_train_data, val_data, test_data])
    #unseen_data = unseen_data.reset_index(drop=True)
    #mask = unseen_data["swapped_t_d"] == 1
    #unseen_data.loc[mask, ["t_features", "d_features"]] = unseen_data.loc[mask, ["d_features", "t_features"]].values

    #print("Len unseen data:", len(unseen_data))
    #print(len(set(unseen_data['topname'])))   


    # TO TEST WHAT THE MODEL SAW IN TRAINING

    # the swapped cases have been seen by the model but we don't have a human name for those targets
    #train_data = train_data.loc[train_data['swapped_t_d'] == 0]
    #train_data = train_data.reset_index(drop=True)

    #print("Len train data non swapped:", len(train_data))
    #print(len(set(train_data['topname'])))   


    check_data = data


    #print("dropout:", settings.dropout)
    print("context:", settings.with_ctx_representation)

    speaker = VQ(feature_len, c_dim, num_layers=3, num_protos=settings.num_protos, num_simultaneous_tokens=1, variational=variational, num_imgs=num_imgs)
    listener = ListenerPragmaticsCosines(feature_len)
    decoder = Decoder(c_dim, feature_len, num_layers=3, num_imgs=num_imgs)
    model = Team(speaker, listener, decoder)
    model.to(settings.device)

   
    print("------------------")
    
    folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"

#    if settings.random_init:
    json_file1 = 'src/saved_models/' + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight1.0/seed0/done_weights.json'
    if settings.random_init:
        json_file2 = 'src/saved_models/' + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight1.0/seed0/done_weights2.json'
        with open(json_file1, 'r') as f:
            done_triplets1 = json.load(f)
            done_triplets1 = [ast.literal_eval(i) for i in list(done_triplets1.keys())]        
        with open(json_file2, 'r') as f:
            done_triplets2 = json.load(f)
        done_triplets2 = [ast.literal_eval(i) for i in list(done_triplets2.keys())]
        done_triplets = done_triplets1 + done_triplets2
    else:
        with open(json_file1, 'r') as f:
            done_triplets = json.load(f)
            done_triplets = [ast.literal_eval(i) for i in list(done_triplets.keys())]

    for t in done_triplets:
        complexity = t[0]
        informativeness = t[1]
        utility = t[2]
        
        settings.kl_weight = normalize_and_adjust([complexity, informativeness, utility])[0] if settings.random_init else 1.0

        folder_utility = "utility" + str(utility) + "/"
        folder_alpha = "alpha" + str(informativeness) + "/"
        folder_complexity = "compl" + str(complexity) + "/"

        if settings.random_init:
            try:
                model_to_eval_path = 'src/saved_models/' + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight1.0/seed0/' + folder_utility + folder_alpha + "4999/"
                model.load_state_dict(torch.load(model_to_eval_path + '/model.pt'))
                model.to(settings.device)
                model.eval()

            except:
                model_to_eval_path = 'src/saved_models/' + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'missing/seed0/' + folder_utility + folder_alpha + folder_complexity + "4999/"
                model.load_state_dict(torch.load(model_to_eval_path + '/model.pt'))
                model.to(settings.device)
                model.eval()
        
        else: #annealing

            # get convergence epoch for that model
            json_file_path = "src/saved_models/" + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(seed) + '/'
            json_file = json_file_path+"objective_merged.json"
            with open(json_file, 'r') as f:
                existing_params = json.load(f)
            convergence_epoch = existing_params["utility"+str(utility)]["inf_weight"+str(informativeness)]['convergence epoch']
            # load model
            model_to_eval_path = 'src/saved_models/' + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(seed) + '/' + folder_utility + folder_alpha + str(convergence_epoch)
            save_eval_path = model_to_eval_path + '/evaluation/'
            model.load_state_dict(torch.load(model_to_eval_path + '/model.pt'))
            model.to(settings.device)
            model.eval()


        model_NID, w_count = get_NID(model, check_data, len(check_data), "topname", glove_data=glove_data, vae=vae)
        print("model NID:", model_NID, "I:", informativeness, "U:", utility)
            
        to_add = pd.DataFrame([[utility, informativeness, complexity, model_NID, w_count]])
        to_add.columns = ["Utility", "Alpha", "Complexity", "NID", "w_count"]

        save_path = "Plots/" + str(settings.num_protos) + "/" + random_init_dir + "simplex/"
        save_file_name = "data_NID_and_count0.1.csv"
        if os.path.exists(save_path + save_file_name):
            df = pd.read_csv(save_path + save_file_name, index_col=0)
            df_new = pd.concat([df, to_add])
            df_new.to_csv(save_path + save_file_name)
        else:
            to_add.to_csv(save_path + save_file_name)


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

