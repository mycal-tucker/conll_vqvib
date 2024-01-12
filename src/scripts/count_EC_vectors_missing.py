import ast
import os
import json
import pandas as pd
import pickle
import random
from scipy import stats
from scipy import spatial
from collections import Counter

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageDraw
from scipy.stats import entropy

import src.settings as settings
from src.data_utils.read_data import get_feature_data, get_glove_vectors
from src.data_utils.helper_fns import gen_batch, get_glove_embedding, get_unique_labels, get_entry_for_labels, get_unique_by_field, get_rand_entries
from src.data_utils.read_data import get_feature_data
from src.data_utils.read_data import load_cleaned_results
from src.models.decoder import Decoder
from src.models.listener_pragmatics import ListenerPragmaticsCosines
from src.models.team import Team
from src.models.vae import VAE
from src.models.vq import VQ 



def normalize_and_adjust(values):
    # Normalize and round the values
    total = sum(values)
    normalized = [round(value / total, 2) for value in values]

    # Calculate the difference from 1
    rounding_error = 1 - sum(normalized)

    # Adjust the largest value(s) by the rounding error
    if rounding_error != 0:
        # Find the index(es) of the largest value(s)
        max_value = max(normalized)
        indexes_of_max = [i for i, v in enumerate(normalized) if v == max_value]

        # Distribute the rounding error among the largest values
        error_per_value = rounding_error / len(indexes_of_max)
        for index in indexes_of_max:
            normalized[index] += error_per_value

    return normalized



def get_ec_words(dataset, team, batch_size, num_samples, utility, alpha, json_path):
    
    json_file = json_path
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            dic = json.load(f)
    else:
        dic = {}


    # PRAGMATICS       
    speaker_obs, _, _, _ = gen_batch(dataset, batch_size, fieldname, p_notseedist=0, vae=vae_model, see_distractors=settings.see_distractor, glove_data=glove_data, num_dist=num_distractors)
    
    EC_words = []
    for i in range(num_samples):
        tmp = []
        for x in speaker_obs:
            # we obtain a one-hot vector, that is the distribution over prototypes
            # 1 is at the index of the closest prototype
            likelihood, _ = team.speaker.get_token_dist(x)
            # we take th index of the closest prototype
            index_of_one = torch.argmax(torch.tensor(likelihood)).item()
            tmp.append(index_of_one)
        EC_words.append(tmp)

    count_trials = [len(set(i)) for i in EC_words]
    
    frequencies = [Counter(i) for i in EC_words]
    total_symbols = [len(i) for i in EC_words]
    probabilities_list = []
    for f,tot in zip(frequencies, total_symbols):
        tmp = []
        for c in f.values():
            tmp.append(c/tot)
        probabilities_list.append(tmp)
    entropies = [entropy(i, base=2) for i in probabilities_list]
    
    print("pragmatics - counts:", count_trials)
    print("pragmatics - average word count:", sum(count_trials)/len(count_trials))
    print("pragmatics - entropies:", entropies)
    print("pragmatics - average entropy:", sum(entropies) / len(entropies))

    pragmatics = {"word counts": count_trials,
                  "average word count": sum(count_trials)/len(count_trials),
                  "entropies": entropies,
                  "average entropy": sum(entropies) / len(entropies)}

    metrics = {"pragmatics": pragmatics}


    # LEXSEM
    speaker_obs, _, _, _ = gen_batch(dataset, batch_size, fieldname, p_notseedist=1, vae=vae_model, see_distractors=settings.see_distractor, glove_data=glove_data, num_dist=num_distractors)

    EC_words = []
    for i in range(num_samples):
        tmp = []
        for x in speaker_obs:
            # we obtain a one-hot vector, that is the distribution over prototypes
            # 1 is at the index of the closest prototype
            likelihood, _ = team.speaker.get_token_dist(x)
            # we take th index of the closest prototype
            index_of_one = torch.argmax(torch.tensor(likelihood)).item()
            tmp.append(index_of_one)
        EC_words.append(tmp)

    count_trials = [len(set(i)) for i in EC_words]

    frequencies = [Counter(i) for i in EC_words]
    total_symbols = [len(i) for i in EC_words]
    probabilities_list = []
    for f,tot in zip(frequencies, total_symbols):
        tmp = []
        for c in f.values():
            tmp.append(c/tot)
        probabilities_list.append(tmp)
    entropies = [entropy(i, base=2) for i in probabilities_list]

    print("lexsem - counts:", count_trials)
    print("lexsem - average word count:", sum(count_trials)/len(count_trials))
    print("lexsem - entropies:", entropies)
    print("lexsem - average entropy:", sum(entropies) / len(entropies))


    lexsem = {"word counts": count_trials, 
              "average word count": sum(count_trials)/len(count_trials),
              "entropies": entropies,
              "average entropy": sum(entropies) / len(entropies)}
    
    metrics["lexsem"] = lexsem

    
    # SAVE STUFF
    dic_tmp = {"inf_weight"+str(alpha): metrics}

    if "utility"+str(utility) in dic.keys():
        dic["utility"+str(utility)].update(dic_tmp)
    else:
        dic["utility"+str(utility)] = dic_tmp

    with open(json_file, 'w') as f:
        json.dump(dic, f, indent=4)





def run():
    if settings.see_distractors_pragmatics:
        num_imgs = 3 if settings.with_ctx_representation else 2
    else:
        num_imgs = 1 if not settings.see_distractor else (num_distractors + 1)

    random_init_dir = "random_init/" if settings.random_init else "anneal/"

    if not settings.eval_someRE:
        data = get_feature_data(t_features_filename, excluded_ids=excluded_ids)
        data = data.sample(frac=1, random_state=seed) # Shuffle the data.
        train_data, test_data, val_data = np.split(data.sample(frac=1, random_state=seed),
                                        [int(.7*len(data)), int(.9*len(data))])
        train_data, test_data, val_data = train_data.reset_index(), test_data.reset_index(), val_data.reset_index()
        train_data, test_data, val_data = train_data.sample(frac=1, random_state=seed).reset_index(), test_data.sample(frac=1, random_state=seed).reset_index(), val_data.sample(frac=1, random_state=seed).reset_index()
        print("len data:", len(train_data))
    else:
        random_init_dir = "random_init/" if settings.random_init else ""
        someRE_data = get_feature_data(t_features_filename, excluded_ids=excluded_ids)
        someRE_data = someRE_data.sample(frac=1) # Shuffle the data.
        topnames_someRE = []
        for i in someRE_data['link_vg']:
            responses_someRE = someRE_prod.loc[someRE_prod['link_vg'] == i, 'responses_someRE'].tolist()[0]
            resp = ast.literal_eval(responses_someRE)
            topnames_someRE.append(list(resp.keys())[0])
        someRE_data['topname_someRE'] = topnames_someRE
        someRE_data = someRE_data.reset_index()
        print("len data:", len(someRE_data))


    speaker = VQ(feature_len, c_dim, num_layers=3, num_protos=settings.num_protos, num_simultaneous_tokens=1, variational=variational, num_imgs=num_imgs)
    listener = ListenerPragmaticsCosines(feature_len)
    decoder = Decoder(c_dim, feature_len, num_layers=3, num_imgs=num_imgs)
    model = Team(speaker, listener, decoder)

    folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
    models_loc = 'src/saved_models/' + str(settings.num_protos) + "/random_init/"+ folder_ctx + 'missing/seed' + str(seed) + '/'

    json_file_path = "src/saved_models/" + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight1.0/seed' + str(seed) + '/'
    json_file = json_file_path+"done_weights2.json"
    with open(json_file, 'r') as f:
        triplets = json.load(f)
    done_triplets = [ast.literal_eval(i) for i in list(triplets.values())][80:]

    for t in done_triplets:
        alpha = t[1]
        complexity = t[0]
        utility = t[2]
        settings.kl_weight = complexity

        folder_utility = "utility"+str(utility)+"/"
        folder_alpha = "alpha"+str(alpha)+"/"
        folder_compl = "compl"+str(complexity)+"/"

        convergence_epoch = 4999
        
        # load model
        model_to_eval_path = models_loc + folder_utility + folder_alpha + folder_compl + str(convergence_epoch)
        model.load_state_dict(torch.load(model_to_eval_path + '/model.pt'))
        model.to(settings.device)

        json_file_ECcount = "src/saved_models/" + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight1.0/seed' + str(seed) + '/word_counts_missing' + str(settings.job_num) + '.json'
#        json_file_ECdistr = "src/saved_models/" + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(seed) + '/word_counts_thresh_' + str(settings.threshold) + '_' + str(settings.job_num) + '.json'
        
        get_ec_words(val_data, model, len(val_data), 10, utility, alpha, json_file_ECcount)


            

if __name__ == '__main__':
    feature_len = 512
    settings.see_distractor = False # MT setting
    settings.see_distractors_pragmatics = True # EG setting

    settings.with_ctx_representation = False
    settings.dropout = False
    settings.see_probabilities = True # this means that we are masking with a complete mask, no dropout

    settings.random_init = True
    settings.eval_someRE = False # this controls the data reading: we don't shuffle target and distractor

    settings.num_protos = 3000 #442
    settings.threshold = 0.001
    settings.job_num = 4

    settings.kl_weight = 1.0 # complexity  
    settings.kl_incr = 0.0

    num_distractors = 1
    settings.num_distractors = num_distractors
    v_period = 100  # How often to test on the validation set and calculate various info metrics.
    num_burnin = 500
    b_size = 1024
    c_dim = 128
    variational = True
    do_calc_complexity = False
    do_plot_comms = False
    fieldname = 'topname'

    settings.entropy_weight = 0.0
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.learned_marginal = False
    settings.max_num_align_data = 1
    settings.distinct_words = False  # FIXME
    with_bbox = False
    image_directory = 'src/data/images_nobox/'

    t_features_filename = 'src/data/t_features.csv'
    settings.d_features_filename = 'src/data/d_features.csv'
    settings.d_bboxes_filename = 'src/data/d_xyxy.tsv'
    settings.ctx_features_filename = 'src/data/ctx_features.csv'
    manynames = load_cleaned_results(filename="src/data/manynames.tsv")
    someRE = pd.read_csv("src/data/someRE.csv", sep = ";")
    if not settings.eval_someRE:
        merged_tmp = pd.merge(manynames, someRE, on=['link_vg'])
        excluded_ids = [i for i in merged_tmp['vg_image_id']]
    
    else:
        # open someRE and save topname
        #someRE = someRE[(someRE['image_name'].str.contains("ambi_spec")) | (someRE['image_name'].str.contains("unam"))]
        someRE = someRE[someRE['image_name'].str.contains("ambi_spec")]
        print("ambi spec:", len(someRE))
        #someRE = someRE[someRE['image_name'].str.contains("unam")]
        #print("unam:", len(someRE))
        someRE_prod = pd.read_table("src/data/someRE_production_annotations.tsv", sep=",")
        someRE_links = [i for i in someRE['link_vg']]
        someRE_ids = manynames[manynames.link_vg.isin(someRE_links)]['vg_image_id']
        excluded_ids = [str(i) for i in manynames['vg_image_id'] if i not in someRE_ids.tolist()]

    print("ids to exclude:", len(excluded_ids))

    vae_model = VAE(512, 32)
    vae_model.load_state_dict(torch.load('src/saved_models/vae0.001.pt'))
    vae_model.to(settings.device)
    settings.embedding_cache = {}
    settings.sample_first = True
    speaker_type = 'vq'  # Options are 'vq', 'cont', or 'onehot'

    seed = 0 # 0,1,2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    glove_data = get_glove_vectors(32)
    run()



