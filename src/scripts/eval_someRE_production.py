import ast
import os
import json
import pandas as pd
import pickle
import random
from scipy import stats

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageDraw

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
from src.utils.mine_pragmatics import get_info, get_cond_info


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

    metrics = {"lexsem": lexsem}


    # SAVE STUFF
    dic_tmp = {"inf_weight"+str(alpha): metrics}

    if "utility"+str(utility) in dic.keys():
        dic["utility"+str(utility)].update(dic_tmp)
    else:
        dic["utility"+str(utility)] = dic_tmp

    with open(json_file, 'w') as f:
        json.dump(dic, f, indent=4)



# Given a model and a set of anchors, compute the relative encoding position of lots of communication vectors?
def get_relative_embedding(eval_type, model, vae, anchor_dataset, glove_data, fieldname):
    # First, compute the anchorsi
    num_anchors = 100
    count = 0
    model_anchors = []
    glove_anchors = []
    itr_count = -1
    for targ_features, distractor_features, ctx_features, word in zip(anchor_dataset['t_features'], anchor_dataset['d_features'], anchor_dataset['ctx_features'], anchor_dataset[fieldname]):
        itr_count += 1
        if settings.with_ctx_representation:
            s_obs = np.expand_dims(np.vstack([targ_features] + [distractor_features] + [ctx_features]), axis=0)
        else:
            s_obs = np.expand_dims(np.vstack([targ_features] + [distractor_features]), axis=0)

        speaker_tensor = torch.Tensor(np.vstack(s_obs).astype(np.float)).to(settings.device)
        if vae is not None:
            with torch.no_grad():
                speaker_tensor, _ = vae(speaker_tensor)
        speaker_tensor = speaker_tensor.unsqueeze(0)

        if eval_type == "lexsem":
            # Add mask over distractor
            # Create a mask of the same shape as the tensor
            mask = torch.ones_like(speaker_tensor, dtype=bool)
            # Generate indices to apply the mask
            num_tensors = speaker_tensor.shape[0]
            indices = np.random.choice(num_tensors, int(num_tensors * 1), replace=False)
            # Apply the mask to the selected indices
            mask[indices, 1, :] = False
            # Apply the mask to the tensor
            speaker_obs = torch.where(mask, speaker_tensor, torch.tensor(0.0, device=settings.device))
        elif eval_type == "pragmatics":
            speaker_obs = speaker_tensor

        #speaker_obs = torch.Tensor(np.array(f)).to(settings.device)
        #speaker_obs = torch.unsqueeze(speaker_obs, 0)
        with torch.no_grad():
            #speaker_obs = speaker_obs.repeat(1, 1)
            comm, _, _ = model.speaker(speaker_obs)
        try:
            if fieldname == 'responses':
                responses = word  # Gross, but true, I think.
                words = []
                probs = []
                for k, v in responses.items():
                    parsed_word = k.split(' ')
                    if len(parsed_word) > 1:
                        # Skip "words" like "tennis player" etc. because they won't be in glove data
                        continue
                    words.append(k)
                    probs.append(v)
                if len(words) == 0:
                    # Failed to find any legal words (e.g., all like "tennis player")
                    continue
                total = np.sum(probs)
                probs = [p / total for p in probs]
                word = np.random.choice(words, p=probs)
            embedding = get_glove_embedding(glove_data, word).to_numpy()
        except AttributeError:
            continue
        np_comm = np.mean(comm.detach().cpu().numpy(), axis=0)
        model_anchors.append(np_comm)
        glove_anchors.append(embedding)
        count += 1
        if count >= num_anchors:
            break
    # print("Got our anchors")
    model_anchors = np.array(model_anchors)
    glove_anchors = np.array(glove_anchors)
    def get_relative(emb, anchors, cosine_based=False):
        # Cosine similarity
        if cosine_based:
            # For cosine, we use *negative* cosine similarity, because less alignment is further apart.
            relative_emb = -np.array([np.dot(emb, anchor) / (np.linalg.norm(emb) * np.linalg.norm(anchor)) for anchor in anchors])
        else: # Instead of being cosine based, do euclidean?
            relative_emb = np.array([np.linalg.norm(emb.squeeze(0) - anchor) for anchor in anchors])

        # Just sanity check by iterating
        # emb is 1 x 64
        # anchors is 100 x 64
        relatives = []
        for anchor in anchors:
            norm1 = np.linalg.norm(emb)
            norm2 = np.linalg.norm(anchor)
            relative = np.dot(emb.squeeze(0), anchor) / (norm1 * norm2)
            relatives.append(relative)
        return np.transpose(relative_emb)
    ec_rel = []
    glove_rel = []
    for idx in range(len(model_anchors)):
        rel_ec = get_relative(np.expand_dims(model_anchors[idx], 0), model_anchors)
        rel_glove = get_relative(np.expand_dims(glove_anchors[idx], 0), glove_anchors, cosine_based=True).squeeze(0)
        ec_rel.append(rel_ec)
        glove_rel.append(rel_glove)
    # Now do the spearman, flatten, etc.
    res = stats.spearmanr(np.hstack(ec_rel), np.hstack(glove_rel))
    return res.correlation




def run():
    if settings.see_distractors_pragmatics:
        num_imgs = 3 if settings.with_ctx_representation else 2
    else:
        num_imgs = 1 if not settings.see_distractor else (num_distractors + 1)

    random_init_dir = "random_init/" if settings.random_init else ""
    someRE_data = get_feature_data(t_features_filename, excluded_ids=excluded_ids)
    someRE_data = someRE_data.sample(frac=1) # Shuffle the data.
    topnames_someRE = []
    for i in someRE_data['link_vg']:
        responses_someRE = someRE_prod.loc[someRE_prod['link_vg'] == i, 'responses_someRE'].tolist()[0]
        resp = ast.literal_eval(responses_someRE)
        topnames_someRE.append(list(resp.keys())[0])
    someRE_data['topname_someRE'] = topnames_someRE

    #someRE_data = someRE_data.loc[someRE_data['swapped_t_d'] == 0]
    #someRE_data = someRE_data.reset_index(drop=True)
    #print("Len val set non swapped:", len(val_data)) # this should be 0

    speaker = VQ(feature_len, c_dim, num_layers=3, num_protos=settings.num_protos, num_simultaneous_tokens=1, variational=variational, num_imgs=num_imgs)
    listener = ListenerPragmaticsCosines(feature_len)
    decoder = Decoder(c_dim, feature_len, num_layers=3, num_imgs=num_imgs)
    model = Team(speaker, listener, decoder)

    for u in settings.utilities:
        print("______________")
        print("utility:", u)
        print("______________")

        for a in settings.alphas:

            folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
            folder_utility = "utility"+str(u)+"/"
            folder_alpha = "alpha"+str(a)+"/"
            print("alpha:", a)

            json_file_path = "src/saved_models/" + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(my_seed) + '/'
            json_file = json_file_path+"objective_merged.json"
            with open(json_file, 'r') as f:
                existing_params = json.load(f)
            convergence_epoch = existing_params["utility"+str(u)]["inf_weight"+str(a)]['convergence epoch']
        
            # load model
            model_to_eval_path = 'src/saved_models/' + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(my_seed) + '/' + folder_utility + folder_alpha + str(convergence_epoch)
            
            model.load_state_dict(torch.load(model_to_eval_path + '/model.pt'))
            model.to(settings.device)
            model.eval()

            #alignment_datasets = [get_rand_entries(someRE_data, num_examples) for _ in range(num_rand_trials)]
           
            # check alignment in lexsem mode
            #consistency_scores = []
            #for j, align_data in enumerate(alignment_datasets):
            #    consistency_score = get_relative_embedding("lexsem", model, vae_model, align_data, glove_data, fieldname='topname')
            #    consistency_scores.append(consistency_score)
            #print("r LEXSEM mode on LEXSEM data:", sum(consistency_scores) / len(consistency_scores))
        
            # check alignment in lexsem mode
            #consistency_scores = []
            #for j, align_data in enumerate(alignment_datasets):
            #    consistency_score = get_relative_embedding("lexsem", model, vae_model, align_data, glove_data, fieldname='topname_someRE')
            #    consistency_scores.append(consistency_score)
            #print("r LEXSEM mode on PRAG data:", sum(consistency_scores) / len(consistency_scores))

            # check alignment in pragmatic mode
            #consistency_scores = []
            #for j, align_data in enumerate(alignment_datasets):
            #    consistency_score = get_relative_embedding("pragmatics", model, vae_model, align_data, glove_data, fieldname='topname')
            #    consistency_scores.append(consistency_score)
            #print("r PRAG mode on LEXSEM data:", sum(consistency_scores) / len(consistency_scores))
            
            # check alignment in pragmatic mode
            #consistency_scores = []
            #for j, align_data in enumerate(alignment_datasets):
            #    consistency_score = get_relative_embedding("pragmatics", model, vae_model, align_data, glove_data, fieldname='topname_someRE')
            #    consistency_scores.append(consistency_score)
            #print("r PRAG mode on PRAG data:", sum(consistency_scores) / len(consistency_scores))
            
            json_file_ECcount = "src/saved_models/" + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(my_seed) + '/word_counts_someRE' + '.json'
            get_ec_words(someRE_data, model, len(someRE_data), 10, u, a, json_file_ECcount)


if __name__ == '__main__':
    feature_len = 512
    settings.see_distractor = False # MT setting
    settings.see_distractors_pragmatics = True # EG setting

    settings.with_ctx_representation = False
    settings.dropout = False
    settings.see_probabilities = True # this means that we are masking with a complete mask, no dropout
   
    settings.random_init = True
    settings.eval_someRE = True # this controls the data reading: we don't shuffle target and distractor
    
    num_distractors = 1
    settings.num_distractors = num_distractors
    n_epochs = 3000
    v_period = 100  # How often to test on the validation set and calculate various info metrics.
    num_burnin = 500
    b_size = 1024
    c_dim = 128
    variational = True
    # Measuring complexity takes a lot of time. For debugging other features, set to false.
    do_calc_complexity = False
    do_plot_comms = False
    
    settings.num_protos = 3000 #442
    #settings.alphas = [0, 0.1, 1.5, 21, 140, 200]  # informativeness
    #settings.utilities = [0, 0.1, 1.5, 21, 140, 200] # utility
    settings.alphas = [0, 1, 7]  # informativeness
    settings.utilities = [0, 1, 7] # utility
    settings.kl_weight = 1.0 # complexity  
    settings.kl_incr = 0.0
    
    settings.entropy_weight = 0.0
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.learned_marginal = False
    settings.max_num_align_data = 1
    settings.distinct_words = False  # FIXME
    with_bbox = False
    image_directory = 'src/data/images_nobox/'

    num_examples = 24
    num_rand_trials = 5 # to regulate get_relative_embeddings

    t_features_filename = 'src/data/t_features.csv'
    settings.d_features_filename = 'src/data/d_features.csv'
    settings.d_bboxes_filename = 'src/data/d_xyxy.tsv'
    settings.ctx_features_filename = 'src/data/ctx_features.csv'
    manynames = load_cleaned_results(filename="src/data/manynames.tsv")
    print("manynames:", len(manynames))

    # open someRE and save topname
    someRE = pd.read_csv("src/data/someRE.csv", sep = ";")
    #someRE = someRE[(someRE['image_name'].str.contains("ambi_spec")) | (someRE['image_name'].str.contains("unam"))]
    someRE = someRE[someRE['image_name'].str.contains("ambi_spec")]
    print("ambi spec:", len(someRE))

    someRE_prod = pd.read_table("src/data/someRE_production_annotations.tsv", sep=",")
    someRE_links = [i for i in someRE['link_vg']]
    someRE_ids = manynames[manynames.link_vg.isin(someRE_links)]['vg_image_id']
    excluded_ids = [str(i) for i in manynames['vg_image_id'] if i not in someRE_ids.tolist()]
 
    print("ids to exclude:", len(excluded_ids))

    vae_model = VAE(512, 32)
    vae_model.load_state_dict(torch.load('src/saved_models/vae0.001.pt'))
    vae_model.to(settings.device)
    np.random.seed(0)
    settings.embedding_cache = {}
    settings.sample_first = True
    speaker_type = 'vq'  # Options are 'vq', 'cont', or 'onehot'

    my_seed = 0
    random.seed(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    
    glove_data = get_glove_vectors(32)
    fieldname = "topname"
    viz_topname = None

    run()


