import os
import ast
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import manifold
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
    
    # find synonym prototypes
    prototypes = model.speaker.vq_layer.prototypes.detach().cpu()
    proto_synonyms = find_synonyms(prototypes, 0.1)
    print("prototypes' clusters found:", len(proto_synonyms))

    # initialize dictionary to store human names probabilities
    human_names = []
    for i in list(data['responses']):
        for j in i.keys():
            human_names.append(j)
    human_names = list(set(human_names))
    human_probs = {key: [] for key in human_names}
    ids_to_names = dict(enumerate(human_names))
    names_to_ids = {v:k for k,v in ids_to_names.items()}
    
    # initialize dictionary for model names
    model_probs = {key: [] for key in range(settings.num_protos)}
    vg_ids = {}
    
    counter = 0

    for targ_idx in list(data.index.values):

        speaker_obs, _, _, _ = gen_batch(data, 1, "topname", p_notseedist=1, glove_data=glove_data, vae=vae, preset_targ_idx=targ_idx)
        
        if speaker_obs != None: # i.e. we have the glove embeds
            
            responses = data['responses'][targ_idx]
            total = sum(list(responses.values()))
            normalized_responses = {key: value/total for key, value in responses.items()}
            for k,v in human_probs.items():
                if k in normalized_responses.keys():
                    human_probs[k].append(normalized_responses[k])
                else:
                    human_probs[k].append(0.0)

            id_ = data['vg_image_id'][targ_idx]
            vg_ids[counter] = id_
            counter += 1

            # we repeat the input to get a sense of the topname
            speaker_obs = speaker_obs.repeat(100, 1, 1)
            
            with torch.no_grad():
                likelihood = model.speaker.get_token_dist(speaker_obs)
                for idx in range(len(likelihood)):
                    model_probs[idx].append(likelihood[idx])

        else:
            pass
    
    # MODEL MATRIX
    
    model_matrix = np.empty([len(model_probs[0]), settings.num_protos])
    
    for i in range(settings.num_protos):
        model_matrix[:, i] = model_probs[i]
    
    # remove synomym columns
    for key, columns in proto_synonyms.items():
        model_matrix[:, key] = np.sum(model_matrix[:, columns], axis=1)
    
    columns_to_remove = set()
    for key, columns in proto_synonyms.items():
        columns_to_remove.update([c for c in columns if c != key])

    columns_to_keep = [c for c in range(model_matrix.shape[1]) if c not in columns_to_remove]
    model_matrix = model_matrix[:, columns_to_keep]
    
    # identify columns that are not all zeros
    non_zero_columns = np.any(model_matrix != 0, axis=0)
    model_matrix = model_matrix[:, non_zero_columns]

    # make it a real matrix
    model_matrix = np.matrix(model_matrix)

    M = model_matrix.shape[0]
    W = model_matrix.shape[1]
    
    # HUMAN MATRIX
    
    human_matrix = np.empty([len(list(human_probs.values())[0]), len(human_names)])
    for i in range(len(human_names)):
        human_matrix[:, i] = human_probs[ids_to_names[i]]
    
    # identify columns that are not all zeros
    #non_zero_columns = np.any(human_matrix != 0, axis=0)
    #human_matrix = human_matrix[:, non_zero_columns]

    human_matrix = np.matrix(human_matrix)
    
    M_human = human_matrix.shape[0]
    W_human = human_matrix.shape[1]

    return M, W, model_matrix, M_human, W_human, human_matrix, ids_to_names, vg_ids



def p_W(M, W, model, p_W_I):
    
    p_image = 1/M
   
    p_words = []
    for i in range(W):
        p_words.append(p_image * sum(p_W_I[:, i]))
    p_words = np.array(p_words).reshape(1, W)

    return p_words, p_image



def p_I_W(p_W_I, p_word, p_image):
     
    return (p_W_I * p_image) / p_word 




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
    utility = 0
    
    folder_utility = "utility" + str(utility) + "/"
    folder_alpha = "alpha" + str(informativeness) + "/"
    folder_complexity = "compl" + str(complexity) + "/"
    
    print(random_init_dir)
    model_to_eval_path = 'src/saved_models/' + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight' + str(complexity) + '/' + 'seed0/' + folder_utility + folder_alpha + "4999/"
    #model_to_eval_path = 'src/saved_models/' + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'missing/seed0/' + folder_utility + folder_alpha + folder_complexity + "4999/"
    model.load_state_dict(torch.load(model_to_eval_path + '/model.pt'))
    model.to(settings.device)
    model.eval()

    M, W, model_p_word_image, M_human, W_human, human_p_word_image, ids_to_names, vg_ids = p_W_I(data, model, vae=vae, glove_data=glove_data)
    print("p_W_I:", model_p_word_image.shape)
    #print(np.sum(model_p_word_image, axis=1))
    print("p_W_I:", human_p_word_image.shape)
    #print(np.sum(human_p_word_image, axis=1))

    #model_p_word, model_p_image = p_W(M, W, model, model_p_word_image)
    #print("model p_word:", model_p_word.shape)
    human_p_word, human_p_image = p_W(M_human, W_human, model, human_p_word_image)
    print("human p_word:", human_p_word.shape)
    print(np.sum(human_p_word, axis=1))

    most_prob_human_words = np.argsort(human_p_word.squeeze())[-10:]
    print([ids_to_names[i] for i in most_prob_human_words])

    human_p_image_word = p_I_W(human_p_word_image, human_p_word, human_p_image)
    print("p_I_W:", human_p_image_word.shape)
    print(np.sum(human_p_image_word, axis=0))


    most_prob_images_per_word_humans = []
    for i in most_prob_human_words:
        most_prob_images_per_word_humans.append(np.argsort(human_p_image_word[:, i].squeeze()).A1[-100:].tolist())

    vectors, human_classes, EC_classes, VG_id_list = [], [], [], []
    for n, ids in zip(most_prob_human_words, most_prob_images_per_word_humans):
        for j in ids:
            vg_image_id = vg_ids[j]
            VG_id_list.append(vg_image_id)
            vectors.append(data.loc[data['vg_image_id'] == vg_image_id, 't_features'].tolist()[0])
            human_classes.append(ids_to_names[n])
            most_prob_EC = np.argmax(model_p_word_image[j, :].squeeze()).tolist()        
            EC_classes.append(most_prob_EC)


    #p_image_word = p_I_W(p_word_image, p_word, p_image)
    #print("p_I_W:", p_image_word.shape)
    #print(np.sum(p_image_word, axis=0))
    #most_prob_words = np.argsort(p_word.squeeze())[-10:]
    #most_prob_images_per_word = []
    #for i in most_prob_words:
    #    most_prob_images_per_word.append(np.argsort(p_image_word[:, i].squeeze()).A1[-100:].tolist())
    
        
    #vectors, human_classes, EC_classes = [], [], []
    #for num, i in enumerate(most_prob_images_per_word):
    #    print(i)
    #    for j in i:
    #        vg_image_id = vg_ids[j]
    #        vectors.append(data.loc[data['vg_image_id'] == vg_image_id, 't_features'].tolist()[0])
    #        n = data.loc[data['vg_image_id'] == vg_image_id, 'topname'].tolist()[0]
    #        human_classes.append(n)
    #        EC_classes.append(num)
    #        print(n)
    
    
    unique_labels = list(set(human_classes))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    human_classes_num = np.array([label_to_int[label] for label in human_classes])
    print(unique_labels)
    print(label_to_int)
    
    unique_labels_EC = list(set(EC_classes))
    label_to_int_EC = {label: i for i, label in enumerate(unique_labels_EC)}
    EC_classes_num = np.array([label_to_int_EC[label] for label in EC_classes])


    # PCA

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(vectors)
    print(list(zip(VG_id_list, X_pca, human_classes)))

    #fig, axs = plt.subplots(1, 2, figsize=(26, 8)) 

    #axs[0] = fig.add_subplot(1, 2, 1, projection='3d')
    #axs[1] = fig.add_subplot(1, 2, 2, projection='3d')
    
    #for ax in axs:
    #    box = ax.get_position()
    #    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height * 0.6])

    #for ax in axs:
    #    ax.xaxis.labelpad = 15  # Adjust the value to suit your needs
    #    ax.yaxis.labelpad = 15
    #    ax.zaxis.labelpad = 15


    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.xaxis.labelpad = 15  
    ax.yaxis.labelpad = 15
    ax.zaxis.labelpad = 15
    
    title_fontsize = 24
    label_fontsize = 16
    legend_fontsize = 18

    XX = [i[0] for i in X_pca]
    XY = [i[1] for i in X_pca]
    XZ = [i[2] for i in X_pca]
    
    scatter1 = ax.scatter(XX, XY, XZ, alpha=0.8, s=25, c=EC_classes_num, cmap='hsv')
    ax.set_xlabel('PC 1', fontsize=label_fontsize)
    ax.set_ylabel('PC 2', fontsize=label_fontsize)
    ax.set_zlabel('PC 3', fontsize=label_fontsize)
    #ax.set_title('EC names', fontsize=title_fontsize)

    #legend = ax.legend(handles=scatter1.legend_elements()[0], labels=unique_labels_EC, title="", fontsize=legend_fontsize)

    plt.tight_layout()

    plt.savefig(f'Plots/3000/random_init/EC_comm/PCA_EC_comm_u{utility}_a{informativeness}_c{complexity}.png', dpi=300)

    plt.show()


    # human
    
    fig = plt.figure(figsize=(15, 8))
    
    ax = fig.add_subplot(111, projection='3d')

    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.zaxis.labelpad = 15

    title_fontsize = 24
    label_fontsize = 16
    legend_fontsize = 18

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height * 0.7])

    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.zaxis.labelpad = 15

    scatter2 = ax.scatter(XX, XY, XZ, alpha=0.8, s=25, c=human_classes_num, cmap='hsv')
    ax.set_xlabel('PC 1', fontsize=label_fontsize)
    ax.set_ylabel('PC 2', fontsize=label_fontsize)
    ax.set_zlabel('PC 3', fontsize=label_fontsize)
    #ax.set_title('Human names', fontsize=title_fontsize)

    
    legend = ax.legend(handles=scatter2.legend_elements()[0], labels=unique_labels, title="", loc='center left', bbox_to_anchor=(1.2, 0.5), fontsize=legend_fontsize)
    
    plt.tight_layout()

    plt.savefig(f'Plots/3000/random_init/EC_comm/PCA_human_u{utility}_a{informativeness}_c{complexity}.png', dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')

    plt.show()

    #scatter1 = axs[0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.8, s=25, c=EC_classes, cmap='viridis')
    #scatter1 = axs[0].scatter(XX, XY, XZ, alpha=0.8, s=25, c=EC_classes, cmap='hsv')
    #axs[0].set_xlabel('PC 1', fontsize=label_fontsize)
    #axs[0].set_ylabel('PC 2', fontsize=label_fontsize)
    #axs[0].set_zlabel('PC 3', fontsize=label_fontsize)
    #axs[0].set_title('EC names', fontsize=title_fontsize)

    #scatter2 = axs[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.8, s=25, c=human_classes_num, cmap='viridis')
    #scatter2 = axs[1].scatter(XX, XY, zs=XZ, alpha=0.8, s=25, c=human_classes_num, cmap='hsv')
    #axs[1].set_xlabel('PC 1', fontsize=label_fontsize)
    #axs[1].set_ylabel('PC 2', fontsize=label_fontsize)
    #axs[1].set_zlabel('PC 3', fontsize=label_fontsize)
    #axs[1].set_title('Human names', fontsize=title_fontsize)

    #legend = axs[1].legend(handles=scatter2.legend_elements()[0], labels=unique_labels, title="", loc='center left', bbox_to_anchor=(1, 0.5), fontsize=legend_fontsize)

    #plt.subplots_adjust(wspace=0.3)  

    #plt.tight_layout()

    #plt.savefig(f'Plots/3000/random_init/EC_comm/PCA_EC_comm_u{utility}_a{informativeness}_c{complexity}.png', dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')

    #plt.show()


    #nMDS
    
    #cos_similarities = cosine_similarity(vectors)
    #dissimilarities = 1 - cos_similarities
    
    #nmds = manifold.MDS(
    #    n_components=2,
    #    metric=False,  # Indicates non-metric MDS
    #    max_iter=1000,  # Adjust based on convergence
    #    eps=1e-12,  # Tighter tolerance since nMDS can be more sensitive to this parameter
    #    dissimilarity="precomputed",
    #    random_state=seed,  # Define your seed for reproducibility
    #    n_jobs=-1,  # Use all CPU cores for parallel computation
    #    n_init=1,  # Number of times the algorithm will be run with different initializations
    #)
    #X_nmds = nmds.fit_transform(dissimilarities)


    #fig, axs = plt.subplots(1, 2, figsize=(23, 8))

    #for ax in axs:
    #    box = ax.get_position()
    #    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height * 0.6])

    #title_fontsize = 24
    #label_fontsize = 18
    #legend_fontsize = 20

    #scatter1 = axs[0].scatter(X_nmds[:, 0], X_nmds[:, 1], alpha=0.8, s=25, c=EC_classes, cmap='viridis')
    #axs[0].set_xlabel('Dimension 1', fontsize=label_fontsize)
    #axs[0].set_ylabel('Dimension 2', fontsize=label_fontsize)
    #axs[0].set_title('EC names', fontsize=title_fontsize)

    #scatter2 = axs[1].scatter(X_nmds[:, 0], X_nmds[:, 1], alpha=0.8, s=25, c=human_classes_num, cmap='viridis')
    #axs[1].set_xlabel('Dimension 1', fontsize=label_fontsize)
    #axs[1].set_ylabel('Dimension 2', fontsize=label_fontsize)
    #axs[1].set_title('Human names', fontsize=title_fontsize)

    #legend = axs[1].legend(handles=scatter2.legend_elements()[0], labels=unique_labels, title="", loc='center left', bbox_to_anchor=(1, 0.5), fontsize=legend_fontsize)

    #plt.subplots_adjust(wspace=0.3)  # Adjust this value as needed to create the desired gap
    #plt.tight_layout()
    
    #plt.savefig(f'Plots/3000/random_init/EC_comm/nMDS_EC_comm_u{utility}_a{informativeness}_c{complexity}.png', dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
    
    #plt.show()



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

    settings.random_init = True
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

