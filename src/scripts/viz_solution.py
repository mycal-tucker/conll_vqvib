import ast
import os
import json
import pandas as pd
import pickle
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
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


def download_images_per_vec(eval_type, save_dir, vector, dict_vect_imgs, img_infos, target, include_target=False):

    #save_dir_ex = save_dir + "example"+str(ex) + "/"
    #save_dir_vec = save_dir_ex + str(vector) + "/"
    save_dir_eval = save_dir + eval_type +  "/"
    if not os.path.exists(save_dir_eval):
        os.makedirs(save_dir_eval)
    save_dir_vec = save_dir_eval + str(vector) + "/"
    if not os.path.exists(save_dir_vec):
        os.makedirs(save_dir_vec)
    
    try:
        sampled_images = random.sample(dict_vect_imgs[vector], 10)
        if include_target:
            sampled_images.append(target)
        sampled_info = [img_infos[i] for i in sampled_images]
    
        targets,distractors = [], []
        for tar_xyxy, dist_xyxy, tar_topname, is_swapped in sampled_info:
            if is_swapped == 1:
                distractors.append(tar_xyxy)
                targets.append(ast.literal_eval(dist_xyxy)) 
            else:
                distractors.append(ast.literal_eval(dist_xyxy))
                targets.append(tar_xyxy)

        images = [Image.open(image_directory + i + ".jpg") for i in sampled_images]

        for n,img,id_ in zip(range(len(images)), images, sampled_images):
            # draw target
            draw = ImageDraw.Draw(img)
            red =  (255, 0, 0)
            t_coord = [(targets[n][0], targets[n][1]), (targets[n][2], targets[n][3])]
            draw.rectangle(t_coord, outline=red, width=4)
                
            if eval_type == "pragmatics":
                # draw distractors
                draw = ImageDraw.Draw(img)
                blue =  (0, 0, 255)
                d_coord = [(distractors[n][0], distractors[n][1]), (distractors[n][2], distractors[n][3])]
                draw.rectangle(d_coord, outline=blue, width=6)

            img.save(save_dir_vec + "/" + id_ + ".png","PNG")
    
    except ValueError:
        pass


def visualize_solution(dataset, team, num_examples, save_path_vecs, save_path_grids, sampled_id):
    
    # PROTOTYPES' SPACE
    prototypes = team.speaker.vq_layer.prototypes.detach().cpu()
    distance_matrix = euclidean_distances(prototypes)
    similarity_dict = {} # we store the most similar prototypes (or in this case, the closest prototypes)
    for i in range(len(prototypes)):
        # The closest prototype has the smallest distance
        sorted_indices = np.argsort(distance_matrix[i])
        sorted_indices = sorted_indices[sorted_indices != i]  # exclude self-distance
        similarity_dict[i] = sorted_indices.tolist()

    # LEXICAL-SEMANTICS
    ids_to_comms = {} # image ids to comm vectors
    img_infos_lexsem = {} # images info
    ids_to_protos = {} # images ids to index of prototype sent
    EC_words_lexsem = {i: [] for i in range(3000)} # keys are prototypes, values are images' indeces
    for targ_idx in list(dataset.index.values):
        vg_image_id = dataset['vg_image_id'][targ_idx]
        tar_xywh = dataset['bbox_xywh'][targ_idx]
        tar_xyxy = [tar_xywh[0], tar_xywh[1], tar_xywh[0]+tar_xywh[2], tar_xywh[1]+tar_xywh[3]]
        dist_xyxy = dataset['dist_xyxy'][targ_idx]
        tar_topname = dataset['topname'][targ_idx]
        is_swapped = dataset['swapped_t_d'][targ_idx]
        img_infos_lexsem[vg_image_id] = [tar_xyxy, dist_xyxy, tar_topname, is_swapped]

        speaker_obs, _, _, _ = gen_batch(dataset, 1, fieldname, p_notseedist=1, vae=vae_model, see_distractors=settings.see_distractor, glove_data=glove_data, num_dist=num_distractors, preset_targ_idx=targ_idx)
        
        if speaker_obs != None:
            # Now get the EC for that speaker obs
            with torch.no_grad():
                comm, _, _ = team.speaker(speaker_obs)
                np_comm = comm.detach().cpu().numpy()
                ids_to_comms[vg_image_id] = np_comm[0]
                likelihood, _ = team.speaker.get_token_dist(speaker_obs)
                index_of_one = torch.argmax(torch.tensor(likelihood)).item()
                ids_to_protos[vg_image_id] = index_of_one
                EC_words_lexsem[index_of_one].append(vg_image_id)
        else : # there is no Glove embedding for that preset_targ_idx
            pass
 
    if not os.path.exists(save_path_vecs):
        os.makedirs(save_path_vecs)

    print("EC word used lexsem:", ids_to_protos[sampled_id])   
    vector_lex = ids_to_protos[sampled_id]
    
    
    # PRAGMATICS
    ids_to_comms = {} # image ids to comm vectors
    img_infos_prag = {} # images info
    ids_to_protos = {} # images ids to index of prototype sent
    EC_words_prag = {i: [] for i in range(3000)}
    for targ_idx in list(dataset.index.values):
        vg_image_id = dataset['vg_image_id'][targ_idx]
        tar_xywh = dataset['bbox_xywh'][targ_idx]
        tar_xyxy = [tar_xywh[0], tar_xywh[1], tar_xywh[0]+tar_xywh[2], tar_xywh[1]+tar_xywh[3]]
        dist_xyxy = dataset['dist_xyxy'][targ_idx]
        tar_topname = dataset['topname'][targ_idx]
        is_swapped = dataset['swapped_t_d'][targ_idx]
        img_infos_prag[vg_image_id] = [tar_xyxy, dist_xyxy, tar_topname, is_swapped]
        
        speaker_obs, _, _, _ = gen_batch(dataset, 1, fieldname, p_notseedist=0, vae=vae_model, see_distractors=settings.see_distractor, glove_data=glove_data, num_dist=num_distractors, preset_targ_idx=targ_idx)

        if speaker_obs != None:
            # Now get the EC for that speaker obs
            with torch.no_grad():
                comm, _, _ = team.speaker(speaker_obs)
                np_comm = comm.detach().cpu().numpy()
                # comms.append(np_comm)
                likelihood, _ = team.speaker.get_token_dist(speaker_obs)
                index_of_one = torch.argmax(torch.tensor(likelihood)).item()
                ids_to_protos[vg_image_id] = index_of_one
                EC_words_prag[index_of_one].append(vg_image_id)
                ids_to_comms[vg_image_id] = np_comm[0]
        else : # there is no Glove embedding for that preset_targ_idx
            pass

    print("EC word used pragmatics:", ids_to_protos[sampled_id])   
    vector_prag = ids_to_protos[sampled_id]

    #if vector_prag != vector_lex:
    if not os.path.exists(save_path_vecs):
        os.makedirs(save_path_vecs)
        
    # download images for vector pragmatics
    download_images_per_vec("pragmatics", save_path_vecs, vector_prag, EC_words_prag, img_infos_prag, sampled_id, include_target=True)

    # download images for vector lexsem and related ones
    for j in range(100):
        vec_close = similarity_dict[vector_lex][j]
        if len(EC_words_lexsem[vec_close]) >= 10:
            break
    print("close vector:", vec_close)
    # and the furthest
    similarity_dict[vector_lex].reverse()
    for j in range(100):
        vec_far = similarity_dict[vector_lex][j]
        if len(EC_words_lexsem[vec_far]) >= 10:
            break
    print("far vector:", vec_far)
        
    download_images_per_vec("lexsem", save_path_vecs, vector_lex, EC_words_lexsem, img_infos_lexsem, sampled_id, include_target=True)
    download_images_per_vec("lexsem", save_path_vecs, vec_close, EC_words_lexsem, img_infos_lexsem, sampled_id, include_target=False)
    download_images_per_vec("lexsem", save_path_vecs, vec_far, EC_words_lexsem, img_infos_lexsem, sampled_id, include_target=False)


        # plot the prototypes
    pca = PCA(n_components=2)
    reduced_proto = pca.fit_transform(prototypes)

    x_coords, y_coords = zip(*reduced_proto)

    colors = ['grey'] * len(reduced_proto)
    sizes = [20] * len(reduced_proto)
    marker_styles = ['o'] * len(reduced_proto)  # Default shape

    colors[vector_lex] = 'blue'
    colors[vec_close] = 'green'
    colors[vec_far] = 'red'
    colors[vector_prag] = 'orange'
    sizes[vector_lex] = 100  # Bigger size for colored dots
    sizes[vec_close] = 100
    sizes[vec_far] = 100
    sizes[vector_prag] = 100
    marker_styles[vector_lex] = '*'  # Star shape for colored dots
    marker_styles[vec_close] = '*'
    marker_styles[vec_far] = '*'
    marker_styles[vector_prag] = '*'

    plt.figure(figsize=(6.5, 5))
    for i in range(len(x_coords)):
        plt.scatter(x_coords[i], y_coords[i], color=colors[i], s=sizes[i], marker=marker_styles[i], alpha=0.3 if colors[i] == 'grey' else 1)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(False)
    plt.xticks([])  # This will remove the x-axis ticks
    plt.yticks([])  # This will remove the y-axis ticks

    plt.show()

    plt.savefig(save_path_vecs + "vectors.png",dpi=300)


   
        #### PLOT: histogram of topnames of closest images
        #top_columns_100 = list(row_values.nlargest(100).index)
        #closest_topnames = [img_infos[i][2] for i in top_columns_100]
        #topname_counts = {}
        #for name in closest_topnames:
        #    topname_counts[name] = topname_counts.get(name, 0) + 1

        #sorted_topname_counts = sorted(topname_counts.items(), key=lambda x: x[1], reverse=True)
        #topname_labels, topname_frequencies = zip(*sorted_topname_counts)
        #topname_values = np.arange(len(topname_labels))

        #plt.figure(figsize=(12, 6))  # Adjust the figure size as per your requirement
        #plt.bar(topname_values, topname_frequencies)
        #plt.xlabel('Names')
        #plt.ylabel('Frequency')
        #plt.title(str(img_infos[sampled_id][2]))
        #plt.xticks(topname_values, topname_labels, rotation=90)  # Rotate x-axis labels if necessary
        #plt.show()
        #plt.savefig(save_path + "topnames_closest_"+sampled_id+"_lexsem.jpg")
    
    # if no 2 images found for the same word
#    else:
#        pass





def vis_sim_per_word(dataset, team, sampled_id):
    
    #To get the cosine similarity between objects for which the same word is used.

    # PRAGMATICS
    ids_to_comms = {}
    img_infos = {}
    for targ_idx in list(dataset.index.values):
        vg_image_id = dataset['vg_image_id'][targ_idx]
        tar_xywh = dataset['bbox_xywh'][targ_idx]
        tar_xyxy = [tar_xywh[0], tar_xywh[1], tar_xywh[0]+tar_xywh[2], tar_xywh[1]+tar_xywh[3]]
        dist_xyxy = dataset['dist_xyxy'][targ_idx]
        tar_topname = dataset['topname'][targ_idx]
        img_infos[vg_image_id] = [tar_xyxy, dist_xyxy, tar_topname, targ_idx]

        speaker_obs, _, _, _ = gen_batch(dataset, 1, fieldname, p_notseedist=0, vae=vae_model, see_distractors=settings.see_distractor, glove_data=glove_data, num_dist=num_distractors, preset_targ_idx=targ_idx)

        if speaker_obs != None:
            # Now get the EC for that speaker obs
            with torch.no_grad():
                comm, _, _ = team.speaker(speaker_obs)
                np_comm = comm.detach().cpu().numpy()
                ids_to_comms[vg_image_id] = np_comm[0]
        else : # there is no Glove embedding for that preset_targ_idx
            pass
    
    arrays = np.array(list(ids_to_comms.values()))
    matrix = cosine_similarity(arrays)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.columns = list(ids_to_comms.keys())
    df_matrix.index = list(ids_to_comms.keys())
    row_values = df_matrix.loc[sampled_id]
    high_columns = row_values[row_values > settings.sim_threshold].index.tolist()
    indices = [img_infos[i][3] for i in high_columns]

    targets_obs, distractors_obs = [], []
    for img in indices:
        speaker_obs, _, _, _ = gen_batch(dataset, 1, fieldname, p_notseedist=0, vae=vae_model, see_distractors=settings.see_distractor, glove_data=glove_data, num_dist=num_distractors, preset_targ_idx=img)
        targets_obs.append(speaker_obs[0][0].detach().cpu().numpy())
        distractors_obs.append(speaker_obs[0][1].detach().cpu().numpy())

    targets_obs_arrays = np.array(targets_obs)
    distractors_obs_arrays = np.array(distractors_obs)
    tar_feat_sim = cosine_similarity(targets_obs_arrays)
    dist_feat_sim = cosine_similarity(distractors_obs_arrays)
    prag_average_vim_sim_target = np.mean(tar_feat_sim)
    prag_average_vim_sim_distractor = np.mean(dist_feat_sim)


    # LEXICAL-SEMANTICS
    # reminder to put dropout = 0 when generating observations to compute the pairwise similarity between distractors (not for the communication)
    ids_to_comms = {}
    img_infos = {}
    for targ_idx in list(dataset.index.values):
        vg_image_id = dataset['vg_image_id'][targ_idx]
        tar_xywh = dataset['bbox_xywh'][targ_idx]
        tar_xyxy = [tar_xywh[0], tar_xywh[1], tar_xywh[0]+tar_xywh[2], tar_xywh[1]+tar_xywh[3]]
        dist_xyxy = dataset['dist_xyxy'][targ_idx]
        tar_topname = dataset['topname'][targ_idx]
        img_infos[vg_image_id] = [tar_xyxy, dist_xyxy, tar_topname, targ_idx]

        speaker_obs, _, _, _ = gen_batch(dataset, 1, fieldname, p_notseedist=1, vae=vae_model, see_distractors=settings.see_distractor, glove_data=glove_data, num_dist=num_distractors, preset_targ_idx=targ_idx)

        if speaker_obs != None:
            # Now get the EC for that speaker obs
            with torch.no_grad():
                comm, _, _ = team.speaker(speaker_obs)
                np_comm = comm.detach().cpu().numpy()
                ids_to_comms[vg_image_id] = np_comm[0]
        else : # there is no Glove embedding for that preset_targ_idx
            pass

    arrays = np.array(list(ids_to_comms.values()))
    matrix = cosine_similarity(arrays)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.columns = list(ids_to_comms.keys())
    df_matrix.index = list(ids_to_comms.keys())
    row_values = df_matrix.loc[sampled_id]
    high_columns = row_values[row_values > settings.sim_threshold].index.tolist()
    indices = [img_infos[i][3] for i in high_columns]

    targets_obs, distractors_obs = [], []
    for img in indices:
        speaker_obs, _, _, _ = gen_batch(dataset, 1, fieldname, p_notseedist=0, vae=vae_model, see_distractors=settings.see_distractor, glove_data=glove_data, num_dist=num_distractors, preset_targ_idx=img)
        targets_obs.append(speaker_obs[0][0].detach().cpu().numpy())
        distractors_obs.append(speaker_obs[0][1].detach().cpu().numpy())

    targets_obs_arrays = np.array(targets_obs)
    distractors_obs_arrays = np.array(distractors_obs)
    tar_feat_sim = cosine_similarity(targets_obs_arrays)
    dist_feat_sim = cosine_similarity(distractors_obs_arrays)
    lex_average_vim_sim_target = np.mean(tar_feat_sim)
    lex_average_vim_sim_distractor = np.mean(dist_feat_sim)
    
    return prag_average_vim_sim_target, prag_average_vim_sim_distractor, lex_average_vim_sim_target, lex_average_vim_sim_distractor



def run(plot_img_grids=True):
    if settings.see_distractors_pragmatics:
        num_imgs = 3 if settings.with_ctx_representation else 2
    else:
        num_imgs = 1 if not settings.see_distractor else (num_distractors + 1)

    data = get_feature_data(t_features_filename, excluded_ids=excluded_ids)
    data = data.sample(frac=1) # Shuffle the data.
    
    train_data, test_data, val_data = np.split(data.sample(frac=1, random_state=46),
                                        [int(.7*len(data)), int(.9*len(data))])
    train_data, test_data, val_data = train_data.reset_index(), test_data.reset_index(), val_data.reset_index()
    train_data, test_data, val_data = train_data.sample(frac=1).reset_index(drop=True), test_data.sample(frac=1).reset_index(drop=True), val_data.sample(frac=1).reset_index(drop=True)
    print("Len test set:", len(test_data))
    print("Len val set:", len(val_data))

    # with the validation set, we make sure that humans and models are talking about the same target
    val_data = val_data.loc[val_data['swapped_t_d'] == 0]
    val_data = val_data.reset_index(drop=True)
    print("Len val set non swapped:", len(val_data))

    speaker = VQ(feature_len, c_dim, num_layers=3, num_protos=settings.num_protos, num_simultaneous_tokens=1, variational=variational, num_imgs=num_imgs)
    listener = ListenerPragmaticsCosines(feature_len)
    decoder = Decoder(c_dim, feature_len, num_layers=3, num_imgs=num_imgs)
    model = Team(speaker, listener, decoder)

    random_init_dir = "random_init/" if settings.random_init else "anneal/"

    if viz_topname != None:
        val_data = val_data.loc[val_data['topname'] == viz_topname]
        
    if plot_img_grids:
        
        # IMAGE GRIDS and HISTOGRAMS
        # here we sample one seed and make plots for a few images (no average across seeds, nor across images)

        sampled_images = random.sample(list(train_data['vg_image_id']), 10)
        sampled_seed = random.sample(settings.seeds, 1)
        
        for u in settings.utilities:

            folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
            folder_utility = "utility"+str(u)+"/"
            folder_alpha = "alpha"+str(settings.alpha)+"/"

            # to save images per vector
            model_id = "utility" + str(u) + "_alpha" + str(settings.alpha) + "/"
            
            for ex,s in zip(range(len(sampled_images)), sampled_images):
                new_path_vecs = "Plots/" + str(settings.num_protos) + '/' + random_init_dir + "EC_comm/" + model_id + "/" + "example" + str(ex) + "/"

                json_file_path = "src/saved_models/" + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(sampled_seed[0]) + '/'
                json_file = json_file_path+"objective_merged.json"
                with open(json_file, 'r') as f:
                    existing_params = json.load(f)
                convergence_epoch = existing_params["utility"+str(u)]["inf_weight"+str(settings.alpha)]['convergence epoch']
                # load model
                model_to_eval_path = 'src/saved_models/' + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(sampled_seed[0]) + '/' + folder_utility + folder_alpha + str(convergence_epoch)
                model.load_state_dict(torch.load(model_to_eval_path + '/model.pt'))
                model.to(settings.device)
                model.eval()

                # to save image grids
                if viz_topname != None:
                    new_path = "Plots/" + str(settings.num_protos) + '/' + random_init_dir + "EC_comm/" + folder_utility + folder_alpha + folder_ctx + 'kl_weight' + str(settings.kl_weight) + "/" + viz_topname + "/"
                else:
                    new_path = "Plots/" + str(settings.num_protos) + '/' + random_init_dir + "EC_comm/" + folder_utility + folder_alpha + folder_ctx + 'kl_weight' + str(settings.kl_weight) + "/"
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                
                print(ex, s)
                visualize_solution(train_data, model, num_examples=110, save_path_vecs=new_path_vecs, save_path_grids=new_path, sampled_id=s)
                 
    else:

        # CHECK DISTRACTORS' AND TARGETS' SIMILARITIES
        # sample N images, average across seeds and across images

        prag_tar_sim, prag_dist_sim, lex_tar_sim, lex_dist_sim = [], [], [], []
        sampled_images = random.sample(list(val_data['vg_image_id']), 1000)
        # we exclude these 4 because we don't have their topname embedding, so they are not in ids_to_comm
        sampled_images = [i for i in sampled_images if i not in ["1034", "2413150", "1515", "1073"]]
        sampled_seed = random.sample(settings.seeds, 1)

        for u in settings.utilities:
            
            print("utility:", u)

            folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
            folder_utility = "utility"+str(u)+"/"
            folder_alpha = "alpha"+str(settings.alpha)+"/"
            
            print("alpha:", settings.alpha)
            model_id = "utility" + str(u) + "_alpha" + str(settings.alpha) + "/"

            json_file_path = "src/saved_models/" + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(sampled_seed[0]) + '/'
            json_file = json_file_path+"objective_merged.json"
            with open(json_file, 'r') as f:
                existing_params = json.load(f)
            convergence_epoch = existing_params["utility"+str(u)]["inf_weight"+str(settings.alpha)]['convergence epoch']
            # load model
            model_to_eval_path = 'src/saved_models/' + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(sampled_seed[0]) + '/' + folder_utility + folder_alpha + str(convergence_epoch)
            
            model.load_state_dict(torch.load(model_to_eval_path + '/model.pt'))
            model.to(settings.device)
            model.eval()
            
            folder_ctx_to_save = "with_ctx_" if settings.with_ctx_representation else "without_ctx_"
            #if viz_topname != None:
            #    new_path = "Plots/" + str(settings.num_protos) + '/' + folder_training_type + folder_ctx_to_save + 'kl_weight' + str(settings.kl_weight) + "_" + viz_topname + "/"
            #else:
            #    new_path = "Plots/" + str(settings.num_protos) + '/' + folder_training_type + folder_ctx_to_save + 'kl_weight' + str(settings.kl_weight) + "_"

            tmp_prag_tar_sim = []
            tmp_prag_dist_sim = []
            tmp_lex_tar_sim = []
            tmp_lex_dist_sim = []

            for image in sampled_images:
                viz  = vis_sim_per_word(val_data, model, sampled_id=image)
                tmp_prag_tar_sim.append(viz[0])
                tmp_prag_dist_sim.append(viz[1])
                tmp_lex_tar_sim.append(viz[2])
                #tmp_lex_dist_sim.append(viz[3])

            tmp_prag_tar_sim = [i for i in tmp_prag_tar_sim if i != 1.0]
            tmp_prag_dist_sim = [i for i in tmp_prag_dist_sim if i != 1.0]
            tmp_lex_tar_sim = [i for i in tmp_lex_tar_sim if i != 1.0]
            tmp_lex_dist_sim = [i for i in tmp_lex_dist_sim if i != 1.0]

            prag_tar_sim.append(sum(tmp_prag_tar_sim) / len(tmp_prag_tar_sim))
            prag_dist_sim.append(sum(tmp_prag_dist_sim) / len(tmp_prag_dist_sim))
            lex_tar_sim.append(sum(tmp_lex_tar_sim) / len(tmp_lex_tar_sim))
            #lex_dist_sim.append(sum(tmp_lex_dist_sim) / len(tmp_lex_dist_sim))
        
        print("Lexsem, targets:", lex_tar_sim)
        print("Pragm, targets:", prag_tar_sim)
        print("Pragm, distractors:", prag_dist_sim)



if __name__ == '__main__':
    feature_len = 512
    settings.see_distractor = False
    settings.see_distractors_pragmatics = True

    settings.with_ctx_representation = False
    settings.dropout = False
    settings.see_probabilities = True
    
    settings.random_init = True

    settings.eval_someRE = False
    settings.sim_threshold = 0.99

    num_distractors = 1
    settings.num_distractors = num_distractors
    n_epochs = 3000
    v_period = 100  # How often to test on the validation set and calculate various info metrics.
    num_burnin = 500
    b_size = 1024
    c_dim = 128
    variational = True
    # Measuring complexity takes a lot of time. For debugging other features, set to false.
    do_calc_complexity = True
    do_plot_comms = False
    
    settings.num_protos = 3000 #442
    settings.alpha = 200 # informativeness
    settings.utilities = [140] 
    settings.kl_weight = 1.0 # complexity  
    settings.kl_incr = 0.0
    
    settings.entropy_weight = 0.0
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.learned_marginal = False
    settings.max_num_align_data = 1
    settings.distinct_words = False  # FIXME
    with_bbox = False
    image_directory = 'src/data/images_nobox/'
    num_rand_trials = 5 # to regulate get_relative_embeddings

    t_features_filename = 'src/data/t_features.csv'
    settings.d_features_filename = 'src/data/d_features.csv'
    settings.d_bboxes_filename = 'src/data/d_xyxy.tsv'
    settings.ctx_features_filename = 'src/data/ctx_features.csv'
    manynames = load_cleaned_results(filename="src/data/manynames.tsv")
    someRE = pd.read_csv("src/data/someRE.csv", sep = ";")
    merged_tmp = pd.merge(manynames, someRE, on=['link_vg'])
    excluded_ids = [i for i in merged_tmp['vg_image_id']]
    vae_model = VAE(512, 32)
    vae_model.load_state_dict(torch.load('src/saved_models/vae0.001.pt'))
    vae_model.to(settings.device)
    np.random.seed(0)
    settings.embedding_cache = {}
    settings.sample_first = True
    speaker_type = 'vq'  # Options are 'vq', 'cont', or 'onehot'

    settings.seeds = [0]
    my_seed = 0
    random.seed(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    
    glove_data = get_glove_vectors(32)
    fieldname = "topname"
    viz_topname = None

    run(plot_img_grids=True)

