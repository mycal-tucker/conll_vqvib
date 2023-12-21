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


def visualize_solution(dataset, team, num_examples, save_path, sampled_id):
   
    # Given a dataset and a team, returns a list of EC comms for some entries in the data,
    # as well as the English words for each of those entries.

    # LEXICAL-SEMANTICS
    ids_to_comms = {}
    img_infos = {}
    for targ_idx in list(dataset.index.values):
        vg_image_id = dataset['vg_image_id'][targ_idx]
        tar_xywh = dataset['bbox_xywh'][targ_idx]
        tar_xyxy = [tar_xywh[0], tar_xywh[1], tar_xywh[0]+tar_xywh[2], tar_xywh[1]+tar_xywh[3]]
        dist_xyxy = dataset['dist_xyxy'][targ_idx]
        tar_topname = dataset['topname'][targ_idx]
        img_infos[vg_image_id] = [tar_xyxy, dist_xyxy, tar_topname]

        speaker_obs, _, _, _ = gen_batch(dataset, 1, fieldname, p_notseedist=1, vae=vae_model, see_distractors=settings.see_distractor, glove_data=glove_data, num_dist=num_distractors, preset_targ_idx=targ_idx)
        
        if speaker_obs != None:
            # Now get the EC for that speaker obs
            with torch.no_grad():
                comm, _, _ = team.speaker(speaker_obs)
                np_comm = comm.detach().cpu().numpy()
                # comms.append(np_comm)
                ids_to_comms[vg_image_id] = np_comm[0]
        else : # there is no Glove embedding for that preset_targ_idx
            pass

    arrays = np.array(list(ids_to_comms.values()))
    matrix = cosine_similarity(arrays)
    #print(np.argwhere(matrix > 0.95))
    df_matrix = pd.DataFrame(matrix)
    df_matrix.columns = list(ids_to_comms.keys())
    df_matrix.index = list(ids_to_comms.keys())
    
    row_values = df_matrix.loc[sampled_id]
    top_columns = list(row_values[row_values > 0.99].index)
    if len(top_columns) > 1:
        #top_columns = list(row_values.nlargest(10).index)
        top_columns.remove(sampled_id)  # make sure that the sampled_id is in first position (in case there are multiple comm vectors with sim=1)
        top_columns.insert(0, sampled_id)
        info_to_write = [str(round(i,2)) for i in row_values[row_values > 0.99]]
        #info_to_write = [str(round(i,3)) for i in row_values.nlargest(10)]
        info_to_write[0] = img_infos[sampled_id][2]
        targets = [img_infos[i][0] for i in top_columns]
        distractors = [ast.literal_eval(img_infos[i][1]) for i in top_columns]

#### PLOT: images with closest communication vector
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 6))
        for i, ax in enumerate(axes.flat):
            if i < len(top_columns):
                # Load and display the image
                image_path = image_directory + top_columns[i] + ".jpg"
                #img = mpimg.imread(image_path)
                img = Image.open(image_path)
                ax.imshow(img)
            
                # draw target
                draw = ImageDraw.Draw(img)
                red =  (255, 0, 0)
                t_coord = [(targets[i][0], targets[i][1]), (targets[i][2], targets[i][3])]
                draw.rectangle(t_coord, outline=red, width=4)

                # draw distractor
                #draw = ImageDraw.Draw(img)
                #blue =  (0, 0, 255)
                #d_coord = [(distractors[i][0], distractors[i][1]), (distractors[i][2], distractors[i][3])]
                #draw.rectangle(d_coord, outline=blue, width=4)

                ax.imshow(img)

                ax.axis('off')  # Turn off the axis labels for cleaner display
                ax.text(0.5, 1.05, info_to_write[i], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='black')

            else:
                # Remove any unused subplots
                fig.delaxes(ax)
    
        plt.suptitle("Lexical-semantics")    
        plt.tight_layout()
        if viz_topname:
            plt.savefig(save_path + "top_comm_"+viz_topname+"_"+sampled_id+"_lexsem.jpg")
        else:
            plt.savefig(save_path + "top_comm_"+sampled_id+"_lexsem.jpg")

        # PRAGMATICS
        ids_to_comms = {}
        img_infos = {}
        for targ_idx in list(dataset.index.values):
            vg_image_id = dataset['vg_image_id'][targ_idx]
            tar_xywh = dataset['bbox_xywh'][targ_idx]
            tar_xyxy = [tar_xywh[0], tar_xywh[1], tar_xywh[0]+tar_xywh[2], tar_xywh[1]+tar_xywh[3]]
            dist_xyxy = dataset['dist_xyxy'][targ_idx]
            tar_topname = dataset['topname'][targ_idx]
            img_infos[vg_image_id] = [tar_xyxy, dist_xyxy, tar_topname]

            speaker_obs, _, _, _ = gen_batch(dataset, 1, fieldname, p_notseedist=0, vae=vae_model, see_distractors=settings.see_distractor, glove_data=glove_data, num_dist=num_distractors, preset_targ_idx=targ_idx)

            if speaker_obs != None:
                # Now get the EC for that speaker obs
                with torch.no_grad():
                    comm, _, _ = team.speaker(speaker_obs)
                    np_comm = comm.detach().cpu().numpy()
                    # comms.append(np_comm)
                    ids_to_comms[vg_image_id] = np_comm[0]
            else : # there is no Glove embedding for that preset_targ_idx
                pass

        arrays = np.array(list(ids_to_comms.values()))
        matrix = cosine_similarity(arrays)
        df_matrix = pd.DataFrame(matrix)
        df_matrix.columns = list(ids_to_comms.keys())
        df_matrix.index = list(ids_to_comms.keys())

        row_values = df_matrix.loc[sampled_id]
        top_columns = list(row_values[row_values > 0.99].index)
        #top_columns = list(row_values.nlargest(10).index)
        top_columns.remove(sampled_id)  # make sure that the sampled_id is in first position (in case there are multiple comm vectors with sim=1)
        top_columns.insert(0, sampled_id)
        info_to_write = [str(round(i,2)) for i in row_values[row_values > 0.99]]
        #info_to_write = [str(round(i,3)) for i in row_values.nlargest(10)]
        info_to_write[0] = img_infos[sampled_id][2]
        targets = [img_infos[i][0] for i in top_columns]
        distractors = [ast.literal_eval(img_infos[i][1]) for i in top_columns]

#### PLOT: images with closest communication vector
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 6))
        for i, ax in enumerate(axes.flat):
            if i < len(top_columns):
                # Load and display the image
                image_path = image_directory + top_columns[i] + ".jpg"
                #img = mpimg.imread(image_path)
                img = Image.open(image_path)
                ax.imshow(img)

                # draw target
                draw = ImageDraw.Draw(img)
                red =  (255, 0, 0)
                t_coord = [(targets[i][0], targets[i][1]), (targets[i][2], targets[i][3])]
                draw.rectangle(t_coord, outline=red, width=4)

                # draw distractor
                draw = ImageDraw.Draw(img)
                blue =  (0, 0, 255)
                d_coord = [(distractors[i][0], distractors[i][1]), (distractors[i][2], distractors[i][3])]
                draw.rectangle(d_coord, outline=blue, width=4)

                ax.imshow(img)

                ax.axis('off')  # Turn off the axis labels for cleaner display
                ax.text(0.5, 1.05, info_to_write[i], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='black')

            else:
                # Remove any unused subplots
                fig.delaxes(ax)

        plt.suptitle("Pragmatics")
        plt.tight_layout()
        if viz_topname:
            plt.savefig(save_path + "top_comm_"+viz_topname+"_"+sampled_id+"_pragmatics.jpg")
        else:
            plt.savefig(save_path + "top_comm_"+sampled_id+"_pragmatics.jpg")

    


#### PLOT: histogram of topnames of closest images
        top_columns_100 = list(row_values.nlargest(100).index)
        closest_topnames = [img_infos[i][2] for i in top_columns_100]
        topname_counts = {}
        for name in closest_topnames:
            topname_counts[name] = topname_counts.get(name, 0) + 1

        sorted_topname_counts = sorted(topname_counts.items(), key=lambda x: x[1], reverse=True)
        topname_labels, topname_frequencies = zip(*sorted_topname_counts)
        topname_values = np.arange(len(topname_labels))

        plt.figure(figsize=(12, 6))  # Adjust the figure size as per your requirement
        plt.bar(topname_values, topname_frequencies)
        plt.xlabel('Names')
        plt.ylabel('Frequency')
        plt.title(str(img_infos[sampled_id][2]))
        plt.xticks(topname_values, topname_labels, rotation=90)  # Rotate x-axis labels if necessary
        plt.show()
        plt.savefig(save_path + "topnames_closest_"+sampled_id+"_lexsem.jpg")
    
    # if no 2 images found for the same word
    else:
        pass





def vis_sim_per_word(dataset, team, sampled_id):

    # Given a dataset and a team, returns a list of EC comms for some entries in the data,
    # as well as the English words for each of those entries.

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

            json_file_path = "src/saved_models/" + str(settings.num_protos) + '/' + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(sampled_seed[0]) + '/'
            json_file = json_file_path+"objective_merged.json"
            with open(json_file, 'r') as f:
                existing_params = json.load(f)
            convergence_epoch = existing_params["utility"+str(u)]["inf_weight"+str(settings.alpha)]['convergence epoch']
            # load model
            model_to_eval_path = 'src/saved_models/' + str(settings.num_protos) + '/' + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(sampled_seed[0]) + '/' + folder_utility + folder_alpha + str(convergence_epoch)
            model.load_state_dict(torch.load(model_to_eval_path + '/model.pt'))
            model.to(settings.device)
            model.eval()

            if viz_topname != None:
                new_path = "Plots/" + str(settings.num_protos) + '/' + folder_utility + folder_alpha + folder_ctx + 'kl_weight' + str(settings.kl_weight) + "/" + viz_topname + "/"
            else:
                new_path = "Plots/" + str(settings.num_protos) + '/' + folder_utility + folder_alpha + folder_ctx + 'kl_weight' + str(settings.kl_weight) + "/"
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            
            for s in sampled_images:
                visualize_solution(train_data, model, num_examples=110, save_path=new_path, sampled_id=s)
                 
    else:

        # PLOTS DISTRACTORS' AND TARGETS' SIMILARITIES
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
            json_file_path = "src/saved_models/" + str(settings.num_protos) + '/' + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(sampled_seed[0]) + '/'
            json_file = json_file_path+"objective_merged.json"
            with open(json_file, 'r') as f:
                existing_params = json.load(f)
            convergence_epoch = existing_params["utility"+str(u)]["inf_weight"+str(settings.alpha)]['convergence epoch']
            # load model
            model_to_eval_path = 'src/saved_models/' + str(settings.num_protos) + '/' + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(sampled_seed[0]) + '/' + folder_utility + folder_alpha + str(convergence_epoch)
            
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
        #fig1 = plt.figure()
        #plt.plot(settings.p_notseedist, lex_tar_sim, label="Same word as in lex - tar sim")
        #plt.plot(settings.p_notseedist, prag_tar_sim, label="Same word as in pragm - tar sim")
        ##plt.plot(settings.p_notseedist, prag_dist_sim, label="Same word as in pragm - dist sim")
        ##plt.plot(settings.p_notseedist, lex_dist_sim, label="Same word as in lex - dist sim")
        #x_label = "p dropout" if settings.dropout else "prob not see distractor"
        #plt.xlabel(x_label)
        #plt.ylabel("Object visual similarity")
        #plt.legend()
        #plt.ylim(0.70, 1)
        #plt.title()
        #fig1.savefig(new_path + "solution_viz_similarity"+ str(settings.sim_threshold) +".png")
  


if __name__ == '__main__':
    feature_len = 512
    settings.see_distractor = False
    settings.see_distractors_pragmatics = True

    settings.with_ctx_representation = False
    settings.dropout = False
    settings.see_probabilities = True
    
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
    settings.utilities = [0]
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

