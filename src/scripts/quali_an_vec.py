import ast
import os
import json
import pandas as pd
import pickle
import random
from scipy import stats
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


# function to normalize the weights and adjust the rounding operation so that they sum to 1
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




def get_image_per_vec(dataset, team, num_data, save_dir, num_examples): 
    
    prototypes = team.speaker.vq_layer.prototypes.detach().cpu()
    
    # pairwise similarity between prototypes
    similarity_matrix = cosine_similarity(prototypes)    
    similarity_dict = {} # we store the most similar prototypes
    for i in range(len(prototypes)):
        sorted_indices = np.argsort(-similarity_matrix[i])
        sorted_indices = sorted_indices[sorted_indices != i]
        similarity_dict[i] = sorted_indices.tolist()

    # images' info
    img_infos = {}
    for targ_idx in list(dataset.index.values):
        vg_image_id = dataset['vg_image_id'][targ_idx]
        tar_xywh = dataset['bbox_xywh'][targ_idx]
        tar_xyxy = [tar_xywh[0], tar_xywh[1], tar_xywh[0]+tar_xywh[2], tar_xywh[1]+tar_xywh[3]]
        dist_xyxy = dataset['dist_xyxy'][targ_idx]
        tar_topname = dataset['topname'][targ_idx]
        img_infos[targ_idx] = [tar_xyxy, dist_xyxy, tar_topname]
    
    
    # LEXSEM
    comms = []        
    speaker_obs, _, _, _ = gen_batch(dataset, num_data, fieldname, p_notseedist=1, vae=vae_model, see_distractors=settings.see_distractor, glove_data=glove_data, num_dist=num_distractors)

    EC_words = {i: [] for i in range(3000)}
    for num,x in enumerate(speaker_obs):
        likelihood, _ = team.speaker.get_token_dist(x)
#        EC_distr.append(_.detach().cpu().numpy())
        index_of_one = torch.argmax(torch.tensor(likelihood)).item()
        EC_words[index_of_one].append(num)
    
    used = {}
    for k,v in EC_words.items():
        if len(v) > 10:
            used[k] = v 

    # we check images for the 2 vectors
    for ex in range(num_examples):
        print("___EX", ex, "___")
        # we sample one vector
        vec = 2112
        #vec = random.sample(list(used.keys()), 1)[0]
        print("vector:", vec)
        # and take the closest one
        for i in similarity_dict[vec]:
            if i in used.keys():
                vec_close = i
                break
        print("close vector:", vec_close)
        similarity_dict[vec].reverse()
        for i in similarity_dict[vec]:
            if i in used.keys():
                vec_far = i
                break
        print("far vector:", vec_far)
        for v in [vec, vec_close, vec_far]:
            save_dir_ex = save_dir + "example"+str(ex) + "/"
            save_dir_vec = save_dir_ex + str(v) + "/" 
            if not os.path.exists(save_dir_vec):
                os.makedirs(save_dir_vec)
            os.makedirs(save_dir_vec+"lexsem/")
            os.makedirs(save_dir_vec+"pragmatics/")

            sampled_images = random.sample(used[v], 10)
    
            targets = [img_infos[i][0] for i in sampled_images]
            images = [Image.open(image_directory + dataset.iloc[idx]['vg_image_id'] + ".jpg") for idx in sampled_images]
    
            for n,img in enumerate(images):
                # draw target
                draw = ImageDraw.Draw(img)
                red =  (255, 0, 0)
                t_coord = [(targets[n][0], targets[n][1]), (targets[n][2], targets[n][3])]
                draw.rectangle(t_coord, outline=red, width=4)

                img.save(save_dir_vec + "lexsem/img" + str(n) + ".png","PNG")


            # and we check what happens in pragmatics for the same vectors
            # PRAGMATICS
            comms = []
            speaker_obs, _, _, _ = gen_batch(dataset, num_examples, fieldname, p_notseedist=0, vae=vae_model, see_distractors=settings.see_distractor, glove_data=glove_data, num_dist=num_distractors)

            EC_words = {i: [] for i in range(3000)}
            for num,x in enumerate(speaker_obs):
                likelihood, _ = team.speaker.get_token_dist(x)
#               EC_distr.append(_.detach().cpu().numpy())
                index_of_one = torch.argmax(torch.tensor(likelihood)).item()
                EC_words[index_of_one].append(num)

            try:
                sampled_images = random.sample(EC_words[v], 10)
            except:
                sampled_images = EC_words[v]
            targets = [img_infos[i][0] for i in sampled_images]
            distractors = [ast.literal_eval(img_infos[i][1]) for i in sampled_images]

            images = [Image.open(image_directory + dataset.iloc[idx]['vg_image_id'] + ".jpg") for idx in sampled_images]

            for n,img in enumerate(images):
                # draw target
                draw = ImageDraw.Draw(img)
                red =  (255, 0, 0)
                t_coord = [(targets[n][0], targets[n][1]), (targets[n][2], targets[n][3])]
                draw.rectangle(t_coord, outline=red, width=4)
                
                # draw distractor
                draw = ImageDraw.Draw(img)
                blue =  (0, 0, 255)
                d_coord = [(distractors[n][0], distractors[n][1]), (distractors[n][2], distractors[n][3])]
                draw.rectangle(d_coord, outline=blue, width=4)

                img.save(save_dir_vec + "pragmatics/img" + str(n) + ".png","PNG")

    
        # plot the prototypes
        pca = PCA(n_components=2)
        reduced_proto = pca.fit_transform(prototypes)

        x_coords, y_coords = zip(*reduced_proto)

        colors = ['grey'] * len(reduced_proto)
        colors[vec] = 'blue'  
        colors[vec_close] = 'green' 
        colors[vec_far] = 'red'   

        plt.figure(figsize=(8, 6))
        plt.scatter(x_coords, y_coords, color=colors)
        plt.title('EC words')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(False)

        plt.show()

        plt.savefig(save_dir_ex +"vectors.png")






#    top_images = images[:5]
#    bot_images = images[5:]

#    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10), gridspec_kw={'hspace': 0.5, 'wspace': 0.1})
#    fig.suptitle('Vector n.' + str(vec), fontsize=16)
    
    #title_top = ""
    #title_bottom = ""
    #axes[0, 2].set_title(title_top, fontsize=14)
    #axes[1, 2].set_title(title_bottom, fontsize=14)

#    for num, ax in enumerate(axes[0]):
#        img = top_images[num]
#        ax.imshow(img)
#        ax.axis('off')  
#    for num, ax in enumerate(axes[1]):
#        img = bot_images[num]
#        ax.imshow(img)
#        ax.axis('off')  

 #       plt.tight_layout()
 #       plt.show()
 #       savedir = "Plots/3000/random_init/EC_comm/vec" + str(vec) + ".png" if settings.random_init else "Plots/3000/anneal/EC_comm/vec" + str(vec) + ".png"
  #      plt.savefig(savedir)



    # LEXSEM
    #speaker_obs, _, _, _ = gen_batch(dataset, num_examples, fieldname, p_notseedist=1, vae=vae_model, see_distractors=settings.see_distractor, glove_data=glove_data, num_dist=num_distractors)

    #EC_words = []
    #for x in speaker_obs:
        # we obtain a one-hot vector, that is the distribution over prototypes
        # 1 is at the index of the closest prototype
    #    likelihood = team.speaker.get_token_dist(x)
        # we take the index of the closest prototype
    #    index_of_one = torch.argmax(torch.tensor(likelihood)).item()
    #    EC_words.append(index_of_one)

    #print("lexsem:", len(set(EC_words)))


def run():
    if settings.see_distractors_pragmatics:
        num_imgs = 3 if settings.with_ctx_representation else 2
    else:
        num_imgs = 1 if not settings.see_distractor else (num_distractors + 1)

    random_init_dir = "random_init/" if settings.random_init else "anneal/"


    data = get_feature_data(t_features_filename, excluded_ids=excluded_ids)
    data = data.sample(frac=1, random_state=seed) # Shuffle the data.
    train_data, test_data, val_data = np.split(data.sample(frac=1, random_state=seed),
                                        [int(.7*len(data)), int(.9*len(data))])
    train_data, test_data, val_data = train_data.reset_index(), test_data.reset_index(), val_data.reset_index()
    train_data, test_data, val_data = train_data.sample(frac=1, random_state=seed).reset_index(), test_data.sample(frac=1, random_state=seed).reset_index(), val_data.sample(frac=1, random_state=seed).reset_index()

    print(len(train_data))

    speaker = VQ(feature_len, c_dim, num_layers=3, num_protos=settings.num_protos, num_simultaneous_tokens=1, variational=variational, num_imgs=num_imgs)
    listener = ListenerPragmaticsCosines(feature_len)
    decoder = Decoder(c_dim, feature_len, num_layers=3, num_imgs=num_imgs)
    model = Team(speaker, listener, decoder)

    for u in settings.utilities:
        for a in settings.alphas:
            
            norm_alpha, norm_ut, norm_compl = normalize_and_adjust([a, u, settings.kl_weight])
            print("==========")
            print(f'Utility: {norm_ut}, Informativeness: {norm_alpha}, Complexity: {norm_compl}')

            folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
            folder_utility = "utility"+str(u)+"/"
            folder_alpha = "alpha"+str(a)+"/"
            print("utility:", u, "alpha:", a)

            
            json_file_path = "src/saved_models/" + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(seed) + '/'
            json_file = json_file_path+"objective_merged.json"
            with open(json_file, 'r') as f:
                existing_params = json.load(f)
            convergence_epoch = existing_params["utility"+str(u)]["inf_weight"+str(a)]['convergence epoch']

            # load model
            model_to_eval_path = 'src/saved_models/' + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(seed) + '/' + folder_utility + folder_alpha + str(convergence_epoch)
            model.load_state_dict(torch.load(model_to_eval_path + '/model.pt'))
            model.to(settings.device)
            model.eval()
            
            model_id = "utility" + str(u) + "_alpha" + str(a) + "/"
            savedir = "Plots/3000/random_init/EC_comm/"+model_id if settings.random_init else "Plots/3000/anneal/EC_comm/"+model_id
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            get_image_per_vec(train_data, model, len(train_data), savedir, 1)
           

if __name__ == '__main__':
    feature_len = 512
    settings.see_distractor = False # MT setting
    settings.see_distractors_pragmatics = True # EG setting

    settings.with_ctx_representation = False
    settings.dropout = False
    settings.see_probabilities = True # this means that we are masking with a complete mask, no dropout

    settings.random_init = False
    settings.eval_someRE = False # this controls the data reading: we don't shuffle target and distractor

    settings.num_protos = 3000 #442
    #settings.alphas = [0, 0.1, 0.5, 1.5, 7, 200]  # informativeness
    #settings.utilities = [0, 0.1, 0.5, 1.5, 7, 200] # utility
    settings.alphas = [200]
    settings.utilities = [88]
    settings.kl_weight = 1.0 # complexity  
    settings.kl_incr = 0.0

    num_distractors = 1
    settings.num_distractors = num_distractors
    v_period = 100  # How often to test on the validation set and calculate various info metrics.
    num_burnin = 500
    b_size = 1024
    c_dim = 128
    variational = True
    settings.num_protos = 3000 # 442 is the number of topnames in MN 
    # Measuring complexity takes a lot of time. For debugging other features, set to false.
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
    merged_tmp = pd.merge(manynames, someRE, on=['link_vg'])
    excluded_ids = [i for i in merged_tmp['vg_image_id']]
    
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



