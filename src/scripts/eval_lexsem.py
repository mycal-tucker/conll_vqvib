import os
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
from src.data_utils.helper_fns import gen_batch, get_glove_embedding, get_unique_labels, get_entry_for_labels, get_unique_by_field, get_rand_entries
from src.data_utils.read_data import get_feature_data
from src.data_utils.read_data import load_cleaned_results
from src.models.decoder import Decoder
from src.models.listener_pragmatics import ListenerPragmaticsCosines
from src.models.team import Team
from src.models.vae import VAE
from src.models.vqvib2 import VQVIB2
from src.models.vq import VQ
from src.models.mlp import MLP
from src.models.proto import ProtoNetwork
from src.utils.mine_pragmatics import get_info, get_cond_info
from src.utils.plotting import plot_metrics, plot_naming, plot_scatter
from src.utils.performance_metrics import PerformanceMetrics

import time

from src.data_utils.read_data import get_glove_vectors


def evaluate_lexsem(model, dataset, batch_size, vae, glove_data, fieldname, num_dist=None):
    model.eval()
    num_test_batches = 10
    num_correct = 0
    total_recons_loss = 0
    num_total = 0
    for _ in range(num_test_batches):
        with torch.no_grad():
            speaker_obs, listener_obs, labels, _ = gen_batch(dataset, batch_size, fieldname, p_notseedist=1, vae=vae, glove_data=glove_data, see_distractors=settings.see_distractor, num_dist=num_dist)
            outputs, _, _, recons = model(speaker_obs, listener_obs)
            recons = torch.squeeze(recons, dim=1)
        pred_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        num_correct += np.sum(pred_labels == labels.cpu().numpy())
        num_total += pred_labels.size
        total_recons_loss += torch.mean(((speaker_obs[:, 0:1, :] - recons[:, 0:1, :]) ** 2)).item()
    acc = num_correct / num_total
    total_recons_loss = total_recons_loss / num_test_batches
    print("Lexical semantics")
    print("Evaluation on test set accuracy", acc)
    print("Evaluation on test set recons loss", total_recons_loss)
    return acc, total_recons_loss



def plot_comms(model, vae, dataset, basepath):
    num_tests = 100  # Generate lots of samples for the same input because it's not deterministic.
    labels = []
    
    for targ_features, distractor_features, ctx_features in zip(dataset['t_features'], dataset['d_features'], dataset['ctx_features']):
        if settings.with_ctx_representation:
            s_obs = np.expand_dims(np.vstack([targ_features] + [distractor_features] + [ctx_features]), axis=0)
        else:
            s_obs = np.expand_dims(np.vstack([targ_features] + [distractor_features]), axis=0)
        speaker_tensor = torch.Tensor(np.vstack(s_obs).astype(np.float)).to(settings.device)
        if vae is not None:
            with torch.no_grad():
                speaker_tensor, _ = vae(speaker_tensor)
        speaker_tensor = speaker_tensor.unsqueeze(0)
        
        # Add mask over distractor
        mask = torch.ones_like(speaker_tensor, dtype=bool)
        num_tensors = speaker_tensor.shape[0]
        indices = np.random.choice(num_tensors, int(num_tensors * 1), replace=False)
        mask[indices, 1, :] = False
        speaker_obs = torch.where(mask, speaker_tensor, torch.tensor(0.0, device=settings.device))
        
        speaker_obs = speaker_obs.repeat(num_tests, 1, 1)
        likelihoods = model.speaker.get_token_dist(speaker_obs)
        top_comm_idx = np.argmax(likelihoods)
        top_likelihood = likelihoods[top_comm_idx]
        label = top_comm_idx if top_likelihood > 0.4 else -1
        labels.append(label)
    
    features = np.vstack(dataset['t_features'])
    topnames = list(dataset['topname'])
    label_np = np.reshape(np.array(labels), (-1, 1))
    all_np = np.hstack([label_np, features])
    all_np = np.array(all_np, dtype=float) # convert to float
   
    regrouped_data = []
    plot_labels = []
    plot_mean = True
    for c in np.unique(labels):
        ix = np.where(all_np[:, 0] == c)
        matching_features = np.vstack(all_np[ix, 1:])
        averaged = np.mean(matching_features, axis=0, keepdims=True)
        plot_features = averaged if plot_mean else matching_features
        regrouped_data.append(plot_features)
        plot_labels.append(c)
    
    plot_naming(regrouped_data, topnames, viz_method='mds', labels=plot_labels, savepath=basepath + 'lexsem_mds')
    plot_naming(regrouped_data, topnames, viz_method='tsne', labels=plot_labels, savepath=basepath + 'lexsem_tsne')



def eval_model_lexsem(model, vae, comm_dim, data, viz_data, glove_data, num_cand_to_metrics, save_eval_path,
               fieldname, calculate_complexity=False, plot_comms_flag=False, alignment_dataset=None, save_model=True):
    # Create a directory to save information, models, etc.

    if not os.path.exists(save_eval_path + 'lexsem/'):
        os.makedirs(save_eval_path + 'lexsem/')
    if calculate_complexity:
        test_complexity = get_cond_info(model, data, targ_dim=comm_dim, p_notseedist=1, glove_data=glove_data, num_epochs=200)
        print("Test complexity", test_complexity)
    else:
        test_complexity = None
        val_complexity = None
    
    # get consistencies
#    consistencies = []
#    for j, align_data in enumerate(alignment_datasets):
#        for use_comm_idx in [False]:
#            use_top = True
#            dummy_eng = english_fieldname
#            if english_fieldname == 'responses':
#                use_top = False
#                dummy_eng = 'topname'
#            consistency_score = get_relative_embedding_LEXSEM(model, align_data, glove_data, test_data, fieldname='responses')
#            consistencies.append(consistency_score)

#    num_runs = len(alignment_datasets)
#    print(consistencies)
#    print("consistencies:", (np.median(consistencies), np.std(consistencies) / np.sqrt(num_runs)))
            
    eval_batch_size = 256
    complexities = [test_complexity]
    for set_distinction in [True]:
        for feature_idx, data in enumerate([data]):
            for num_candidates in num_cand_to_metrics.get(set_distinction).keys():
                settings.distinct_words = set_distinction
                acc, recons = evaluate_lexsem(model, data, eval_batch_size, vae, glove_data, fieldname=fieldname, num_dist=num_candidates - 1)
            relevant_metrics = num_cand_to_metrics.get(set_distinction).get(num_candidates)[feature_idx]
            relevant_metrics.add_data("eval_epoch", complexities[feature_idx], -1 * recons, acc, settings.kl_weight)
    
    # Plot some of the metrics for online visualization
    comm_accs = []
    regressions = []
    labels = []
    epoch_idxs = None
    plot_metric_data = num_cand_to_metrics.get(True)
    for feature_idx, label in enumerate(['test']):
        for num_candidates in sorted(plot_metric_data.keys()):
            comm_accs.append(plot_metric_data.get(num_candidates)[feature_idx].comm_accs)
#           regressions.append(plot_metric_data.get(num_candidates)[feature_idx].embed_r2)
            labels.append(" ".join([label, str(num_candidates), "utility"]))
            if epoch_idxs is None:
                epoch_idxs = plot_metric_data.get(num_candidates)[feature_idx].epoch_idxs
    plot_metrics(comm_accs, labels, epoch_idxs, save_eval_path + 'lexsem/')
#    plot_metrics(regressions, ['r2 score'], epoch_idxs, save_eval_path + 'regression_')

    # Visualize some of the communication
    try:
        if plot_comms_flag:
            plot_comms(model, vae_model, viz_data, save_eval_path)
    except AssertionError:
        print("Can't plot comms for whatever reason (e.g., continuous communication makes categorizing hard)")
    # Save the model and metrics to files.
    for feature_idx, label in enumerate(['test']):
        for set_distinction in num_cand_to_metrics.keys():
            for num_candidates in sorted(num_cand_to_metrics.get(set_distinction).keys()):
                metric = num_cand_to_metrics.get(set_distinction).get(num_candidates)[feature_idx]
                metric.to_file(save_eval_path + 'lexsem/' + "_".join([label, str(set_distinction), str(num_candidates), "metrics"]))
    if not save_model:
        return
    torch.save(model.state_dict(), save_eval_path + 'lexsem/model.pt')



# Given a model and a set of anchors, compute the relative encoding position of lots of communication vectors?
def get_relative_embedding_LEXSEM(model, vae, anchor_dataset, glove_data, rel_abs_data, fieldname):
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
   
    for seed in settings.seeds:
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        data = get_feature_data(t_features_filename, excluded_ids=excluded_ids)
        data = data.sample(frac=1, random_state=seed) # Shuffle the data.
        train_data, test_data, val_data = np.split(data.sample(frac=1, random_state=seed), 
                                        [int(.7*len(data)), int(.9*len(data))])
        train_data, test_data, val_data = train_data.reset_index(), test_data.reset_index(), val_data.reset_index()
        train_data, test_data, val_data = train_data.sample(frac=1, random_state=seed).reset_index(), test_data.sample(frac=1, random_state=seed).reset_index(), val_data.sample(frac=1, random_state=seed).reset_index()

        print("Len test set:", len(test_data))
        print("Len val set:", len(val_data))
  
        viz_data = val_data  

        speaker = VQ(feature_len, c_dim, num_layers=3, num_protos=settings.num_protos, num_simultaneous_tokens=1, variational=variational, num_imgs=num_imgs)
        #speaker = VQVIB2(feature_len, c_dim, num_layers=3, num_protos=settings.num_protos, num_simultaneous_tokens=1, variational=variational, num_images=num_imgs)
        listener = ListenerPragmaticsCosines(feature_len)
        decoder = Decoder(c_dim, feature_len, num_layers=3, num_imgs=num_imgs)
        model = Team(speaker, listener, decoder)

        for u in settings.utilities:
            print(u)
            for a in settings.alphas:
                print(a)
                
                try:
                    folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
                    folder_utility = "utility"+str(u)+"/"
                    folder_alpha = "alpha"+str(a)+"/"

                    # get convergence epoch for that model
                    json_file_path = "src/saved_models/" + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(seed) + '/'
                    json_file = json_file_path+"objective_merged.json"
                    with open(json_file, 'r') as f:
                        existing_params = json.load(f)
                    convergence_epoch = existing_params["utility"+str(u)]["inf_weight"+str(a)]['convergence epoch']
                    # load model
                    model_to_eval_path = 'src/saved_models/' + str(settings.num_protos) + '/' + random_init_dir + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(seed) + '/' + folder_utility + folder_alpha + str(convergence_epoch)
                    save_eval_path = model_to_eval_path + '/evaluation/'
                    model.load_state_dict(torch.load(model_to_eval_path + '/model.pt'))
                    model.to(settings.device)

                    print("Lexical semantics task") 
                    num_cand_to_metrics = {True: {2: []}}
                    for empty_list in num_cand_to_metrics.get(True).values():
                        empty_list.extend([PerformanceMetrics()])
                    eval_model_lexsem(model, vae_model, c_dim, val_data, viz_data, glove_data, num_cand_to_metrics, save_eval_path, fieldname='topname', calculate_complexity=do_calc_complexity, plot_comms_flag=do_plot_comms)
                    #alignment_datasets = [get_rand_entries(val_data, num_examples) for _ in range(num_rand_trial)]
                    #consistency_scores = []
                    #for j, align_data in enumerate(alignment_datasets):
                    #    consistency_score = get_relative_embedding_LEXSEM(model, vae_model, align_data, glove_data, val_data, fieldname='responses')     
                    #    consistency_scores.append(consistency_score)
                    #print(consistency_scores)
                    #print("average:", sum(consistency_scores) / len(consistency_scores))
                except: # not trained
                    print(u, a, "not found")
                    pass



if __name__ == '__main__':
    feature_len = 512
    settings.see_distractor = False
    settings.see_distractors_pragmatics = True

    settings.with_ctx_representation = False    
    settings.dropout = False
    settings.see_probabilities = True

    settings.eval_someRE = False
   
    settings.random_init = True
    random_init_dir = "random_init/" if settings.random_init else ""

    num_distractors = 1
    settings.num_distractors = num_distractors
    n_epochs = 3000
    v_period = 200  # How often to test on the validation set and calculate various info metrics.
    num_burnin = 500
    b_size = 1024
    c_dim = 128
    variational = True
    # Measuring complexity takes a lot of time. For debugging other features, set to false.
    do_calc_complexity = False
    do_plot_comms = False

    settings.num_protos = 3000 # 442 is the number of topnames in MN 
    #settings.alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 2.2, 3.7, 5, 6, 7, 8, 9, 10.5, 12.8, 21, 33, 88, 140, 200]
    #settings.utilities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 3.7, 5, 6, 7, 8, 10.5, 12.8, 21, 33, 88, 140, 200]
    settings.alphas = [0, 0.1, 0.2, 0.4, 0.7, 0.9, 1, 1.2, 1.5, 2.2, 3, 3.7, 5, 7, 9, 21, 33, 88, 140, 200]
    settings.utilities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 2.2, 3, 3.7, 5, 7, 9, 12.8, 21, 33, 88, 140, 200, 1.1, 1.3, 1.7, 1.9]
    settings.kl_weight = 1.0 # complexity
    
    # to get alignment
    num_rand_trial = 5 #5
    num_examples = 100 #500

    settings.kl_incr = 0.0
    settings.entropy_weight = 0.0
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.learned_marginal = False
    settings.max_num_align_data = 1
    settings.distinct_words = False  # FIXME
    with_bbox = False
    train_fraction = 0.5
    test_classes = ['couch', 'counter', 'bowl']
    viz_names = ['airplane', 'plane',
                 'animal', 'cow', 'dog', 'cat']
    
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
    settings.seeds = [0] #0,1,2 
    
    glove_data = get_glove_vectors(32)
    run()

