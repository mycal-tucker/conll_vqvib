import os

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
from src.models.listener_pragmatics import ListenerPragmatics
from src.models.team import Team
from src.models.vae import VAE
from src.models.vq import VQ
from src.models.mlp import MLP
from src.models.proto import ProtoNetwork
from src.utils.mine import get_info
from src.utils.plotting import plot_metrics, plot_naming, plot_scatter
from src.utils.performance_metrics import PerformanceMetrics

import time

from src.data_utils.read_data import get_glove_vectors

def evaluate_pragmatics(model, dataset, batch_size, vae, glove_data, fieldname, num_dist=None):
    model.eval()
    num_test_batches = 10
    num_correct = 0
    total_recons_loss = 0
    num_total = 0
    for _ in range(num_test_batches):
        with torch.no_grad():
            speaker_obs, listener_obs, labels, _ = gen_batch(dataset, batch_size, fieldname, vae=vae, glove_data=glove_data, see_distractors=settings.see_distractor, num_dist=num_dist)
            outputs, _, _, recons = model(speaker_obs, listener_obs)
            recons = torch.squeeze(recons, dim=1)
        pred_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        num_correct += np.sum(pred_labels == labels.cpu().numpy())
        num_total += pred_labels.size
        total_recons_loss += torch.mean(((speaker_obs - recons) ** 2)).item()
    acc = num_correct / num_total
    total_recons_loss = total_recons_loss / num_test_batches
    print("Evaluation on test set accuracy", acc)
    print("Evaluation on test set recons loss", total_recons_loss)
    return acc, total_recons_loss


def evaluate_lexsem(model, dataset, batch_size, vae, glove_data, fieldname, num_dist=None):
    model.eval()
    num_test_batches = 10
    num_correct = 0
    total_recons_loss = 0
    num_total = 0
    for _ in range(num_test_batches):
        with torch.no_grad():
            speaker_obs, listener_obs, labels, _ = gen_batch(dataset, batch_size, fieldname, vae=vae, glove_data=glove_data, see_distractors=settings.see_distractor, num_dist=num_dist)
            mask = torch.ones_like(speaker_obs, device=speaker_obs.device)
            mask[:, 1, :] = 0 # we mask the second tensor that is always the distractor 
            masked_tensor = speaker_obs * mask
            speaker_obs = masked_tensor
            outputs, _, _, recons = model(speaker_obs, listener_obs)
            recons = torch.squeeze(recons, dim=1)
        pred_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        num_correct += np.sum(pred_labels == labels.cpu().numpy())
        num_total += pred_labels.size
        total_recons_loss += torch.mean(((speaker_obs - recons) ** 2)).item()
    acc = num_correct / num_total
    total_recons_loss = total_recons_loss / num_test_batches
    print("Evaluation on test set accuracy", acc)
    print("Evaluation on test set recons loss", total_recons_loss)
    return acc, total_recons_loss



def plot_comms(model, dataset, basepath):
    num_tests = 1000  # Generate lots of samples for the same input because it's not deterministic.
    labels = []
    if settings.with_ctx_representation:
        for f, f_d, f_ctx in zip(dataset['t_features'], dataset['d_features'], dataset['ctx_features']):
            speaker_obs = np.expand_dims(np.vstack([f] + [f_d] + [f_ctx]), axis=0)
            speaker_obs = torch.Tensor(np.vstack(speaker_obs).astype(np.float)).to(settings.device)   
            speaker_obs = speaker_obs.unsqueeze(0)
            speaker_obs = speaker_obs.repeat(num_tests, 1, 1)
            speaker_obs = speaker_obs.view(3000, -1)

            likelihoods = model.speaker.get_token_dist(speaker_obs)
            top_comm_idx = np.argmax(likelihoods)
            top_likelihood = likelihoods[top_comm_idx]
            label = top_comm_idx if top_likelihood > 0.4 else -1
            labels.append(label)
    features = np.vstack(dataset)
    label_np = np.reshape(np.array(labels), (-1, 1))
    all_np = np.hstack([label_np, features])
    regrouped_data = []
    plot_labels = []
    plot_mean = False
    for c in np.unique(labels):
        ix = np.where(all_np[:, 0] == c)
        matching_features = np.vstack(all_np[ix, 1:])
        averaged = np.mean(matching_features, axis=0, keepdims=True)
        plot_features = averaged if plot_mean else matching_features
        regrouped_data.append(plot_features)
        plot_labels.append(c)
    plot_naming(regrouped_data, viz_method='mds', labels=plot_labels, savepath=basepath + 'training_mds')
    plot_naming(regrouped_data, viz_method='tsne', labels=plot_labels, savepath=basepath + 'training_tsne')



def eval_model_pragmatics(model, vae, comm_dim, data, viz_data, glove_data, num_cand_to_metrics, savepath,
               fieldname, calculate_complexity=False, plot_comms_flag=False, alignment_dataset=None, save_model=True):
    # Create a directory to save information, models, etc.
    
    if not os.path.exists(save_eval_path + 'pragmatics/'):
        os.makedirs(save_eval_path + 'pragmatics/')
    if calculate_complexity:
        test_complexity = get_info(model, data, targ_dim=comm_dim, glove_data=glove_data, num_epochs=200)
        print("Test complexity", test_complexity)
    else:
        test_complexity = None
        val_complexity = None
    
    eval_batch_size = 256
    complexities = [test_complexity]
    for set_distinction in [True]:
        for feature_idx, data in enumerate([data]):
            for num_candidates in num_cand_to_metrics.get(set_distinction).keys():
                settings.distinct_words = set_distinction
                acc, recons = evaluate_pragmatics(model, data, eval_batch_size, vae, glove_data, fieldname=fieldname, num_dist=num_candidates - 1)
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
    plot_metrics(comm_accs, labels, epoch_idxs, save_eval_path + 'pragmatics/')
#    plot_metrics(regressions, ['r2 score'], epoch_idxs, save_eval_path + 'regression_')

    # Visualize some of the communication
    try:
        if plot_comms_flag:
            plot_comms(model, viz_data['features'], basepath)
    except AssertionError:
        print("Can't plot comms for whatever reason (e.g., continuous communication makes categorizing hard)")
    # Save the model and metrics to files.
    for feature_idx, label in enumerate(['test']):
        for set_distinction in num_cand_to_metrics.keys():
            for num_candidates in sorted(num_cand_to_metrics.get(set_distinction).keys()):
                metric = num_cand_to_metrics.get(set_distinction).get(num_candidates)[feature_idx]
                metric.to_file(save_eval_path + 'pragmatics/' + "_".join([label, str(set_distinction), str(num_candidates), "metrics"]))
    if not save_model:
        return
    torch.save(model.state_dict(), save_eval_path + 'pragmatics/model.pt')



def eval_model_lexsem(model, vae, comm_dim, data, viz_data, glove_data, num_cand_to_metrics, savepath,
               fieldname, calculate_complexity=False, plot_comms_flag=False, alignment_dataset=None, save_model=True):
    # Create a directory to save information, models, etc.

    if not os.path.exists(save_eval_path + 'lexsem/'):
        os.makedirs(save_eval_path + 'lexsem/')
    if calculate_complexity:
        test_complexity = get_info(model, data, targ_dim=comm_dim, glove_data=glove_data, num_epochs=200)
        print("Test complexity", test_complexity)
    else:
        test_complexity = None
        val_complexity = None

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
            plot_comms(model, viz_data['features'], basepath)
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




def run():
    if settings.see_distractors_pragmatics:
        num_imgs = 3 if settings.with_ctx_representation else 2
    else:
        num_imgs = 1 if not settings.see_distractor else (num_distractors + 1)
    if speaker_type == 'cont':
        speaker = MLP(feature_len, c_dim, num_layers=3, onehot=False, variational=variational, num_imgs=num_imgs)
    elif speaker_type == 'onehot':
        speaker = MLP(feature_len, c_dim, num_layers=3, onehot=True, variational=variational, num_imgs=num_imgs)
    elif speaker_type == 'vq':
        # speaker = VQ(feature_len, c_dim, num_layers=3, num_protos=1763, num_simultaneous_tokens=1, variational=variational, num_imgs=num_imgs)
        speaker = VQ(feature_len, c_dim, num_layers=3, num_protos=32, num_simultaneous_tokens=8, variational=variational, num_imgs=num_imgs)
    if settings.see_distractors_pragmatics:
        listener = ListenerPragmatics(feature_len + num_imgs * feature_len, num_distractors+1, num_imgs=num_imgs)
    else:
        listener = Listener(feature_len)
    decoder = Decoder(c_dim, feature_len, num_layers=3, num_imgs=num_imgs)
    model = Team(speaker, listener, decoder)
    model.to(settings.device)

    data = get_feature_data(t_features_filename, excluded_ids=excluded_ids)
    data = data.sample(frac=1) # Shuffle the data.
    train_data, test_data, val_data = np.split(data.sample(frac=1, random_state=46), 
                                        [int(.7*len(data)), int(.9*len(data))])
    train_data, test_data, val_data = train_data.reset_index(), test_data.reset_index(), val_data.reset_index()
    train_data, test_data, val_data = train_data.sample(frac=1).reset_index(), test_data.sample(frac=1).reset_index(), val_data.sample(frac=1).reset_index() 
    print("Len train set:", len(train_data), "Len val set:", len(val_data), "Len test set:", len(test_data))
   
    viz_data = train_data  # For debugging, it's faster to just reuse datasets
    model.load_state_dict(torch.load(model_to_eval_path + '/model.pt'))
    num_cand_to_metrics = {True: {2: []}}
    for empty_list in num_cand_to_metrics.get(True).values():
            empty_list.extend([PerformanceMetrics()])
    eval_model_pragmatics(model, vae_model, c_dim, test_data, viz_data, glove_data, num_cand_to_metrics, save_eval_path, fieldname='topname', calculate_complexity=do_calc_complexity, plot_comms_flag=do_plot_comms)
    eval_model_lexsem(model, vae_model, c_dim, test_data, viz_data, glove_data, num_cand_to_metrics, save_eval_path, fieldname='topname', calculate_complexity=do_calc_complexity, plot_comms_flag=do_plot_comms)


if __name__ == '__main__':
    feature_len = 512
    settings.see_distractor = False
    settings.see_distractors_pragmatics = True
    settings.with_ctx_representation = True
    settings.eval_someRE = False
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
    settings.alpha = 1
    settings.kl_weight = 0.0  # For cont
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
    np.random.seed(0)
    settings.embedding_cache = {}
    settings.sample_first = True
    speaker_type = 'vq'  # Options are 'vq', 'cont', or 'onehot'
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    save_loc = 'src/saved_models/' + speaker_type + '/seed' + str(seed) + '/'
    glove_data = get_glove_vectors(32)
    model_to_eval_path = 'src/saved_models/with_ctx/vq/seed0/2999' if settings.with_ctx_representation else 'src/saved_models/without_ctx/vq/seed0/2999'
    save_eval_path = model_to_eval_path + '/evaluation/'
    run()

