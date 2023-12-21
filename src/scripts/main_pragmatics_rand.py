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
from src.utils.mine_pragmatics import get_cond_info
from src.utils.plotting import plot_metrics, plot_naming, plot_scatter
from src.utils.performance_metrics import PerformanceMetrics

import time

from src.data_utils.read_data import get_glove_vectors



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


def evaluate(model, dataset, batch_size, p_notseedist, vae, glove_data, fieldname, num_dist=None):
    model.eval()
    num_test_batches = 10
    num_correct = 0
    total_recons_loss = 0
    num_total = 0
    for _ in range(num_test_batches):
        with torch.no_grad():
            speaker_obs, listener_obs, labels, _ = gen_batch(dataset, batch_size, fieldname, p_notseedist, vae=vae, glove_data=glove_data, see_distractors=settings.see_distractor, num_dist=num_dist)
            outputs, _, _, recons = model(speaker_obs, listener_obs)
            recons = torch.squeeze(recons, dim=1)
        pred_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        num_correct += np.sum(pred_labels == labels.cpu().numpy())
        num_total += pred_labels.size
        total_recons_loss += torch.mean(((speaker_obs[:, 0:1, :] - recons[:, 0:1, :]) ** 2)).item()
    acc = num_correct / num_total
    total_recons_loss = total_recons_loss / num_test_batches
    print("Evaluation accuracy", acc)
    print("Evaluation recons loss", total_recons_loss)
    return acc, total_recons_loss



#def evaluate_with_english(model, dataset, vae, embed_to_tok, glove_data, fieldname, eng_fieldname=None, use_top=True, num_dist=None, eng_dec=None, eng_list=None, tok_to_embed=None, use_comm_idx=False, comm_map=None):
    # topwords = dataset[fieldname]
#    unique_topnames = get_unique_by_field(dataset, 'topname')
#    if eng_fieldname is None:
#        eng_fieldname = fieldname
#    topwords = dataset[eng_fieldname]
#    responses = dataset['responses']
#    model.eval()
#    num_nosnap_correct = 0
#    num_snap_correct = 0
#    num_total = 0
#    num_unmatched = 0
#    eng_correct = 0
    # eval_dataset_size = len(dataset)
#    eval_dataset_size = 100  # How many do test inputs? Time scales linearly, basically.
    # print("Evaluating English performance on", eval_dataset_size, "examples")
    # TODO: batching this stuff could make things way faster.
#    for targ_idx in range(eval_dataset_size):
#        speaker_obs, listener_obs, labels, _ = gen_batch(dataset, 1, fieldname, vae=vae, see_distractors=settings.see_distractor, glove_data=glove_data,
#                                            num_dist=num_dist, preset_targ_idx=targ_idx)
#        if labels is None:  # If there was no glove embedding for that word.
#            continue
#        labels = labels.cpu().numpy()
        # Can pick just the topname, or one of the random responses.
#        topword = topwords[targ_idx]
#        if not use_top:
            # all_responses = list(responses[targ_idx])
#            all_responses = responses[targ_idx]
#            words = []
#            probs = []
#            for k, v in all_responses.items():
#                parsed_word = k.split(' ')
#                if len(parsed_word) > 1:
                    # Skip "words" like "tennis player" etc. because they won't be in glove data
#                    continue
#                words.append(k)
#                probs.append(v)
#            if len(words) == 0:
                # Failed to find any legal words (e.g., all like "tennis player")
#                continue
#            total = np.sum(probs)
#            probs = [p / total for p in probs]
#            word = np.random.choice(words, p=probs)

            # if len(all_responses) == 1:
            #     continue  # Can't use a synonym if we only know one word
            # # This isn't right. We can have the topname
            # word = topword
            # num_tries = 0
            # while word in unique_topnames and num_tries < 10:
            #     word = random.choice(list(responses[targ_idx]))
            #     num_tries += 1
            # if num_tries == 10:
            #     continue  # Failed to find a non topname response quickly enough.
#        else:
#            word = topword
        # Create the embedding, and then the token from the embedding.
#        try:
#            embedding = get_glove_embedding(glove_data, word).to_numpy()
#        except AttributeError:  # If we don't have an embedding for the word, we get None back, so let's just move on.
#            continue
#        token = embed_to_tok.predict(np.expand_dims(embedding, 0))
#        with torch.no_grad():
#            tensor_token = torch.Tensor(token).to(settings.device)
#            nosnap_prediction = model.pred_from_comms(tensor_token, listener_obs)
            # Snap the token to the nearest acceptable communication token
#            if isinstance(model.speaker, ProtoNetwork):
#                snap_prediction = None
#            else:
#                tensor_token = model.speaker.snap_comms(tensor_token)
#                snap_prediction = model.pred_from_comms(tensor_token, listener_obs)
#        nosnap_pred_labels = np.argmax(nosnap_prediction.detach().cpu().numpy(), axis=1)
#        num_nosnap_correct += np.sum(nosnap_pred_labels == labels)
#        if snap_prediction is not None:
#            snap_pred_labels = np.argmax(snap_prediction.detach().cpu().numpy(), axis=1)
#        else:
#            snap_pred_labels = -1
#        num_snap_correct += np.sum(snap_pred_labels == labels)
#        num_total += num_nosnap_correct.size

        # If parameters are provided, also evaluate EC -> English
#        if eng_dec is None:
#            continue
#        with torch.no_grad():
#            ec_comm, _, _ = model.speaker(speaker_obs)
            # comm_id = model.speaker.get_comm_id(ec_comm).detach().cpu().numpy()
            # relevant_comm = ec_comm if not use_comm_idx else comm_id

#            ec_key = tuple(np.round(ec_comm.detach().cpu().squeeze(dim=0).numpy(), 3))

#            if use_comm_idx and comm_map.get(ec_key) is None:
#                print("Using comm idx but couldn't find entry")
#                num_unmatched += 1
#                eng_correct += 0.5
#                continue
            # Convert comm id to onehot
#            ec_comm = ec_comm.detach().cpu().numpy()
#            if not use_comm_idx:
#                eng_comm = tok_to_embed.predict(ec_comm)
#                eng_comm = torch.Tensor(eng_comm).to(settings.device)
#            else:
                # Just look up the English embedding for the most common english word associated with this ec comm.
#                word = comm_map.get(ec_key)
#                eng_comm = torch.Tensor(get_glove_embedding(glove_data, word).to_numpy()).to(settings.device)
#            recons = eng_dec(eng_comm)
#            prediction = eng_list(recons, listener_obs)
#            pred_label = np.argmax(prediction.detach().cpu().numpy(), axis=1)
#            eng_correct += np.sum(pred_label == labels)
    # print("Percent unmatched ec comms", num_unmatched / num_total)
#    return num_nosnap_correct / num_total, num_snap_correct / num_total, eng_correct / num_total


#def plot_comms(model, dataset, basepath):
#    num_tests = 1000  # Generate lots of samples for the same input because it's not deterministic.
#    labels = []
#    if settings.with_ctx_representation:
#        for f, f_d, f_ctx in zip(dataset['t_features'], dataset['d_features'], dataset['ctx_features']):
#            speaker_obs = np.expand_dims(np.vstack([f] + [f_d] + [f_ctx]), axis=0)
#            speaker_obs = torch.Tensor(np.vstack(speaker_obs).astype(np.float)).to(settings.device)   
#            speaker_obs = speaker_obs.unsqueeze(0)
#            speaker_obs = speaker_obs.repeat(num_tests, 1, 1)
#            speaker_obs = speaker_obs.view(3000, -1)

#            likelihoods = model.speaker.get_token_dist(speaker_obs)
#            top_comm_idx = np.argmax(likelihoods)
#            top_likelihood = likelihoods[top_comm_idx]
#            label = top_comm_idx if top_likelihood > 0.4 else -1
#            labels.append(label)
#    features = np.vstack(dataset)
#    label_np = np.reshape(np.array(labels), (-1, 1))
#    all_np = np.hstack([label_np, features])
#    regrouped_data = []
#    plot_labels = []
#    plot_mean = False
#    for c in np.unique(labels):
#        ix = np.where(all_np[:, 0] == c)
#        matching_features = np.vstack(all_np[ix, 1:])
#        averaged = np.mean(matching_features, axis=0, keepdims=True)
#        plot_features = averaged if plot_mean else matching_features
#        regrouped_data.append(plot_features)
#        plot_labels.append(c)
#    plot_naming(regrouped_data, viz_method='mds', labels=plot_labels, savepath=basepath + 'training_mds')
#    plot_naming(regrouped_data, viz_method='tsne', labels=plot_labels, savepath=basepath + 'training_tsne')




def eval_model(model, vae, comm_dim, train_data, val_data, viz_data, glove_data, num_cand_to_metrics, savepath,
               epoch, fieldname, p_notseedist, calculate_complexity=False, plot_comms_flag=False, alignment_dataset=None, save_model=True):
    # Create a directory to save information, models, etc.
    basepath = savepath + str(epoch) + '/'
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    # Calculate efficiency values like complexity and informativeness.
    # Can estimate complexity by sampling inputs and measuring communication probabilities.
    # get_probs(model.speaker, train_data)
    # Or we can use MINE to estimate complexity and informativeness.
    if calculate_complexity:
        print("Eval complexity over tons of batches!!! FIXME")
        train_complexity = get_cond_info(model, train_data, targ_dim=comm_dim, p_notseedist=p_notseedist, glove_data=glove_data, num_epochs=200)
        #val_complexity = get_cond_info(model, val_features, targ_dim=comm_dim, comm_targ=True)
        val_complexity = None
        print("Train complexity", train_complexity)
        print("Val complexity", val_complexity)
    else:
        train_complexity = None
        val_complexity = None
    # And compare to english word embeddings (doesn't depend on number of distractors)
#    align_data = train_data if alignment_dataset is None else alignment_dataset
#    tok_to_embed, embed_to_tok, tokr2, embr2, _ = get_embedding_alignment(model, align_data, glove_data, fieldname=fieldname)
    eval_batch_size = 256
    val_is_train = len(train_data) == len(val_data)  # Not actually true, but close enough
    if val_is_train:
        print("WARNING: ASSUMING VALIDATION AND TRAIN ARE SAME")

#    distinct_val = settings.distinct_words
    complexities = [train_complexity, val_complexity]
    for set_distinction in [True]:
        for feature_idx, data in enumerate([train_data, val_data]):
            if feature_idx == 1 and val_is_train:
                pass
            for num_candidates in num_cand_to_metrics.get(set_distinction).keys():
                if feature_idx == 1 and val_is_train:
                    pass  # Just save the values from last time.
                else:
                    settings.distinct_words = set_distinction
                    acc, recons = evaluate(model, data, eval_batch_size, p_notseedist, vae, glove_data, fieldname=fieldname, num_dist=num_candidates - 1)
                relevant_metrics = num_cand_to_metrics.get(set_distinction).get(num_candidates)[feature_idx]
                relevant_metrics.add_data(epoch, complexities[feature_idx], -1 * recons, acc, settings.kl_weight)

    # Plot some of the metrics for online visualization
    comm_accs = []
    regressions = []
    labels = []
    epoch_idxs = None
    plot_metric_data = num_cand_to_metrics.get(True)
    for feature_idx, label in enumerate(['train', 'val']):
        for num_candidates in sorted(plot_metric_data.keys()):
            comm_accs.append(plot_metric_data.get(num_candidates)[feature_idx].comm_accs)
#            regressions.append(plot_metric_data.get(num_candidates)[feature_idx].embed_r2)
            labels.append(" ".join([label, str(num_candidates), "utility"]))
            if epoch_idxs is None:
                epoch_idxs = plot_metric_data.get(num_candidates)[feature_idx].epoch_idxs
    plot_metrics(comm_accs, labels, epoch_idxs, basepath=basepath)
#    plot_metrics(regressions, ['r2 score'], epoch_idxs, basepath=basepath + 'regression_')
    # Visualize some of the communication
    try:
        if plot_comms_flag:
            col_name = 't_features' if settings.see_distractors_pragmatics else 'features'
            plot_comms(model, viz_data[col_name], basepath)
    except AssertionError:
        print("Can't plot comms for whatever reason (e.g., continuous communication makes categorizing hard)")
    # Save the model and metrics to files.
    for feature_idx, label in enumerate(['train', 'val']):
        for set_distinction in num_cand_to_metrics.keys():
            for num_candidates in sorted(num_cand_to_metrics.get(set_distinction).keys()):
                metric = num_cand_to_metrics.get(set_distinction).get(num_candidates)[feature_idx]
                metric.to_file(basepath + "_".join([label, str(set_distinction), str(num_candidates), "metrics"]))
    if not save_model:
        return
    torch.save(model.state_dict(), basepath + 'model.pt')
    torch.save(model, basepath + 'model_obj.pt')



def train(model, train_data, val_data, viz_data, glove_data, p_notseedist, utility, vae, savepath, logs_dir, comm_dim, fieldname, batch_size=1024, burnin_epochs=500, val_period=200, plot_comms_flag=False, calculate_complexity=False):
    unique_topnames, _ = get_unique_labels(train_data)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.0001)  
    optimizer = optim.Adam(model.parameters(), lr=0.001)
       
    idx = 0
    while idx != len(settings.alphas):
        converged = False
        alpha = settings.alphas[idx] # we take the new alpha
        savepath_new = savepath + "alpha"+str(alpha) + '/'
        epoch = 0
        running_acc = 0
        running_mse = 0
        num_cand_to_metrics = {True: {2: []}}
        for set_distinct in [True]:
            for empty_list in num_cand_to_metrics.get(set_distinct).values():
                empty_list.extend([PerformanceMetrics(), PerformanceMetrics()])  # Train metrics, validation metrics        
        complexity_weight = settings.kl_weight_unnorm
        print("unnorm complexity:", complexity_weight)
        informativeness_weight = alpha # we vary this iterating over settings.alphas
        print("unnorm informativeness:", informativeness_weight)
        utility_weight = utility
        print("unnorm utility:", utility_weight)

        # normalize loss weights
        norm_weights = normalize_and_adjust([complexity_weight, informativeness_weight, utility_weight])
        trained_weights_dir = logs_dir + "done_weights.json"
        try:
            with open(trained_weights_dir, "r") as file:
                done = json.load(file)
                done_triplets = list(done.values())
        except FileNotFoundError:
            done_triplets = []
            done = {}
            os.makedirs(logs_dir, exist_ok=True)
        if str(norm_weights) in done_triplets: # this triplet has been trained already
            idx += 1 # new alpha
            print("have already:", str(norm_weights))

        else: # if we have to train this triplet
            key = str([complexity_weight, informativeness_weight, utility_weight])
            done[key] = str(norm_weights)
            print("new", str(norm_weights))
            with open(trained_weights_dir, 'w') as f:
                json.dump(done, f, indent=4)

            norm_weights = normalize_and_adjust([complexity_weight, informativeness_weight, utility_weight])
            settings.kl_weight = norm_weights[0]
            print("norm complexity:", settings.kl_weight) # complexity weight (normalized)
            informativeness_weight = norm_weights[1]
            print("norm informativeness:", informativeness_weight)
            utility_weight = norm_weights[2]
            print("norm utility:", utility_weight)

            while converged == False:
                epoch += 1

                speaker_obs, listener_obs, labels, _ = gen_batch(train_data, batch_size, fieldname, p_notseedist, vae=vae, glove_data=glove_data, see_distractors=settings.see_distractor)
                #start_time = time.time()
                optimizer.zero_grad()
                outputs, speaker_loss, info, recons = model(speaker_obs, listener_obs)
            
                error = criterion(outputs, labels)
                utility_loss = utility_weight * error # copy to store later untouched
                loss = utility_weight * error

                if len(speaker_obs.shape) == 2:
                    speaker_obs = torch.unsqueeze(speaker_obs, 1)

                # we only care about target reconstruction
                recons_loss = torch.mean(((speaker_obs[:, 0:1, :] - recons[:, 0:1, :]) ** 2))
                loss += informativeness_weight * recons_loss
                loss += speaker_loss

                loss.backward()
                optimizer.step()

                #end_time = time.time()
                # print("Elapsed time", end_time - start_time)

                # Metrics
                pred_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                num_correct = np.sum(pred_labels == labels.cpu().numpy())
                num_total = pred_labels.size
                running_acc = running_acc * 0.95 + 0.05 * num_correct / num_total
                running_mse = running_mse * 0.95 + 0.05 * recons_loss.item()
           
                if epoch % val_period == val_period - 1: # if it's a validation epoch
                    eval_model(model, vae, comm_dim, train_data, val_data, viz_data, glove_data, num_cand_to_metrics, savepath_new, epoch, fieldname, p_notseedist, calculate_complexity=calculate_complexity and epoch == num_epochs - 1, plot_comms_flag=plot_comms_flag)
                
                if epoch == n_epochs-1: # if we are done with the training
                    converged = True
                    # save model info
                    json_file = objective_json+"objective" + str(idx_job) + ".json"
                    if os.path.exists(json_file):
                        with open(json_file, 'r') as f:
                            existing_params = json.load(f)
                    else:
                        existing_params = {}

                    dic = {}
                    dic["inf_weight"+str(alpha)] = {"objective": loss.item(),
                                                #"unw objective": unweighted_loss.item(),
                                                "speaker loss": speaker_loss.item(), # weighted already
                                                "recons loss": informativeness_weight * recons_loss.item(),
                                                "unw recons loss": recons_loss.item(),
                                                "utility loss": utility_loss.item(), # weighted already
                                                "unw utility loss": error.item(),
                                                "convergence epoch": epoch} 

                    if "utility"+str(utility) in existing_params.keys():
                        existing_params["utility"+str(utility)].update(dic)
                    else:
                        existing_params["utility"+str(utility)] = dic

                    with open(json_file, 'w') as f:
                        json.dump(existing_params, f, indent=4)
                        
                    idx += 1 # new alpha
                    break

                else: # not validation epoch and not done with the training
                    continue 


def run():
    if settings.see_distractors_pragmatics:
        num_imgs = 3 if settings.with_ctx_representation else 2
    else:
        num_imgs = 1 if not settings.see_distractor else (num_distractors + 1)

    data = get_feature_data(t_features_filename, excluded_ids=excluded_ids)
    data = data.sample(frac=1, random_state=seed) # Shuffle the data.
    train_data, test_data, val_data = np.split(data.sample(frac=1, random_state=seed), 
                                        [int(.7*len(data)), int(.9*len(data))])
    train_data, test_data, val_data = train_data.reset_index(), test_data.reset_index(), val_data.reset_index()
    train_data, test_data, val_data = train_data.sample(frac=1, random_state=seed).reset_index(), test_data.sample(frac=1, random_state=seed).reset_index(), val_data.sample(frac=1, random_state=seed).reset_index() 
    print("Len train set:",len(train_data), "Len val set:", len(val_data), "Len test set:", len(test_data))
    viz_data = train_data  # For debugging, it's faster to just reuse datasets
     
    #print("dropout:", settings.dropout)
    print("context:", settings.with_ctx_representation)

    for u in settings.utilities:
        starting_utility = u
        speaker = VQ(feature_len, c_dim, num_layers=3, num_protos=settings.num_protos, num_simultaneous_tokens=1, variational=variational, num_imgs=num_imgs)
        listener = ListenerPragmaticsCosines(feature_len)
        decoder = Decoder(c_dim, feature_len, num_layers=3, num_imgs=num_imgs)
        model = Team(speaker, listener, decoder)
        model.to(settings.device)

        folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
        folder_utility_type = "utility"+str(starting_utility)+"/"
        save_loc = 'src/saved_models/' + str(settings.num_protos) + "/random_init/"+ folder_ctx + 'kl_weight' + str(settings.kl_weight_unnorm) + '/seed' + str(seed) + '/' + folder_utility_type
        json_file_path = "src/saved_models/" + str(settings.num_protos) + "/random_init/" + folder_ctx + 'kl_weight' + str(settings.kl_weight_unnorm) + '/seed' + str(seed) + '/'
        train(model, train_data, val_data, viz_data, glove_data=glove_data, utility=starting_utility, p_notseedist=0, vae=vae_model, savepath=save_loc, logs_dir=json_file_path, comm_dim=c_dim, fieldname='topname', batch_size=b_size, burnin_epochs=num_burnin, val_period=v_period, plot_comms_flag=do_plot_comms, calculate_complexity=do_calc_complexity)


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
    v_period = 200  # How often to test on the validation set and calculate various info metrics.
    num_burnin = 500
    b_size = 1024
    c_dim = 128
    variational = True
    settings.num_protos = 3000 # 442 is the number of topnames in MN 
    # Measuring complexity takes a lot of time. For debugging other features, set to false.
    do_calc_complexity = False
    do_plot_comms = False

    settings.alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 2, 3, 4, 5, 7, 10, 20, 40, 88, 140, 200]
    settings.alphas.reverse()
    #settings.utilities = [0, 0.1, 0.2, 0.3]
    #settings.utilities = [0.4, 0.5, 0.7, 0.9]
    settings.utilities = [1, 1.5, 2, 3]
    #settings.utilities = [4, 5, 7, 10]
    #settings.utilities = [20, 40, 88, 140, 200]
    
    idx_job = 2
    settings.kl_weight_unnorm = 1.0 # complexity
    look_back = 10
    epsilon_convergence = 0.0001
    n_epochs = 5000
   
    settings.kl_incr = 0.0 # complexity increase
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
    excluded_ids = [str(i) for i in merged_tmp['vg_image_id']]
    print(len(excluded_ids))
            
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

