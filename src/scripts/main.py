import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
import random

import src.settings as settings
from src.data_utils.helper_fns import gen_batch, get_glove_embedding, get_unique_labels, get_entry_for_labels
from src.data_utils.read_data import get_feature_data
from src.models.decoder import Decoder
from src.models.listener import Listener
from src.models.team import Team
from src.models.vae import VAE
from src.models.vq import VQ
from src.models.mlp import MLP
from src.utils.mine import get_info
from src.utils.plotting import plot_metrics, plot_naming, plot_scatter
from src.utils.performance_metrics import PerformanceMetrics


def evaluate(model, dataset, batch_size, vae, glove_data, fieldname, num_dist=None):
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
    print("Evaluation accuracy", acc)
    print("Evaluation recons loss", total_recons_loss)
    return acc, total_recons_loss


def evaluate_with_english(model, dataset, vae, embed_to_tok, glove_data, fieldname, use_top=True, num_dist=None, eng_dec=None, eng_list=None,
                          tok_to_embed=None, use_comm_idx=False, comm_map=None):
    topwords = dataset[fieldname]
    responses = dataset['responses']
    model.eval()
    num_nosnap_correct = 0
    num_snap_correct = 0
    num_total = 0
    num_unmatched = 0
    eng_correct = 0
    # eval_dataset_size = len(dataset)
    eval_dataset_size = 1000
    print("Evaluating English performance on", eval_dataset_size, "examples")
    # TODO: batching this stuff could make things way faster. I am pretty confused by why it's so bad.
    for targ_idx in range(eval_dataset_size):
        speaker_obs, listener_obs, labels, _ = gen_batch(dataset, 1, fieldname, vae=vae, see_distractors=settings.see_distractor, glove_data=glove_data,
                                            num_dist=num_dist, preset_targ_idx=targ_idx)
        if labels is None:  # If there was no glove embedding for that word.
            continue
        labels = labels.cpu().numpy()
        # Can pick just the topname, or one of the random responses.
        topword = topwords[targ_idx]
        if not use_top:
            all_responses = list(responses[targ_idx])
            if len(all_responses) == 1:
                continue  # Can't use a synonym if we only know one word
            word = topword
            while word == topword:
                word = random.choice(list(responses[targ_idx]))
        else:
            word = topword
        # Create the embedding, and then the token from the embedding.
        try:
            embedding = get_glove_embedding(glove_data, word).to_numpy()
        except AttributeError:  # If we don't have an embedding for the word, we get None back, so let's just move on.
            continue
        token = embed_to_tok.predict(np.expand_dims(embedding, 0))
        with torch.no_grad():
            tensor_token = torch.Tensor(token).to(settings.device)
            nosnap_prediction = model.pred_from_comms(tensor_token, listener_obs)
            # Snap the token to the nearest acceptable communication token
            tensor_token = model.speaker.snap_comms(tensor_token)
            snap_prediction = model.pred_from_comms(tensor_token, listener_obs)
        nosnap_pred_labels = np.argmax(nosnap_prediction.detach().cpu().numpy(), axis=1)
        num_nosnap_correct += np.sum(nosnap_pred_labels == labels)
        snap_pred_labels = np.argmax(snap_prediction.detach().cpu().numpy(), axis=1)
        num_snap_correct += np.sum(snap_pred_labels == labels)
        num_total += num_nosnap_correct.size

        # If parameters are provided, also evaluate EC -> English
        if eng_dec is None:
            continue
        with torch.no_grad():
            ec_comm, _, _ = model.speaker(speaker_obs)
            # comm_id = model.speaker.get_comm_id(ec_comm).detach().cpu().numpy()
            # relevant_comm = ec_comm if not use_comm_idx else comm_id

            ec_key = tuple(np.round(ec_comm.detach().cpu().squeeze(dim=0).numpy(), 3))

            if use_comm_idx and comm_map.get(ec_key) is None:
                # print("Using comm idx but couldn't find entry")
                num_unmatched += 1
                eng_correct += 0.5
                continue
            # Convert comm id to onehot
            ec_comm = ec_comm.detach().cpu().numpy()
            if not use_comm_idx:
                eng_comm = tok_to_embed.predict(ec_comm)
                eng_comm = torch.Tensor(eng_comm).to(settings.device)
            else:
                # Just look up the English embedding for the most common english word associated with this ec comm.
                word = comm_map.get(ec_key)
                eng_comm = torch.Tensor(get_glove_embedding(glove_data, word).to_numpy()).to(settings.device)
            recons = eng_dec(eng_comm)
            prediction = eng_list(recons, listener_obs)
            pred_label = np.argmax(prediction.detach().cpu().numpy(), axis=1)
            eng_correct += np.sum(pred_label == labels)
    print("Percent unmatched ec comms", num_unmatched / num_total)
    return num_nosnap_correct / num_total, num_snap_correct / num_total, eng_correct / num_total


def plot_comms(model, dataset, basepath):
    num_tests = 1000  # Generate lots of samples for the same input because it's not deterministic.
    labels = []
    for f in dataset:
        speaker_obs = torch.Tensor(np.array(f)).to(settings.device)
        speaker_obs = torch.unsqueeze(speaker_obs, 0)
        speaker_obs = speaker_obs.repeat(num_tests, 1)
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


def get_embedding_alignment(model, dataset, glove_data, fieldname):
    num_tests = 1
    comms = []
    embeddings = []
    features = []
    comm_to_id = {}
    max_num_align_data = 1000   # FIXME
    for f, word in zip(dataset['features'], dataset[fieldname]):
        speaker_obs = torch.Tensor(np.array(f)).to(settings.device)
        speaker_obs = torch.unsqueeze(speaker_obs, 0)
        with torch.no_grad():
            speaker_obs = speaker_obs.repeat(num_tests, 1)
            comm, _, _ = model.speaker(speaker_obs)
        try:
            embedding = get_glove_embedding(glove_data, word).to_numpy()
        except AttributeError:
            continue
        np_comm = np.mean(comm.detach().cpu().numpy(), axis=0)
        comms.append(np_comm)

        dict_comm = tuple(np.round(np_comm, 3))
        matching_entries = comm_to_id.get(dict_comm)
        if matching_entries is None:
            matching_entries = {word: 1}
            comm_to_id[dict_comm] = matching_entries
        else:
            for_word = matching_entries.get(word)
            if for_word is None:
                matching_entries[word] = 1
            else:
                matching_entries[word] += 1

        embeddings.append(embedding)
        features.append(np.array(f))
        if len(embeddings) > max_num_align_data:
            print("Breaking after", len(embeddings), "examples for word alignment")
            break
    comms = np.vstack(comms)
    relevant_comms = comms
    embeddings = np.vstack(embeddings)
    # Now learn two linear regressions to map to/from tokens and word embeddings.
    reg1 = LinearRegression()
    reg1.fit(relevant_comms, embeddings)
    tok_to_embed_r2 = reg1.score(relevant_comms, embeddings)
    print("Tok to word embedding regression score\t\t", tok_to_embed_r2)
    # reg1.fit(comms, comms)  # Actually, if we just want to hardwire it in, "refit" the regression to be 1-1
    reg2 = LinearRegression()
    reg2.fit(embeddings, comms)
    embed_to_tok_r2 = reg2.score(embeddings, comms)
    print("Word embedding to token regression score\t\t", embed_to_tok_r2)
    # reg2.fit(embeddings, embeddings)  # Actually, if we just want to hardwire it in, "refit" the regression to be 1-1
    print("Found", len(comm_to_id), "unique comm vectors!")
    # Gross. Instead of regression score, pass back the number of unique vectors

    # Turn comm_map into comm to most common associated
    comm_modes = {}
    for ec_comm, word_counts in comm_to_id.items():
        best_word = None
        max_count = 0
        for word, count in word_counts.items():
            if count > max_count:
                max_count = count
                best_word = word
        comm_modes[ec_comm] = best_word

    return reg1, reg2, tok_to_embed_r2, len(comm_to_id), comm_modes


def eval_model(model, vae, comm_dim, train_data, val_data, viz_data, glove_data, num_cand_to_metrics, savepath,
               epoch, fieldname, calculate_complexity=False, plot_comms_flag=False, alignment_dataset=None, save_model=True):
    # Create a directory to save information, models, etc.
    basepath = savepath + str(epoch) + '/'
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    # Calculate efficiency values like complexity and informativeness.
    # Can estimate complexity by sampling inputs and measuring communication probabilities.
    # get_probs(model.speaker, train_data)
    # Or we can use MINE to estimate complexity and informativeness.
    if calculate_complexity:
        train_complexity = get_info(model, train_data, targ_dim=comm_dim, glove_data=glove_data, num_epochs=200)
        # val_complexity = get_info(model, val_features, targ_dim=comm_dim, comm_targ=True)
        val_complexity = None
        print("Train complexity", train_complexity)
        print("Val complexity", val_complexity)
    else:
        train_complexity = None
        val_complexity = None
    # And compare to english word embeddings (doesn't depend on number of distractors)
    align_data = train_data if alignment_dataset is None else alignment_dataset
    tok_to_embed, embed_to_tok, tokr2, embr2, _ = get_embedding_alignment(model, align_data, glove_data, fieldname=fieldname)
    eval_batch_size = 256
    val_is_train = len(train_data) == len(val_data)  # Not actually true, but close enough
    if val_is_train:
        print("WARNING: ASSUMING VALIDATION AND TRAIN ARE SAME")

    distinct_val = settings.distinct_words
    complexities = [train_complexity, val_complexity]
    for set_distinction in [True, False]:
        for feature_idx, data in enumerate([train_data, val_data]):
            if feature_idx == 1 and val_is_train:
                pass
            for num_candidates in num_cand_to_metrics.get(set_distinction).keys():
                if feature_idx == 1 and val_is_train:
                    pass  # Just save the values from last time.
                else:
                    settings.distinct_words = set_distinction
                    acc, recons = evaluate(model, data, eval_batch_size, vae, glove_data, fieldname=fieldname, num_dist=num_candidates - 1)
                relevant_metrics = num_cand_to_metrics.get(set_distinction).get(num_candidates)[feature_idx]
                relevant_metrics.add_data(epoch, complexities[feature_idx], -1 * recons, acc, settings.kl_weight,
                                          tokr2, embr2)
    settings.distinct_words = distinct_val
    # Plot some of the metrics for online visualization
    comm_accs = []
    regressions = []
    labels = []
    epoch_idxs = None
    plot_metric_data = num_cand_to_metrics.get(False)
    for feature_idx, label in enumerate(['train', 'val']):
        for num_candidates in sorted(plot_metric_data.keys()):
            comm_accs.append(plot_metric_data.get(num_candidates)[feature_idx].comm_accs)
            regressions.append(plot_metric_data.get(num_candidates)[feature_idx].embed_r2)
            labels.append(" ".join([label, str(num_candidates), "utility"]))
            if epoch_idxs is None:
                epoch_idxs = plot_metric_data.get(num_candidates)[feature_idx].epoch_idxs
    plot_metrics(comm_accs, labels, epoch_idxs, basepath=basepath)
    plot_metrics(regressions, ['r2 score'], epoch_idxs, basepath=basepath + 'regression_')
    # Visualize some of the communication
    try:
        if plot_comms_flag:
            plot_comms(model, viz_data['features'], basepath)
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


def get_supervised_data(train_data, num_examples, glove_data, vae):
    speaker_obs = []
    embs = []
    for i in range(len(train_data)):
        if len(embs) == num_examples:
            break
        # Fieldname doesn't matter if only doing things for the listener
        obs, _, _, emb = gen_batch(train_data, 1, fieldname='topname', glove_data=glove_data, vae=vae, num_dist=0, preset_targ_idx=i)
        if emb[0] is not None:  # It's a list
            embs.append(emb)
            speaker_obs.append(obs.cpu())  # Put on the cpu to get into numpy for moving. Awkward, I know, but whatever.
    speaker_obs = torch.Tensor(np.vstack(speaker_obs)).to(settings.device)
    embs = torch.Tensor(np.vstack(embs)).to(settings.device)
    return [speaker_obs, embs]


def get_super_loss(supervised_data, speaker):
    supervision_crit = nn.MSELoss()
    speaker.eval_mode = True
    comms, _, _ = speaker(supervised_data[0])
    speaker.eval_mode = False
    supervised_loss = supervision_crit(comms, supervised_data[1])
    # print("Supervised loss", supervised_loss)
    return supervised_loss


def train(model, train_data, val_data, viz_data, glove_data, vae, savepath, comm_dim, fieldname, num_epochs=3000, batch_size=1024,
          burnin_epochs=500, val_period=200, plot_comms_flag=False, calculate_complexity=False):
    unique_topnames, _ = get_unique_labels(train_data)
    sup_dataset = pd.concat([get_entry_for_labels(train_data, unique_topnames) for _ in range(3)])
    sup_dataset = sup_dataset.sample(frac=1).reset_index(drop=True)  # Shuffle the data.
    supervised_data = get_supervised_data(sup_dataset, 2, glove_data, vae)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 0.0001 is good for no recons, or for onehot things.
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 0.001 is good for alpha 10 (and is the default)
    running_acc = 0
    running_mse = 0
    # num_cand_to_metrics = {True: {2: [], 16: [], 32: []},  # Don't try many distractors when
    num_cand_to_metrics = {True: {2: []},  # Don't try many distractors when
                           False: {2: [], 16: [], 32: []}}
    for set_distinct in [True, False]:
        for empty_list in num_cand_to_metrics.get(set_distinct).values():
            empty_list.extend([PerformanceMetrics(), PerformanceMetrics()])  # Train metrics, validation metrics
    settings.epoch = 0
    for epoch in range(num_epochs):
        settings.epoch += 1
        if epoch > burnin_epochs:
            settings.kl_weight += settings.kl_incr
        speaker_obs, listener_obs, labels, _ = gen_batch(train_data, batch_size, fieldname, vae=vae, glove_data=glove_data, see_distractors=settings.see_distractor)
        optimizer.zero_grad()
        outputs, speaker_loss, info, recons = model(speaker_obs, listener_obs)

        loss = criterion(outputs, labels)
        supervised_loss = 0
        if settings.supervision_weight != 0:  # Don't even bother computing the loss if it's not used.
            supervised_loss = get_super_loss(supervised_data, model.speaker)
            loss = loss + settings.supervision_weight * supervised_loss
        if len(speaker_obs.shape) == 2:
            speaker_obs = torch.unsqueeze(speaker_obs, 1)
        recons_loss = torch.mean(((speaker_obs - recons) ** 2))
        loss += settings.alpha * recons_loss
        loss += speaker_loss
        # print("Speaker loss fraction:\t", speaker_loss.item() / loss.item())
        # print("Recons loss fraction:\t", settings.alpha * recons_loss.item() / loss.item())
        loss.backward()
        optimizer.step()

        # Metrics
        pred_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        num_correct = np.sum(pred_labels == labels.cpu().numpy())
        num_total = pred_labels.size
        running_acc = running_acc * 0.95 + 0.05 * num_correct / num_total
        running_mse = running_mse * 0.95 + 0.05 * recons_loss.item()
        if epoch % 20 == 0:
            print('epoch', epoch, 'of', num_epochs)
            print("Overall loss", loss.item())
            print("Kl weight", settings.kl_weight)
            print("Running acc", running_acc)
            print("Running mse", running_mse)
            print("Supervised loss", supervised_loss)

        if epoch % val_period == val_period - 1:
            eval_model(model, vae, comm_dim, train_data, val_data, viz_data, glove_data, num_cand_to_metrics,
                       savepath, epoch, fieldname, calculate_complexity=calculate_complexity, plot_comms_flag=plot_comms_flag)

def run():
    num_imgs = 1 if not settings.see_distractor else (num_distractors + 1)
    if speaker_type == 'cont':
        speaker = MLP(feature_len, c_dim, num_layers=3, onehot=False, variational=variational, num_imgs=num_imgs)
    elif speaker_type == 'onehot':
        speaker = MLP(feature_len, c_dim, num_layers=3, onehot=True, variational=variational, num_imgs=num_imgs)
    elif speaker_type == 'vq':
        # speaker = VQ(feature_len, c_dim, num_layers=3, num_protos=1763, num_simultaneous_tokens=1, variational=variational, num_imgs=num_imgs)
        speaker = VQ(feature_len, c_dim, num_layers=3, num_protos=32, num_simultaneous_tokens=8, variational=variational, num_imgs=num_imgs)
    listener = Listener(feature_len)
    decoder = Decoder(c_dim, feature_len, num_layers=3, num_imgs=num_imgs)
    model = Team(speaker, listener, decoder)
    model.to(settings.device)

    train_data = get_feature_data(features_filename, selected_fraction=train_fraction)
    train_topnames, train_responses = get_unique_labels(train_data)
    val_data = get_feature_data(features_filename, excluded_names=train_responses)
    # val_data = train_data  # For debugging, it's faster to just reuse datasets
    # test_data = get_feature_data(features_filename, desired_names=test_classes)
    # viz_data = get_feature_data(features_filename, desired_names=viz_names, max_per_class=40) if do_plot_comms else train_data
    viz_data = train_data  # For debugging, it's faster to just reuse datasets
    train(model, train_data, val_data, viz_data, vae=vae_model, savepath=save_loc, comm_dim=c_dim, num_epochs=n_epochs,
          batch_size=b_size, burnin_epochs=num_burnin, val_period=v_period, plot_comms_flag=do_plot_comms,
          calculate_complexity=do_calc_complexity)


if __name__ == '__main__':
    feature_len = 512
    settings.see_distractor = False
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
    settings.alpha = 0
    # settings.kl_weight = 0.00001  # For cont
    settings.kl_weight = 0.0  # For cont
    settings.kl_incr = 0.0
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.learned_marginal = False
    with_bbox = False
    train_fraction = 0.5
    test_classes = ['couch', 'counter', 'bowl']
    viz_names = ['airplane', 'plane',
                 'animal', 'cow', 'dog', 'cat']
    features_filename = 'data/features.csv' if with_bbox else 'data/features_nobox.csv'
    np.random.seed(0)
    vae_model = VAE(512, 32)
    vae_model.load_state_dict(torch.load('saved_models/vae0.001.pt'))
    vae_model.to(settings.device)
    settings.embedding_cache = {}
    settings.sample_first = True
    speaker_type = 'vq'  # Options are 'vq', 'cont', or 'onehot'
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    save_loc = 'saved_models/' + speaker_type + '/seed' + str(seed) + '/'
    run()
