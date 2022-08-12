import src.settings as settings
from src.data_utils.helper_fns import get_unique_labels, get_all_embeddings
from src.data_utils.read_data import get_feature_data, get_glove_vectors
from src.models.decoder import Decoder
from src.models.listener import Listener
from src.models.mlp import MLP
from src.models.team import Team
from src.models.vq import VQ
import numpy as np
from src.models.vae import VAE
import torch
import random
from src.scripts.main import train


def run_trial():
    # Load data
    glove_data = get_glove_vectors(comm_dim)
    train_data = get_feature_data(features_filename, selected_fraction=train_fraction)
    train_topnames, train_responses = get_unique_labels(train_data)
    val_data = get_feature_data(features_filename, excluded_names=train_responses)
    # val_data = train_data  # FIXME
    if len(train_data) < 1000 or len(val_data) < 1000:
        return
    # viz_data = get_feature_data(features_filename, desired_names=viz_names, max_per_class=40)
    viz_data = val_data  # Turn off viz data because we don't use it during trials.

    all_embeddings = None if not use_embed_tokens else get_all_embeddings(glove_data, train_topnames)

    # Initialize the agents for training
    num_imgs = 1 if not settings.see_distractor else (num_distractors + 1)
    if speaker_type == 'cont':
        speaker = MLP(feature_len, comm_dim, num_layers=3, onehot=False, variational=variational, num_imgs=num_imgs)
    elif speaker_type == 'onehot':
        speaker = MLP(feature_len, comm_dim, num_layers=3, onehot=True, variational=variational, num_imgs=num_imgs)
    elif speaker_type == 'vq':
        speaker = VQ(feature_len, comm_dim, num_layers=3, num_protos=num_prototypes, specified_tok=all_embeddings, num_simultaneous_tokens=num_tokens,
                     variational=variational, num_imgs=num_imgs)
    listener = Listener(feature_len)
    decoder = Decoder(comm_dim, feature_len, num_layers=3, num_imgs=num_imgs)
    model = Team(speaker, listener, decoder)
    model.to(settings.device)

    train(model, train_data, val_data, viz_data, glove_data, vae=vae, savepath=savepath, comm_dim=comm_dim, num_epochs=num_epochs,
          batch_size=batch_size, burnin_epochs=num_burnin, val_period=val_period,
          plot_comms_flag=False, calculate_complexity=True)


if __name__ == '__main__':
    feature_len = 512
    settings.see_distractor = False
    num_distractors = 1
    num_epochs = 5000  # 1000 is way too short, but it's quick for debugging.
    num_burnin = 2000
    val_period = 500  # How often to test on the validation set and calculate various info metrics.
    batch_size = 1024
    comm_dim = 64
    features_filename = 'data/features_nobox.csv'

    train_fraction = 0.2
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # settings.kl_weight = 0.001  # For cont
    settings.kl_weight = 0.0002  # For VQ
    settings.kl_incr = 0.00001
    settings.num_distractors = num_distractors
    settings.learned_marginal = False
    settings.embedding_cache = {}
    settings.sample_first = True
    variational = True
    settings.supervision_weight = 0
    settings.hardcoded_vq = False

    use_embed_tokens = False
    if use_embed_tokens:
        assert comm_dim <= 100, "Can't use 100D glove embeddings in greater than 100 comm dim"

    vae = VAE(512, 32)
    vae_beta = 0.001
    vae.load_state_dict(torch.load('saved_models/vae' + str(vae_beta) +'.pt'))
    vae.to(settings.device)

    # num_unique_messages = 3 ** 8
    # num_prototypes = int(num_unique_messages ** (1 / num_tokens))
    # num_prototypes = 32
    num_prototypes = 256

    seeds = [i for i in range(1, 2)]
    # comm_types = ['vq', 'cont']
    comm_types = ['vq']
    for num_tokens in [1]:
        for alpha in [10]:
            settings.alpha = alpha
            for seed in seeds:
                for speaker_type in comm_types:
                    print("Training comm type", speaker_type, "seed", seed, "for", num_tokens, "num tokens and", alpha, "alpha")
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    savepath = 'saved_models/beta' + str(vae_beta) + '/alpha' + str(settings.alpha) + '_' + str(num_tokens) + 'tok/' + speaker_type + '/seed' + str(seed) + '/'
                    run_trial()
