import numpy as np
import random
import torch
import torch.optim as optim

import src.settings as settings
from src.data_utils.helper_fns import gen_batch
from src.data_utils.read_data import get_feature_data, get_glove_vectors
from src.models.vae import VAE


def run():
    model = VAE(512, 32)
    model.to(settings.device)
    train_data = get_feature_data(features_filename)
    optimizer = optim.Adam(model.parameters())
    running_loss = 0
    for epoch in range(num_epochs):
        features, _, _, _ = gen_batch(train_data, batch_size, p_notseedist=0, glove_data=glove_data, fieldname='topname')  # Fieldname doesn't matter
        optimizer.zero_grad()
        reconstruction, loss = model(features)
        loss.backward()
        optimizer.step()
        running_loss = 0.95 * running_loss + 0.05 * loss.item()
        if epoch % 100 == 0:
            print("Epoch", epoch, "of", num_epochs)
            print("Loss", running_loss)
    torch.save(model.state_dict(), savepath)


if __name__ == '__main__':
    settings.see_distractors_pragmatics = False
    features_filename = 'src/data/t_features.csv'
    savepath = 'src/saved_models/vae0.001.pt'
    glove_data = get_glove_vectors(32)
    num_epochs = 10000
    batch_size = 32
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.num_distractors = 1
    settings.embedding_cache = {}
    settings.distinct_words = False
    
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run()
