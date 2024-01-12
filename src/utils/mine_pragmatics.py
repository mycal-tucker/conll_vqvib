import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import src.settings as settings
from src.data_utils.helper_fns import gen_batch


class Net(nn.Module):
    def __init__(self, input1_size, input2_size):
        super(Net, self).__init__()
        self.fc_1 = nn.Linear(input1_size + input2_size, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, 128)
        self.fc_4 = nn.Linear(128, 1)

    def forward(self, x, y):
        catted = torch.hstack((x, y))
        h1 = F.relu(self.fc_1(catted))
        h2 = F.relu(self.fc_2(h1))
        h3 = F.relu(self.fc_3(h2))
        output = self.fc_4(h3)
        return output


# we trick the model to always output the glove embedding of the human topname, and calculate complexity
def get_info_humans(model, dataset, targ_dim, glove_data=None, num_epochs=200, batch_size=1024):
    # Define a network that takes in the two variables to calculate the MI of.
    if settings.see_distractors_pragmatics:
        if settings.with_ctx_representation:
            mine_net = Net(512 * (settings.num_distractors+2), targ_dim)
        else:
            mine_net = Net(512 * (settings.num_distractors+1), targ_dim)
    else:
        mine_net = Net(512, targ_dim)
    mine_net.to(settings.device)
    optimizer = optim.Adam(mine_net.parameters())
    running_loss = 0
    for epoch in range(num_epochs):
        speaker_obs, _, _, glove_embeds = gen_batch(dataset, batch_size, p_notseedist=1, fieldname='topname', glove_data=glove_data)
        if glove_embeds is None:  # If there was no glove embedding for that word.
            continue
        targ_var = torch.Tensor(glove_embeds).to(settings.device) # human name

        # Shuffle the target variable so we can get a marginal of sorts.
        targ_shuffle = torch.Tensor(np.random.permutation(targ_var.cpu().numpy())).to(settings.device)
        optimizer.zero_grad()
        
        if settings.see_distractors_pragmatics:
            speaker_obs = speaker_obs.view(batch_size, -1)
        pred_xy = mine_net(speaker_obs, targ_var)
        pred_x_y = mine_net(speaker_obs, targ_shuffle)
        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss = -ret  # maximize
        loss.backward()
        optimizer.step()
        running_loss = 0.95 * running_loss + 0.05 * loss.item()
        #if epoch % 100 == 0:
        #    print(running_loss)

    # Now evaluate the mutual information instead of actively training.
    summed_loss = 0
    num_eval_epochs = 20
    for epoch in range(num_eval_epochs):
        speaker_obs, _, _, glove_embeds = gen_batch(dataset, batch_size, p_notseedist=1, fieldname='topname', glove_data=glove_data) 
        targ_var = torch.Tensor(glove_embeds).to(settings.device) 
        targ_shuffle = torch.Tensor(np.random.permutation(targ_var.cpu().numpy())).to(settings.device)
        optimizer.zero_grad()

        if settings.see_distractors_pragmatics:
            speaker_obs = speaker_obs.view(batch_size, -1)
        pred_xy = mine_net(speaker_obs, targ_var)
        pred_x_y = mine_net(speaker_obs, targ_shuffle)
        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        summed_loss += ret.item()
    mutual_info = summed_loss / num_eval_epochs
    
    return mutual_info




def get_info_lexsem(model, dataset, targ_dim, glove_data=None, num_epochs=200, batch_size=1024):
    # Define a network that takes in the two variables to calculate the MI of.
    if settings.see_distractors_pragmatics:
        if settings.with_ctx_representation:
            mine_net = Net(512 * (settings.num_distractors+2), targ_dim)
        else:
            mine_net = Net(512 * (settings.num_distractors+1), targ_dim)
    else:
        mine_net = Net(512, targ_dim)
    mine_net.to(settings.device)
    optimizer = optim.Adam(mine_net.parameters())
    running_loss = 0
    for epoch in range(num_epochs):
        speaker_obs, _, _, _ = gen_batch(dataset, batch_size, p_notseedist=1, fieldname='topname', glove_data=glove_data)
        with torch.no_grad():
            targ_var, _, _ = model.speaker(speaker_obs)  # Communication

        # Shuffle the target variable so we can get a marginal of sorts.
        targ_shuffle = torch.Tensor(np.random.permutation(targ_var.cpu().numpy())).to(settings.device)
        optimizer.zero_grad()

        if settings.see_distractors_pragmatics:
            speaker_obs = speaker_obs.view(batch_size, -1)
        pred_xy = mine_net(speaker_obs, targ_var)
        pred_x_y = mine_net(speaker_obs, targ_shuffle)
        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss = -ret  # maximize
        loss.backward()
        optimizer.step()
        running_loss = 0.95 * running_loss + 0.05 * loss.item()
        #if epoch % 100 == 0:
        #    print(running_loss)
    # Now evaluate the mutual information instead of actively training.
    summed_loss = 0
    num_eval_epochs = 20
    for epoch in range(num_eval_epochs):
        #speaker_obs, _, _, _ = gen_batch(dataset, 1024, fieldname='topname', glove_data=glove_data)
        speaker_obs, _, _, _ = gen_batch(dataset, batch_size, p_notseedist=1, fieldname='topname', glove_data=glove_data)
        with torch.no_grad():
            targ_var, _, _ = model.speaker(speaker_obs)  # Communication
        targ_shuffle = torch.Tensor(np.random.permutation(targ_var.cpu().numpy())).to(settings.device)
        optimizer.zero_grad()

        if settings.see_distractors_pragmatics:
            speaker_obs = speaker_obs.view(batch_size, -1)
        pred_xy = mine_net(speaker_obs, targ_var)
        pred_x_y = mine_net(speaker_obs, targ_shuffle)
        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        summed_loss += ret.item()
    mutual_info = summed_loss / num_eval_epochs
    return mutual_info



# TENTATIVELY, VERY UGLY, NOT SURE IF CORRECT
def get_cond_info(model, dataset, targ_dim, p_notseedist, glove_data=None, num_epochs=200, batch_size=1024):
    # Define a network that takes in the two variables to calculate the MI of.
    if settings.with_ctx_representation:
        mine_net = Net(512 * (settings.num_distractors+2), targ_dim)
    else:
        mine_net = Net(512 * (settings.num_distractors+1), targ_dim)
    mine_net.to(settings.device)
    optimizer = optim.Adam(mine_net.parameters())
    running_loss = 0

    # word = x
    # target = y
    # distractor = z
    # goal = p(x,y|z)

    for epoch in range(num_epochs):
        speaker_obs, _, _, _ = gen_batch(dataset, batch_size, p_notseedist=p_notseedist, fieldname='topname', glove_data=glove_data)
    
        # Get communication
        with torch.no_grad():
            comm, _, _ = model.speaker(speaker_obs)  

        # Shuffle the target variable so we can get a marginal of sorts.
        comm_shuffle = torch.Tensor(np.random.permutation(comm.cpu().numpy())).to(settings.device)
        optimizer.zero_grad()
        speaker_obs = speaker_obs.view(batch_size, -1)

        pred_xy = mine_net(speaker_obs, comm)
        pred_x_y = mine_net(speaker_obs, comm_shuffle)

        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss = -ret  # maximize
        loss.backward()
        optimizer.step()
        running_loss = 0.95 * running_loss + 0.05 * loss.item()
        if epoch % 100 == 0:
            print(running_loss)


    summed_loss_xz = 0
    summed_loss_xz1 = 0
    num_eval_epochs = 20
    
    for epoch in range(num_eval_epochs):
        speaker_obs_batch, _, _, _ = gen_batch(dataset, batch_size, p_notseedist=p_notseedist, fieldname='topname', glove_data=glove_data)
        
        # Compute I(X; Z1) = I(X; Y,Z)
        with torch.no_grad():
            comm_z1, _, _ = model.speaker(speaker_obs_batch)
        comm_z1_shuffle = torch.Tensor(np.random.permutation(comm_z1.cpu().numpy())).to(settings.device)
        optimizer.zero_grad()
        speaker_obs = speaker_obs_batch.view(batch_size, -1)

        pred_xz1 = mine_net(speaker_obs, comm_z1)
        pred_x_z1 = mine_net(speaker_obs, comm_z1_shuffle)
        ret = torch.mean(pred_xz1) - torch.log(torch.mean(torch.exp(pred_x_z1)))
        
        summed_loss_xz1 += ret.item()

        # Compute I(X;Z)
        # swap target and distractor: now the distractor is in target position
        speaker_obs_batch[:, [0, 1]] = speaker_obs_batch[:, [1, 0]]
        # mask the target
        mask = torch.ones_like(speaker_obs_batch)
        mask[:, 1, :] = 0
        speaker_obs_batch = speaker_obs_batch * mask
        #speaker_obs = speaker_obs_batch.view(batch_size, -1)
        
        # Get communication
        with torch.no_grad():
            comm_z, _, _ = model.speaker(speaker_obs_batch)

        comm_z_shuffle = torch.Tensor(np.random.permutation(comm_z.cpu().numpy())).to(settings.device)
        optimizer.zero_grad()
        speaker_obs = speaker_obs_batch.view(batch_size, -1)

        pred_xz = mine_net(speaker_obs, comm_z)
        pred_x_z = mine_net(speaker_obs, comm_z_shuffle)
        ret = torch.mean(pred_xz) - torch.log(torch.mean(torch.exp(pred_x_z)))
        summed_loss_xz += ret.item()

    mutual_info_XZ = summed_loss_xz / num_eval_epochs
    mutual_info_XZ1 = summed_loss_xz1 / num_eval_epochs
   
    conditional_MI = mutual_info_XZ1 - mutual_info_XZ
    
    return conditional_MI

        

