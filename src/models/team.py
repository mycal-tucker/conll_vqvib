import torch.nn as nn


class Team(nn.Module):
    def __init__(self, speaker, listener, decoder):
        super(Team, self).__init__()
        self.speaker = speaker
        self.listener = listener
        self.decoder = decoder

    def forward(self, speaker_x, listener_x):
        comm, speaker_loss, info = self.speaker(speaker_x)
        recons = self.decoder(comm)
        prediction = self.listener(recons, listener_x)
        return prediction, speaker_loss, info, recons

    # Helper method for going from communication to a listener's prediction
    def pred_from_comms(self, comm, listener_x):
        recons = self.decoder(comm)
        prediction = self.listener(recons, listener_x)
        return prediction


class TeamLexSem(nn.Module):
    def __init__(self, speaker, decoder):
        super(TeamLexSem, self).__init__()
        self.speaker = speaker
        self.decoder = decoder

    def forward(self, speaker_x):
        comm, speaker_loss, info = self.speaker(speaker_x)
        recons = self.decoder(comm)
        return speaker_loss, info, recons

