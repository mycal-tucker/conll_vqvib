import pickle


class PerformanceMetrics:
    def __init__(self):
        self.epoch_idxs = []
        self.complexities = []
        self.recons = []
        self.comm_accs = []
        self.weights = []
#        self.embed_r2 = []
#        self.tok_r2 = []

#    def add_data(self, epoch_idx, complexity, recons_loss, comm_acc, kl_weight, tok_r2, embed_r2):
    def add_data(self, epoch_idx, complexity, recons_loss, comm_acc, kl_weight):
        self.epoch_idxs.append(epoch_idx)
        self.complexities.append(complexity)
        self.recons.append(recons_loss)
        self.comm_accs.append(comm_acc)
        self.weights.append(kl_weight)
#        self.tok_r2.append(tok_r2)
#        self.embed_r2.append(embed_r2)

    def to_file(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def from_file(path):
        with open(path, 'rb') as file:
            loaded = pickle.load(file)
        return loaded
