import matplotlib.pyplot as plt
from src.utils.performance_metrics import PerformanceMetrics


with_ctx_dir = "src/saved_models/with_ctx/vq/seed0/2999/train_True_2_metrics"
without_ctx_dir = "src/saved_models/without_ctx/vq/seed0/2999/train_True_2_metrics"
save_path = "Plots/"

metrics_with = PerformanceMetrics.from_file(with_ctx_dir)
metrics_without = PerformanceMetrics.from_file(without_ctx_dir)

accs_with, recons_with, epochs_with = metrics_with.comm_accs, metrics_with.recons, metrics_with.epoch_idxs
accs_without, recons_without, epochs_without = metrics_without.comm_accs, metrics_without.recons, metrics_without.epoch_idxs

fig1 = plt.figure()
plt.plot(epochs_with, accs_with, label='with ctx repr')
plt.plot(epochs_without, accs_without, label='without ctx repr')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
#plt.title()
plt.legend()
fig1.savefig(save_path + "training_accuracy.png")

fig2 = plt.figure()
plt.plot(epochs_with, recons_with, label='with ctx repr')
plt.plot(epochs_without, recons_without, label='without ctx repr')
plt.xlabel("Epochs")
plt.ylabel("Recons loss")
#plt.title()
plt.legend()
fig2.savefig(save_path + "training_recons.png")











