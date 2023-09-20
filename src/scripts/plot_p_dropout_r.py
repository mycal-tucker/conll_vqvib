from matplotlib import pyplot as plt
from src.utils.performance_metrics import PerformanceMetrics


save_path = "Plots/"
p_dropouts = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]

recons_losses = []
accs = []
for p in p_dropouts:
    dir_ = "src/saved_models/without_ctx/vq/p_mask" + str(p)  + '/seed0/2999/evaluation/lexsem/test_True_2_metrics'
    metrics = PerformanceMetrics.from_file(dir_)
    team = Team(speaker, listener, decoder)
    team.load_state_dict(torch.load(basepath + str(checkpoint) + '/model.pt'))
    team.to(settings.device)

    for j, align_data in enumerate(alignment_datastes):
        for use_comm_idx in [False]:
            use_top = True
            dummy_eng = english_fieldname
            if english_fieldname == 'responses':
                use_top = False
                dummy_eng = 'topname'
            consistency_score = get_relative_embedding(team, align_data, glove_data, train_data, fieldname='responses')


fig1 = plt.figure()
plt.plot(p_dropouts, recons_losses)
plt.xlabel("p masked distractor")
plt.ylabel("Informativeness in LexSem")
fig1.savefig(save_path + "p_mask_recons.png")

