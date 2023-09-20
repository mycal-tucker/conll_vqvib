import os
from matplotlib import pyplot as plt
import src.settings as settings
from src.utils.performance_metrics import PerformanceMetrics


recons_losses_train = []
recons_losses_prag = []
recons_losses_lex = []
accs_train = []
accs_prag = []
accs_lex = []
comps_train = []
comps_prag = []
comps_lex = []


def run():
    save_path = "Plots/train_dropout/" if settings.dropout else "Plots/train_probab/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with_ctx_type = "with_ctx" if settings.with_ctx_representation else "without_ctx"
    
    for p in settings.prob_notseedist:

        train_type = "train_dropout/p_dropout" if settings.dropout else "train_probab/notsee_prob" 
        tmp_train = []
        tmp_prag = []
        tmp_lex = []

        for s in settings.seeds:
            dir_train = "src/saved_models/" + train_type + str(p)+ "/" + with_ctx_type + "/" + "kl_weight" + f"{settings.kl_weight}" + "/seed" + str(s) + '/1999/evaluation/training/test_True_2_metrics'
            dir_prag = "src/saved_models/" + train_type + str(p) + "/" + with_ctx_type  + "/" + "kl_weight" + f"{settings.kl_weight}" + "/seed" + str(s) + '/1999/evaluation/pragmatics/test_True_2_metrics'
            dir_lex = "src/saved_models/" + "/" + train_type + str(p) + "/" + with_ctx_type + "/" + "kl_weight" f"{settings.kl_weight}" + "/seed" + str(s) + '/1999/evaluation/lexsem/test_True_2_metrics'
            tmp_train.append(PerformanceMetrics.from_file(dir_train)) # performance on val set during training
            tmp_prag.append(PerformanceMetrics.from_file(dir_prag)) # performance on val set during pragmatic test
            tmp_lex.append(PerformanceMetrics.from_file(dir_lex)) # performance on val set during lexical-semantics test
        
        # RECONS

        tmp_recons_losses_train = []
        for i in tmp_train:
            tmp_recons_losses_train.append(i.recons[-1])
        # average across seeds
        recons_losses_train.append(sum(tmp_recons_losses_train) / len(tmp_recons_losses_train))

        tmp_recons_losses_prag = []
        for i in tmp_prag:
            tmp_recons_losses_prag.append(i.recons[-1])
        # average across seeds
        recons_losses_prag.append(sum(tmp_recons_losses_prag) / len(tmp_recons_losses_prag))

        tmp_recons_losses_lex = []
        for i in tmp_lex:
            tmp_recons_losses_lex.append(i.recons[-1])
        # average across seeds
        recons_losses_lex.append(sum(tmp_recons_losses_lex) / len(tmp_recons_losses_lex))


        # ACCURACY

        tmp_accs_train = []
        for i in tmp_train:
            tmp_accs_train.append(i.comm_accs[-1])
        # average across seeds
        accs_train.append(sum(tmp_accs_train) / len(tmp_accs_train))

        tmp_accs_prag = []
        for i in tmp_prag:
            tmp_accs_prag.append(i.comm_accs[-1])
        # average across seeds
        accs_prag.append(sum(tmp_accs_prag) / len(tmp_accs_prag))

        tmp_accs_lex = []
        for i in tmp_lex:
            tmp_accs_lex.append(i.comm_accs[-1])
        # average across seeds
        accs_lex.append(sum(tmp_accs_lex) / len(tmp_accs_lex))

        
        # COMPLEXITY

        tmp_comps_train = []
        for i in tmp_train:
            tmp_comps_train.append(i.complexities[-1])
        # average across seeds
        comps_train.append(sum(tmp_comps_train) / len(tmp_comps_train))

        tmp_comps_prag = []
        for i in tmp_prag:
            tmp_comps_prag.append(i.complexities[-1])
        # average across seeds
        comps_prag.append(sum(tmp_comps_prag) / len(tmp_comps_prag))

        tmp_comps_lex = []
        for i in tmp_lex:
            tmp_comps_lex.append(i.complexities[-1])
        # average across seeds
        comps_lex.append(sum(tmp_comps_lex) / len(tmp_comps_lex))


    fig1 = plt.figure()
    plt.plot(settings.prob_notseedist, recons_losses_lex, label="P mask = 1; lex_sem" if settings.dropout == False else "P dropout = 1; lex_sem")
    plt.plot(settings.prob_notseedist, recons_losses_prag, label="P mask = 0; pragm" if settings.dropout == False else "P dropout = 0; pragm")
    plt.plot(settings.prob_notseedist, recons_losses_train, label="P mask = train"  if settings.dropout == False else "P dropout = train" )
    x_label = "p dropout in training" if settings.dropout else "p mask in training"
    plt.xlabel(x_label)
    plt.ylim(-0.321, -0.255)
    #plt.ylim(-0.355, -0.23)
    plt.ylabel("Reconstruction loss")
    plt.legend()
    plt.title("Performances val set")
    fig1.savefig(save_path + with_ctx_type + "_kl_weight" + f"{settings.kl_weight}" + "_" + "reconstruction_loss.png")


    fig2 = plt.figure()
    plt.plot(settings.prob_notseedist, accs_lex, label="P mask = 1; lex_sem" if settings.dropout == False else "P dropout = 1; lex_sem")
    plt.plot(settings.prob_notseedist, accs_prag, label="P mask = 0; pragm" if settings.dropout == False else "P dropout = 0; pragm")
    plt.plot(settings.prob_notseedist, accs_train, label="P mask = train"  if settings.dropout == False else "P dropout = train" )
    x_label = "p dropout in training" if settings.dropout else "p mask in training"
    plt.xlabel(x_label)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.ylim(0.5, 1)
    plt.title("Performances val set")
    fig2.savefig(save_path + with_ctx_type + "_kl_weight" + f"{settings.kl_weight}" + "_" + "accuracy.png")

    fig3 = plt.figure()
    plt.plot(settings.prob_notseedist, comps_lex, label="P mask = 1; lex_sem" if settings.dropout == False else "P dropout = 1; lex_sem")
    plt.plot(settings.prob_notseedist, comps_prag, label="P mask = 0; pragm" if settings.dropout == False else "P dropout = 0; pragm")
    plt.plot(settings.prob_notseedist, comps_train, label="P mask = train"  if settings.dropout == False else "P dropout = train" )
    x_label = "p dropout in training" if settings.dropout else "p mask in training"
    plt.xlabel(x_label)
    plt.ylabel("Complexity")
    plt.legend()
    plt.title("Performances val set")
    fig3.savefig(save_path + with_ctx_type + "_kl_weight" + f"{settings.kl_weight}" + "_" + "complexity.png")



if __name__ == '__main__':
    settings.see_distractor = False
    settings.see_distractors_pragmatics = True

    settings.with_ctx_representation = False
    settings.prob_notseedist = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    settings.dropout = False
    settings.see_probabilities = True

    settings.eval_someRE = False
    settings.seeds = [0, 1, 2]
    settings.kl_weight = 0.1
    run()

