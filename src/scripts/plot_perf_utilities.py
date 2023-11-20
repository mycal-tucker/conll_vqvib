import os
from matplotlib import pyplot as plt
import src.settings as settings
from src.utils.performance_metrics import PerformanceMetrics


True_recons_losses_prag = []
True_recons_losses_lex = []
True_accs_prag = []
True_accs_lex = []
False_recons_losses_prag = []
False_recons_losses_lex = []
False_accs_prag = []
False_accs_lex = []

def run():
    save_path = "Plots/" + str(num_protos) + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # True
    with_ctx_type = "with_ctx"
    for u in settings.utilities:
            
            tmp_prag = []
            tmp_lex = []

            for s in settings.seeds:
                dir_prag = "src/saved_models/" + str(num_protos) + "/utility" + str(u) + "/" + with_ctx_type  + "/" + "kl_weight" + f"{settings.kl_weight}" + "/seed" + str(s) + '/' + str(n_epochs-1) + '/evaluation/pragmatics/test_True_2_metrics'
                dir_lex = "src/saved_models/" + str(num_protos) + "/utility" + str(u) + "/" + with_ctx_type + "/" + "kl_weight" f"{settings.kl_weight}" + "/seed" + str(s) + '/' + str(n_epochs-1) + '/evaluation/lexsem/test_True_2_metrics'
                tmp_prag.append(PerformanceMetrics.from_file(dir_prag)) # performance on val set during pragmatic test
                tmp_lex.append(PerformanceMetrics.from_file(dir_lex)) # performance on val set during lexical-semantics test
        
            # RECONS

            # average across seeds
            tmp_recons_losses_prag = []
            for i in tmp_prag:
                tmp_recons_losses_prag.append(i.recons[-1])
            # average across seeds
            True_recons_losses_prag.append(sum(tmp_recons_losses_prag) / len(tmp_recons_losses_prag))

            tmp_recons_losses_lex = []
            for i in tmp_lex:
                tmp_recons_losses_lex.append(i.recons[-1])
            # average across seeds
            True_recons_losses_lex.append(sum(tmp_recons_losses_lex) / len(tmp_recons_losses_lex))


            # ACCURACY

            tmp_accs_prag = []
            for i in tmp_prag:
                tmp_accs_prag.append(i.comm_accs[-1])
            # average across seeds
            True_accs_prag.append(sum(tmp_accs_prag) / len(tmp_accs_prag))

            tmp_accs_lex = []
            for i in tmp_lex:
                tmp_accs_lex.append(i.comm_accs[-1])
            # average across seeds
            True_accs_lex.append(sum(tmp_accs_lex) / len(tmp_accs_lex))

        
    # False
    with_ctx_type = "without_ctx"
    for u in settings.utilities:

            tmp_prag = []
            tmp_lex = []

            for s in settings.seeds:
                dir_prag = "src/saved_models/" + str(num_protos) + "/utility" + str(u) + "/" + with_ctx_type  + "/" + "kl_weight" + f"{settings.kl_weight}" + "/seed" + str(s) + '/' + str(n_epochs-1) + '/evaluation/pragmatics/test_True_2_metrics'
                dir_lex = "src/saved_models/" + str(num_protos) + "/utility" + str(u) + "/" + with_ctx_type + "/" + "kl_weight" f"{settings.kl_weight}" + "/seed" + str(s) + '/' + str(n_epochs-1) + '/evaluation/lexsem/test_True_2_metrics'
                tmp_prag.append(PerformanceMetrics.from_file(dir_prag)) # performance on val set during pragmatic test
                tmp_lex.append(PerformanceMetrics.from_file(dir_lex)) # performance on val set during lexical-semantics test

            # RECONS

            # average across seeds
            tmp_recons_losses_prag = []
            for i in tmp_prag:
                tmp_recons_losses_prag.append(i.recons[-1])
            # average across seeds
            False_recons_losses_prag.append(sum(tmp_recons_losses_prag) / len(tmp_recons_losses_prag))

            tmp_recons_losses_lex = []
            for i in tmp_lex:
                tmp_recons_losses_lex.append(i.recons[-1])
            # average across seeds
            False_recons_losses_lex.append(sum(tmp_recons_losses_lex) / len(tmp_recons_losses_lex))


            # ACCURACY

            tmp_accs_prag = []
            for i in tmp_prag:
                tmp_accs_prag.append(i.comm_accs[-1])
            # average across seeds
            False_accs_prag.append(sum(tmp_accs_prag) / len(tmp_accs_prag))

            tmp_accs_lex = []
            for i in tmp_lex:
                tmp_accs_lex.append(i.comm_accs[-1])
            # average across seeds
            False_accs_lex.append(sum(tmp_accs_lex) / len(tmp_accs_lex))


   

    fig1 = plt.figure()
    plt.plot(settings.utilities, True_recons_losses_lex, c="blue", label="lex sem - w scene", linestyle='-') # True
    plt.plot(settings.utilities, True_recons_losses_prag, c="orange", label="pragm - w scene", linestyle='-') # True
    plt.plot(settings.utilities, False_recons_losses_lex, c="blue", label="lex sem - w/o scene", linestyle='--') # True
    plt.plot(settings.utilities, False_recons_losses_prag, c="orange", label="pragm - w/o scene", linestyle='--') # True
    x_label = "utility weight"
    plt.xlabel(x_label)
    #plt.ylim(-0.321, -0.255)
    #plt.ylim(-0.355, -0.23)
    plt.ylabel("Reconstruction loss")
    plt.legend()
    plt.title("Performances val set")
    fig1.savefig(save_path + "kl_weight" + f"{settings.kl_weight}" + "_" + "reconstruction_loss.png")


    fig2 = plt.figure()
    plt.plot(settings.utilities, True_accs_lex, c="blue", label="lex sem - w scene", linestyle='-')
    plt.plot(settings.utilities, True_accs_prag, c="orange", label="pragm - w scene", linestyle='-') 
    plt.plot(settings.utilities, False_accs_lex, c="blue", label="lex sem - w/o scene", linestyle='--')
    plt.plot(settings.utilities, False_accs_prag, c="orange", label="pragm - w/o scene", linestyle='--')
    x_label = "utility weight"
    plt.xlabel(x_label)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.ylim(0.5, 1)
    plt.title("Performances val set")
    fig2.savefig(save_path + "kl_weight" + f"{settings.kl_weight}" + "_" + "accuracy.png")

    #fig3 = plt.figure()
    #plt.plot(settings.utilities, comps_lex, label="lex sem")
    #plt.plot(settings.utilities, comps_prag, label="pragm")
    #x_label = "utility weight"
    #plt.xlabel(x_label)
    #plt.ylabel("Complexity")
    #plt.legend()
    #plt.title("Performances val set")
    #fig3.savefig(save_path + with_ctx_type + "_kl_weight" + f"{settings.kl_weight}" + "_" + "complexity.png")



if __name__ == '__main__':
    settings.see_distractor = False
    settings.see_distractors_pragmatics = True

    n_epochs = 3000

    num_protos = 884
    settings.utilities = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    settings.dropout = False
    settings.see_probabilities = True

    settings.eval_someRE = False
    settings.seeds = [0, 1] #2
    settings.kl_weight = 1.0
    run()

