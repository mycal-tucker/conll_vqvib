import os
import src.settings as settings
from src.utils.performance_metrics import PerformanceMetrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

def heatmap(models_path, savepath, metric, eval_type, zoom=False):

    plot_data = []
    for utility_folder in os.listdir(models_path):
        if utility_folder == 'objective.json':
            continue
        utility_value = float(utility_folder.replace("utility", "")) 
        utility_path = os.path.join(models_path, utility_folder)
        utility_value_folder = int(utility_value) if utility_value.is_integer() else utility_value
    
        for alpha_folder in os.listdir(utility_path):
            alpha_value = float(alpha_folder.replace("alpha", ""))
            alpha_path = os.path.join(utility_path, alpha_folder)
            alpha_value_folder = int(alpha_value) if alpha_value.is_integer() else alpha_value

            json_file_path = "src/saved_models/" + str(settings.num_protos) + '/' + settings.folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(settings.seed) + '/'
            json_file = json_file_path+"objective.json"
            with open(json_file, 'r') as f:
                existing_params = json.load(f)
            convergence_epoch = existing_params["utility"+str(utility_value_folder)]["inf_weight"+str(alpha_value_folder)]['convergence epoch']
           
            metrics = PerformanceMetrics.from_file(os.path.join(alpha_path, str(convergence_epoch) + '/evaluation/'+ eval_type +'/test_True_2_metrics'))
            if metric == "accuracy":
                to_append = metrics.comm_accs[-1]
            elif metric == "informativeness":
                to_append = - (metrics.recons[-1])
            plot_data.append((utility_value, alpha_value, to_append))
            
    df = pd.DataFrame(plot_data, columns=["Utility", "Alpha", metric])
    if zoom:
        df = df.loc[(df['Utility'] <= zoom_par) & (df['Alpha'] <= zoom_par)]

    plt.figure(figsize=(10, 8))
    #cmap = sns.cubehelix_palette(as_cmap=True)
    cmap = "viridis"
    
    if metric == "accuracy":
        sc = plt.scatter(df["Utility"], df["Alpha"], c=df[metric], cmap=cmap, s=50, vmin=0.75, vmax=0.9)
    elif metric == "informativeness":
        sc = plt.scatter(df["Utility"], df["Alpha"], c=df[metric], cmap=cmap, s=50)
   
    plt.title('Heatmap ' + metric + " on val set - " + eval_type, fontsize=25)
    plt.xlabel('utility weight', fontsize=20) 
    plt.ylabel('informativeness weight', fontsize=20)
    #plt.colorbar(sc, label=metric if metric == "accuracy" else "negative MSE")
    plt.colorbar(sc, label=metric if metric == "accuracy" else "recons loss")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if zoom == False:
        plt.savefig(savepath + metric + "_" + eval_type + "C.png")
    else:
        plt.savefig(savepath + metric + "_" + eval_type + "_zoomC.png")


def run():
    settings.with_ctx_representation = False
    settings.kl_weight = 1.0
    settings.folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
    settings.seed = 0
    settings.num_protos = 442
    settings.n_epochs = 3000
    basedir = "src/saved_models/"+ str(settings.num_protos) + '/' + settings.folder_ctx + "kl_weight" f"{settings.kl_weight}" + "/seed" + str(settings.seed) + '/'
    save_path = "Plots/" + str(settings.num_protos) + '/heatmaps/'
    heatmap(basedir, save_path, "accuracy", "lexsem", zoom=False)
    heatmap(basedir, save_path, "informativeness", "lexsem", zoom=False)
    heatmap(basedir, save_path, "accuracy", "pragmatics", zoom=False)
    heatmap(basedir, save_path, "informativeness", "pragmatics", zoom=False)

    heatmap(basedir, save_path, "accuracy", "lexsem", zoom=True)
    heatmap(basedir, save_path, "informativeness", "lexsem", zoom=True)
    heatmap(basedir, save_path, "accuracy", "pragmatics", zoom=True)
    heatmap(basedir, save_path, "informativeness", "pragmatics", zoom=True)

zoom_par = 15
run()







