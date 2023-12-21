import os
import pandas as pd
import json
import itertools
import numpy as np
import ternary
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from src.utils.performance_metrics import PerformanceMetrics
import src.settings as settings


def normalize(values):
    total = sum(values)
    return tuple(round(value / total, 2) for value in values)


def select_row(row, metric):
    if pd.isna(row[metric + '_df1']):
        return pd.Series([row['Utility'], row['Alpha'], row['Complexity'], row[metric + '_df2']])
    elif pd.isna(row[metric + '_df2']):
        return pd.Series([row['Utility'], row['Alpha'], row['Complexity'], row[metric + '_df1']])
    else:
        return pd.Series([row['Utility'], row['Alpha'], row['Complexity'],
                          max(row[metric + '_df1'], row[metric + '_df2'])])


def simplex_weights(points, scale=1, filename="simplex_plot.png"):

    plt.figure(figsize=(13, 10))
    figure, tax = ternary.figure(scale=scale)
    tax.boundary(linewidth=1.0)
    tax.gridlines(multiple=0.1, color="blue")
    tax.set_title("Simplex Weights", fontsize=18, pad=18)

    fontsize = 15
    tax.left_axis_label("Utility", fontsize=fontsize, offset=0.16, color='orange')
    tax.right_axis_label("Informativeness", fontsize=fontsize, offset=0.16, color='green')
    tax.bottom_axis_label("Complexity", fontsize=fontsize, offset=0.16, color='blue')
    tax.ticks(axis='lbr', linewidth=1, multiple=0.1, tick_formats="%.1f", fontsize=12,  offset=0.02, axes_colors={'l': 'orange', 'r':'g', 'b': 'b'})
    
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    
    tax.scatter(points, marker='o', color='red', label="Combinations", s=8)

    plt.savefig(filename, bbox_inches='tight')
    tax.show()    
    


def simplex_metric(models_path_anneal, models_path_rand, eval_type, metric, savepath, scale=1):
    
    plot_data = []
    for utility_folder in os.listdir(models_path_anneal):
        if "objective" in utility_folder or "old" in utility_folder:
            continue
        utility_value = float(utility_folder.replace("utility", ""))
        utility_path = os.path.join(models_path_anneal, utility_folder)
        utility_value_folder = int(utility_value) if utility_value.is_integer() else utility_value

        for alpha_folder in os.listdir(utility_path):
            alpha_value = float(alpha_folder.replace("alpha", ""))
            alpha_path = os.path.join(utility_path, alpha_folder)
            alpha_value_folder = int(alpha_value) if alpha_value.is_integer() else alpha_value

            json_file_path = "src/saved_models/" + str(settings.num_protos) + '/'+ settings.folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(settings.seed) + '/'
            json_file = json_file_path+"objective_merged.json"
            with open(json_file, 'r') as f:
                existing_params = json.load(f)
            try:
                convergence_epoch = existing_params["utility"+str(utility_value_folder)]["inf_weight"+str(alpha_value_folder)]['convergence epoch']
                
                if eval_type != "training":
                    metrics = PerformanceMetrics.from_file(os.path.join(alpha_path, str(convergence_epoch) + '/evaluation/'+ eval_type +'/test_True_2_metrics'))
                    if metric == "accuracy":
                        to_append = metrics.comm_accs[-1]
                    elif metric == "informativeness":
                        to_append = metrics.recons[-1]
                    plot_data.append((utility_value, alpha_value, to_append))
                else:
                    metrics = PerformanceMetrics.from_file(os.path.join(alpha_path, str(convergence_epoch) + '/train_True_2_metrics'))
                    if metric == "accuracy":
                        to_append = metrics.comm_accs[-1]
                    elif metric == "informativeness":
                        to_append = metrics.recons[-1]
                    plot_data.append((utility_value, alpha_value, to_append))
            except:
                print(utility_value_folder, alpha_value_folder, "not found")
                pass
    df_anneal = pd.DataFrame(plot_data, columns=["Utility", "Alpha", metric])
    df_anneal['Complexity'] = 1

    
    plot_data = []
    for utility_folder in os.listdir(models_path_rand):
        if "objective" in utility_folder or "old" in utility_folder:
            continue
        utility_value = float(utility_folder.replace("utility", ""))
        utility_path = os.path.join(models_path_rand, utility_folder)
        utility_value_folder = int(utility_value) if utility_value.is_integer() else utility_value

        for alpha_folder in os.listdir(utility_path):
            alpha_value = float(alpha_folder.replace("alpha", ""))
            alpha_path = os.path.join(utility_path, alpha_folder)
            alpha_value_folder = int(alpha_value) if alpha_value.is_integer() else alpha_value

            json_file_path = "src/saved_models/" + str(settings.num_protos) + '/random_init/' +settings.folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(settings.seed) + '/'
            json_file = json_file_path+"objective_merged.json"

            json_file = json_file_path+"objective_merged.json"
            with open(json_file, 'r') as f:
                existing_params = json.load(f)
            try:
                convergence_epoch = existing_params["utility"+str(utility_value_folder)]["inf_weight"+str(alpha_value_folder)]['convergence epoch']

                if eval_type != "training":
                    metrics = PerformanceMetrics.from_file(os.path.join(alpha_path, str(convergence_epoch) + '/evaluation/'+ eval_type +'/test_True_2_metrics'))
                    if metric == "accuracy":
                        to_append = metrics.comm_accs[-1]
                    elif metric == "informativeness":
                        to_append = metrics.recons[-1]
                    plot_data.append((utility_value, alpha_value, to_append))
                else:
                    metrics = PerformanceMetrics.from_file(os.path.join(alpha_path, str(convergence_epoch) + '/train_True_2_metrics'))

                    if metric == "accuracy":
                        to_append = metrics.comm_accs[-1]
                    elif metric == "informativeness":
                        to_append = metrics.recons[-1]
                    plot_data.append((utility_value, alpha_value, to_append))
            except:
                print(utility_value_folder, alpha_value_folder, "not found")
                pass
    
    df_rand = pd.DataFrame(plot_data, columns=["Utility", "Alpha", metric])
    df_rand['Complexity'] = 1


    if metric == 'training':
        col_metric = 'training'
    elif metric == "accuracy":
        col_metric = "accuracy"
    else:
        col_metric = "informativeness"

    merged_df = pd.merge(df_anneal, df_rand, on=['Utility', 'Alpha', 'Complexity'], how='outer', 
                     suffixes=('_df1', '_df2'))
    print(merged_df.columns)

    merged_df = merged_df.apply(lambda row: select_row(row, col_metric), axis=1)
    merged_df.columns = ['Utility', 'Alpha', 'Complexity', metric]


    # normalize weights 
    normalized_points = []
    metric_res = []

    for _, row in merged_df.iterrows():
        total = row['Utility'] + row['Alpha'] + row['Complexity']
        normalized_utility = row['Utility'] / total
        normalized_informativeness = row['Alpha'] / total
        normalized_complexity = row['Complexity'] / total
        normalized_points.append((normalized_complexity, normalized_informativeness, normalized_utility))
        if metric == "accuracy":
            metric_res.append(row['accuracy'])
        else:
            metric_res.append(row['informativeness'])
    

    points_and_colors = sorted(zip(normalized_points, metric_res), key=lambda x: x[1])
    points, colors = zip(*points_and_colors)

    
    # Create a colormap
    colormap = plt.cm.hot_r
    if metric == "informativeness":
        normalize = mcolors.Normalize(vmin=-0.4, vmax=-0.15)
    else:
        normalize = mcolors.Normalize(vmin=0.45, vmax=1)
    scalar_map = cm.ScalarMappable(norm=normalize, cmap=colormap)

    # Map the normalized accuracies to colors
    colors_mapped = [scalar_map.to_rgba(metric_res) for metric_res in colors]
    
    # create simplex
    plt.figure(figsize=(10, 10))
    figure, tax = ternary.figure(scale=scale)
    tax.boundary(linewidth=1.0)
    tax.gridlines(multiple=0.1, color="blue")
    if eval_type != "training":
        if metric == "accuracy":
            tax.set_title("Utility - val set", fontsize=18, pad=18)
        else:
            tax.set_title("Informativeness - val set", fontsize=18, pad=18)
    else:
        if metric == "accuracy":
            tax.set_title("Utility - train set", fontsize=18, pad=18)
        else:
            tax.set_title("Informativeness - train set", fontsize=18, pad=18)
    fontsize = 15
    tax.left_axis_label("Utility", fontsize=fontsize, offset=0.16, color='orange')
    tax.right_axis_label("Informativeness", fontsize=fontsize, offset=0.16, color='green')
    tax.bottom_axis_label("Complexity", fontsize=fontsize, offset=0.16, color='blue')
    tax.ticks(axis='lbr', linewidth=1, multiple=0.1, tick_formats="%.1f", fontsize=12,  offset=0.02, axes_colors={'l': 'orange', 'r':'g', 'b': 'b'})

    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')

    tax.scatter(points, marker='o', color=colors_mapped, label="Combinations", s=12)

    # Create the colorbar
    fig = tax.get_axes().figure
    cbar = fig.colorbar(scalar_map, ax=tax.get_axes(), orientation='vertical', shrink=0.6)
    if metric == "accuracy":
        lab = "Utility"
    else:
        lab = "negative MSE"
    cbar.set_label(lab, fontsize=12)

    plt.savefig(savepath + "simplex_" + metric + "_" + eval_type + ".png", bbox_inches='tight')
    tax.show()



def simplex_objective(models_path, savepath, scale=1):
    
    metric = 'objective'
    plot_data = []
    for utility_folder in os.listdir(models_path):
        #if "objective" in utility_folder or "old" in utility_folder or utility_folder == "utility2.2" or utility_folder == "utility9":
        if "objective" in utility_folder:
            continue
        print(utility_folder)
        utility_value = float(utility_folder.replace("utility", ""))
        utility_path = os.path.join(models_path, utility_folder)
        utility_value_folder = int(utility_value) if utility_value.is_integer() else utility_value

        for alpha_folder in os.listdir(utility_path):
            alpha_value = float(alpha_folder.replace("alpha", ""))
            alpha_path = os.path.join(utility_path, alpha_folder)
            alpha_value_folder = int(alpha_value) if alpha_value.is_integer() else alpha_value

            json_file_path = "src/saved_models/" + str(settings.num_protos) + '/' + random_init_dir + settings.folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(settings.seed) + '/'
            json_file = json_file_path+"objective_merged.json"
            with open(json_file, 'r') as f:
                existing_params = json.load(f)
            
            print("--", alpha_value_folder)
            objective = existing_params["utility"+str(utility_value_folder)]["inf_weight"+str(alpha_value_folder)]['objective']
            
            plot_data.append((utility_value, alpha_value, objective))

    df = pd.DataFrame(plot_data, columns=["Utility", "Alpha", "Objective"])
    df['Complexity'] = 1

    # normalize weights 
    normalized_points = []
    metric_res = []

    for _, row in df.iterrows():
        total = row['Utility'] + row['Alpha'] + row['Complexity']
        normalized_utility = row['Utility'] / total
        normalized_informativeness = row['Alpha'] / total
        normalized_complexity = row['Complexity'] / total
        normalized_points.append((normalized_complexity, normalized_informativeness, normalized_utility)) 
        metric_res.append(row['Objective'])

    points_and_colors = sorted(zip(normalized_points, metric_res), key=lambda x: x[1])
    points, colors = zip(*points_and_colors)

    # Create a colormap
    colormap = plt.cm.hot_r
    #colormap = plt.cm.hot
    normalize = mcolors.Normalize(vmin=min(metric_res), vmax=max(metric_res))
    scalar_map = cm.ScalarMappable(norm=normalize, cmap=colormap)

    # Map the normalized accuracies to colors
    colors_mapped = [scalar_map.to_rgba(metric_res) for metric_res in colors]

    # create simplex
    plt.figure(figsize=(10, 10))
    figure, tax = ternary.figure(scale=scale)
    tax.boundary(linewidth=1.0)
    tax.gridlines(multiple=0.1, color="blue")
    tax.set_title("Objective", fontsize=18, pad=18)

    fontsize = 15
    tax.left_axis_label("Utility", fontsize=fontsize, offset=0.16, color='orange')
    tax.right_axis_label("Informativeness", fontsize=fontsize, offset=0.16, color='green')
    tax.bottom_axis_label("Complexity", fontsize=fontsize, offset=0.16, color='blue')
    tax.ticks(axis='lbr', linewidth=1, multiple=0.1, tick_formats="%.1f", fontsize=12,  offset=0.02, axes_colors={'l': 'orange', 'r':'g', 'b': 'b'})

    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')

    tax.scatter(points, marker='o', color=colors_mapped, label="Combinations", s=15)

    # Create the colorbar
    fig = tax.get_axes().figure
    cbar = fig.colorbar(scalar_map, ax=tax.get_axes(), orientation='vertical', shrink=0.6)
    lab = 'objective'
    cbar.set_label(lab, fontsize=12)

    plt.savefig(savepath + "simplex_objective.png", bbox_inches='tight')
    tax.show()




def run():
    complexity = settings.complexities
    informativeness = settings.alphas
    utility = settings.utilities

    # Weights plot
    # create all the combinations
    combinations = list(itertools.product(complexity, informativeness, utility))
    normalized_combinations = [normalize(comb) for comb in combinations]
    scale=1
    normalized_points = [(c * scale, i * scale, u * scale) for (c, i, u) in normalized_combinations]
    simplex_weights(normalized_points, scale=scale, filename="Plots/" + str(settings.num_protos) + "/" + random_init_dir + "simplex/simplex_weights.png")
    
    # Metrics' plots 
    basedir_anneal = "src/saved_models/"+ str(settings.num_protos) + '/' + settings.folder_ctx + "kl_weight" f"{settings.kl_weight}" + "/seed" + str(settings.seed) + '/'
    basedir_rand = "src/saved_models/"+ str(settings.num_protos) + '/random_init/' + settings.folder_ctx + "kl_weight" f"{settings.kl_weight}" + "/seed" + str(settings.seed) + '/'
    save_path = "Plots/" + str(settings.num_protos) + "/merged/"+ "simplex/"
    simplex_metric(basedir_anneal, basedir_rand, "pragmatics", "accuracy", save_path, scale=1)
    simplex_metric(basedir_anneal, basedir_rand, "pragmatics", "informativeness", save_path, scale=1)
    simplex_metric(basedir_anneal, basedir_rand, "lexsem", "accuracy", save_path, scale=1)
    simplex_metric(basedir_anneal, basedir_rand, "lexsem", "informativeness", save_path, scale=1)
    simplex_metric(basedir_anneal, basedir_rand, "training", "accuracy", save_path, scale=1)
    simplex_metric(basedir_anneal, basedir_rand, "training", "informativeness", save_path, scale=1)
    #simplex_objective(basedir, save_path, scale=1)



if __name__ == '__main__':
    #settings.alphas =  [0, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12.8, 21, 33, 88, 140, 233] 
    #settings.utilities =  [0, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12.8, 21, 33, 88, 140, 233]
    settings.alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 3.7, 5, 6, 7, 8, 10.5, 12.8, 21, 33, 88, 140, 200]
    settings.utilities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 3.7, 5, 6, 7, 8, 10.5, 12.8, 21, 33, 88, 140, 200]
    settings.random_init = True
    settings.complexities = [1] 
    settings.with_ctx_representation = False
    settings.kl_weight = 1.0
    settings.folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
    settings.seed = 0
    settings.num_protos = 3000
    settings.n_epochs = 3000

    random_init_dir = "random_init/" 

    run()

