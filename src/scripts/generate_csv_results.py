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



def normalize_and_adjust(values):
    total = sum(values)
    normalized = [round(value / total, 2) for value in values]
    rounding_error = 1 - sum(normalized)
    # adjust the largest value(s) by the rounding error
    if rounding_error != 0:
        max_value = max(normalized)
        indexes_of_max = [i for i, v in enumerate(normalized) if v == max_value]
        error_per_value = rounding_error / len(indexes_of_max)
        for index in indexes_of_max:
            normalized[index] += error_per_value
    return [round(i,3) for i in normalized]


def select_row(row, metric):
    if pd.isna(row[metric + '_df1']):
        return pd.Series([row['Utility'], row['Alpha'], row['Complexity'], row[metric + '_df2']])
    elif pd.isna(row[metric + '_df2']):
        return pd.Series([row['Utility'], row['Alpha'], row['Complexity'], row[metric + '_df1']])
    else:
        return pd.Series([row['Utility'], row['Alpha'], row['Complexity'],
                          max(row[metric + '_df1'], row[metric + '_df2'])])

    

# TAKES THE BEST RESULT WHEN BOTH ANNEALED AND RANDOM MODEL ARE TRAINED
def generate_csv_metric(models_path_anneal, models_path_rand, models_path_missing, eval_type, metric, savepath):
    
    # ANNEAL
    plot_data = []
    for utility_folder in os.listdir(models_path_anneal):
        if "objective" in utility_folder or "done_weights" in utility_folder or "word" in utility_folder:
            continue
        utility_value = float(utility_folder.replace("utility", ""))
        utility_path = os.path.join(models_path_anneal, utility_folder)
        utility_value_folder = int(utility_value) if utility_value.is_integer() else utility_value

        for alpha_folder in os.listdir(utility_path):
            alpha_value = float(alpha_folder.replace("alpha", ""))
            alpha_path = os.path.join(utility_path, alpha_folder)
            alpha_value_folder = int(alpha_value) if alpha_value.is_integer() else alpha_value

            json_file_path = "src/saved_models/" + str(settings.num_protos) + '/anneal/'+ settings.folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(settings.seed) + '/'
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
                print(utility_value_folder, alpha_value_folder, "not trained pair")
                pass
    df_anneal = pd.DataFrame(plot_data, columns=["Utility", "Alpha", metric])
    df_anneal['Complexity'] = 1

    # RAND
    plot_data = []
    for utility_folder in os.listdir(models_path_rand):
        if "objective" in utility_folder or "done_weights" in utility_folder or "word" in utility_folder:
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
                print(utility_value_folder, alpha_value_folder, "not trained pair")
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
    
    merged_df = merged_df.apply(lambda row: select_row(row, col_metric), axis=1)
    merged_df.columns = ['Utility', 'Alpha', 'Complexity', metric]

    
    # MISSING (models trained in second round, only ranomg init)
    plot_data = []
    for utility_folder in os.listdir(models_path_missing):
        if "objective" in utility_folder or "done_weights" in utility_folder or "word" in utility_folder:
            continue
        utility_value = float(utility_folder.replace("utility", ""))
        utility_path = os.path.join(models_path_missing, utility_folder)
        utility_value_folder = int(utility_value) if utility_value.is_integer() else utility_value

        for alpha_folder in os.listdir(utility_path):
            alpha_value = float(alpha_folder.replace("alpha", ""))
            alpha_path = os.path.join(utility_path, alpha_folder)
            alpha_value_folder = int(alpha_value) if alpha_value.is_integer() else alpha_value
            
            for compl_folder in os.listdir(alpha_path):
                compl_value = float(compl_folder.replace("compl", ""))
                compl_path = os.path.join(utility_path, alpha_folder, compl_folder)
                compl_value_folder = int(compl_value) if compl_value.is_integer() else compl_value
                
                print(utility_value, alpha_value, compl_value)

                
                convergence_epoch = 4999
                if eval_type != "training":
                    metrics = PerformanceMetrics.from_file(os.path.join(compl_path, str(convergence_epoch) + '/evaluation/'+ eval_type +'/test_True_2_metrics'))
                    if metric == "accuracy":
                        to_append = metrics.comm_accs[-1]
                    elif metric == "informativeness":
                        to_append = metrics.recons[-1]
                    plot_data.append((utility_value, alpha_value, compl_value, to_append))
                else:
                    metrics = PerformanceMetrics.from_file(os.path.join(compl_path, str(convergence_epoch) + '/train_True_2_metrics'))

                    if metric == "accuracy":
                        to_append = metrics.comm_accs[-1]
                    elif metric == "informativeness":
                        to_append = metrics.recons[-1]
                    plot_data.append((utility_value, alpha_value, compl_value, to_append)) 


    df_miss = pd.DataFrame(plot_data, columns=["Utility", "Alpha", "Complexity", metric])

    if metric == 'training':
        col_metric = 'training'
    elif metric == "accuracy":
        col_metric = "accuracy"
    else:
        col_metric = "informativeness"

    merged_df2 = pd.merge(df_miss, merged_df, on=['Utility', 'Alpha', 'Complexity', metric], how='outer',
                     suffixes=('_df1', '_df2'))
    
    #merged_df = merged_df2.apply(lambda row: select_row(row, col_metric), axis=1)
    #merged_df2.columns = ['Utility', 'Alpha', 'Complexity', metric]


    merged_df2.to_csv(savepath + "merged_df_" + eval_type + "_" + metric + ".csv")







def run():
    complexity = settings.complexities
    informativeness = settings.alphas
    utility = settings.utilities

    # Weights plot
    # create all the combinations
#    combinations = list(itertools.product(complexity, informativeness, utility))
#    normalized_combinations = [normalize(comb) for comb in combinations]
#    scale=1
#    normalized_points = [(c * scale, i * scale, u * scale) for (c, i, u) in normalized_combinations]
#    simplex_weights(normalized_points, scale=scale, filename="Plots/" + str(settings.num_protos) + "/" + random_init_dir + "simplex/simplex_weights.png")
    
    # Metrics' csvs
    basedir_anneal = "src/saved_models/"+ str(settings.num_protos) + '/anneal/' + settings.folder_ctx + "kl_weight" f"{settings.kl_weight}" + "/seed" + str(settings.seed) + '/'
    basedir_rand = "src/saved_models/"+ str(settings.num_protos) + '/random_init/' + settings.folder_ctx + "kl_weight" f"{settings.kl_weight}" + "/seed" + str(settings.seed) + '/'
    basedir_miss = "src/saved_models/"+ str(settings.num_protos) + '/random_init/' + settings.folder_ctx + "missing/seed" + str(settings.seed) + '/'
    save_path = "Plots/" + str(settings.num_protos) + "/merged/"+ "simplex/"
    generate_csv_metric(basedir_anneal, basedir_rand, basedir_miss, "pragmatics", "accuracy", save_path)
    generate_csv_metric(basedir_anneal, basedir_rand, basedir_miss, "pragmatics", "informativeness", save_path)
    generate_csv_metric(basedir_anneal, basedir_rand, basedir_miss, "lexsem", "accuracy", save_path)
    generate_csv_metric(basedir_anneal, basedir_rand, basedir_miss, "lexsem", "informativeness", save_path)
    generate_csv_metric(basedir_anneal, basedir_rand, basedir_miss, "training", "accuracy", save_path)
    generate_csv_metric(basedir_anneal, basedir_rand, basedir_miss, "training", "informativeness", save_path)
    #simplex_objective(basedir, save_path, scale=1)



if __name__ == '__main__':
    #settings.alphas =  [0, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12.8, 21, 33, 88, 140, 233] 
    #settings.utilities =  [0, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12.8, 21, 33, 88, 140, 233]
    settings.alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 2, 3, 4, 5, 7, 10, 20, 40, 88, 140, 200]
    settings.utilities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 2, 3, 4, 5, 7, 10, 20, 40, 88, 140, 200]
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

