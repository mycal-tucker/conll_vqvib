import numpy as np
import ast
import pandas as pd
import os
import json
import itertools
import ternary
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

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
    return normalized



def generate_csv(models_path, models_path_missing, metric, eval_type, savepath, scale=1):

    plot_data = []
    models = os.listdir(models_path) 
    models2 = os.listdir(models_path_missing)
    
    for utility_folder in models:
        if "objective" in utility_folder or "word_counts" in utility_folder or "done_weights" in utility_folder:
            continue
        utility_value = float(utility_folder.replace("utility", ""))
        utility_path = os.path.join(models_path, utility_folder)
        utility_value_folder = int(utility_value) if utility_value.is_integer() else utility_value
        
        for alpha_folder in os.listdir(utility_path):
            alpha_value = float(alpha_folder.replace("alpha", ""))
            alpha_path = os.path.join(utility_path, alpha_folder)
            alpha_value_folder = int(alpha_value) if alpha_value.is_integer() else alpha_value
            print("--", alpha_value_folder)

            for num in range(20):
                count = 0
                try:
                    json_file_path = "src/saved_models/" + str(settings.num_protos) + '/' + random_init_dir + settings.folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(settings.seed) + '/'
                    json_file = json_file_path+"word_counts" + str(num) + ".json"
                    json_file_missing = json_file_path + "word_counts_missing" + str(num) + ".json"
                    for i in [json_file, json_file_missing]:
                        with open(i, 'r') as f:
                            existing_params = json.load(f) 
                        if metric == "word count":
                            count += existing_params["utility"+str(utility_value_folder)]["inf_weight"+str(alpha_value_folder)][eval_type]['average word count']
                        else:
                            count += existing_params["utility"+str(utility_value_folder)]["inf_weight"+str(alpha_value_folder)][eval_type]['average entropy']
                except:
                    pass
                if count != 0:
                    print("count:", count)
                    break
            
                
            plot_data.append((utility_value, alpha_value, count))

    df = pd.DataFrame(plot_data, columns=["Utility", "Alpha", "Count"])
    print(len(df))

    json_file_path = "src/saved_models/" + str(settings.num_protos) + '/' + random_init_dir + settings.folder_ctx + 'kl_weight1.0/seed' + str(settings.seed) + '/'
    done_triplets_file = json_file_path+"done_weights.json"
    with open(done_triplets_file, 'r') as f:
        done_triplets = json.load(f)
    done_triplets = [ast.literal_eval(i) for i in list(done_triplets.keys())]
    
    complexities = []
    for u,a in zip(df['Utility'], df['Alpha']):
        print(u,a)
        for t in done_triplets:
            if t[2] == u and t[1] == a:
                complexities.append(t[0])
    df['Complexity'] = complexities
    

    plot_data = []
    for utility_folder in models2:
        if "objective" in utility_folder or "word_counts" in utility_folder or "done_weights" in utility_folder:
            continue
        utility_value = float(utility_folder.replace("utility", ""))
        utility_path = os.path.join(models_path_missing, utility_folder)
        utility_value_folder = int(utility_value) if utility_value.is_integer() else utility_value

        for alpha_folder in os.listdir(utility_path):
            alpha_value = float(alpha_folder.replace("alpha", ""))
            alpha_path = os.path.join(utility_path, alpha_folder)
            alpha_value_folder = int(alpha_value) if alpha_value.is_integer() else alpha_value
            print("--", alpha_value_folder)

            for num in range(20):
                count = 0
                try:
                    json_file_path = "src/saved_models/" + str(settings.num_protos) + '/' + random_init_dir + settings.folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(settings.seed) + '/'
                    json_file = json_file_path+"word_counts" + str(num) + ".json"
                    json_file_missing = json_file_path + "word_counts_missing" + str(num) + ".json"
                    for i in [json_file, json_file_missing]:
                        with open(i, 'r') as f:
                            existing_params = json.load(f)
                        if metric == "word count":
                            count += existing_params["utility"+str(utility_value_folder)]["inf_weight"+str(alpha_value_folder)][eval_type]['average word count']
                        else:
                            count += existing_params["utility"+str(utility_value_folder)]["inf_weight"+str(alpha_value_folder)][eval_type]['average entropy']
                except:
                    pass
                if count != 0:
                    print("count:", count)
                    break


            plot_data.append((utility_value, alpha_value, count))

    df2 = pd.DataFrame(plot_data, columns=["Utility", "Alpha", "Count"])
    
    done_triplets2_file = json_file_path+"done_weights2.json"
    with open(done_triplets2_file, 'r') as f:
        done_triplets2 = json.load(f)
    done_triplets2 = [ast.literal_eval(i) for i in list(done_triplets2.keys())]

    complexities = []
    for u,a in zip(df2['Utility'], df2['Alpha']):
        for t in done_triplets2:
            if t[2] == u and t[1] == a:
                complexities.append(t[0])
    df2['Complexity'] = complexities
   
    merged_df = pd.concat([df, df2])
    merged_df.to_csv(savepath + "random_init_" + eval_type + "_" + metric + ".csv")





def parse_results(file_path):
    results = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    current_utility = None
    current_alpha = None
    cuttent_complexity = None
    for line in lines:
        line = line.strip()
        if line.startswith('Utility:'):
            parts = line.split(',')
            current_utility = round(float(parts[0].split(':')[1].strip()), 20)
            current_alpha = round(float(parts[1].split(':')[1].strip()), 20)
            current_complexity = round(float(parts[2].split(':')[1].strip()), 20)
        elif line.startswith('pragmatics:') or line.startswith('lexsem:'):
            count = int(line.split(':')[1].strip())
            key = (current_utility, current_alpha, current_complexity)
            if key not in results:
                results[key] = []
            results[key].append(count)    
    return results




def run():
    complexity = settings.complexities
    informativeness = settings.alphas
    utility = settings.utilities
    m_path = "src/saved_models/"+ str(settings.num_protos) + '/' + random_init_dir + settings.folder_ctx + "kl_weight" f"{settings.kl_weight}" + "/seed" + str(settings.seed) + '/'
    m_path_miss = "src/saved_models/"+ str(settings.num_protos) + '/' + random_init_dir + settings.folder_ctx + "missing/seed" + str(settings.seed) + '/'
    save_path = "Plots/" + str(settings.num_protos) + "/random_init/simplex/" if settings.random_init else "Plots/" + str(settings.num_protos) + "/anneal/simplex/"
    generate_csv(m_path, m_path_miss, "word count", "pragmatics", save_path, scale=1)
    generate_csv(m_path, m_path_miss, "word count", "lexsem", save_path, scale=1)
    generate_csv(m_path, m_path_miss, "entropy", "pragmatics", save_path, scale=1)
    generate_csv(m_path, m_path_miss, "entropy", "lexsem", save_path, scale=1)



if __name__ == '__main__':
    settings.alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 2, 3, 4, 5, 7, 10, 20, 40, 88, 140, 200]
    settings.utilities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 2, 3, 4, 5, 7, 10, 20, 40, 88, 140, 200]
    settings.complexities = 1.0
    settings.with_ctx_representation = False
    settings.random_init = True
    settings.kl_weight = 1.0
    settings.folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
    settings.seed = 0
    settings.num_protos = 3000
    settings.n_epochs = 3000

    random_init_dir = "random_init/" if settings.random_init else ""

    run()


