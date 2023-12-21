import numpy as np
import pandas as pd
import os
import json
import itertools
import ternary
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

import src.settings as settings


def normalize(values):
    total = sum(values)
    return tuple(round(value / total, 2) for value in values)


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



def simplex(models_path, eval_type, savepath, scale=1):

    metric = 'word count'
    plot_data = []
    for utility_folder in os.listdir(models_path):
        #if "objective" in utility_folder or "old" in utility_folder or utility_folder == "utility2.2" or utility_folder == "utility9":
        if "objective" in utility_folder or "word_counts" in utility_folder or "old" in utility_folder:
            continue
        print(utility_folder)
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
                    with open(json_file, 'r') as f:
                        existing_params = json.load(f) 
                    count += existing_params["utility"+str(utility_value_folder)]["inf_weight"+str(alpha_value_folder)][eval_type]['average word count']
                except:
                    pass
                if count != 0:
                    print("count:", count)
                    break
            
                
            plot_data.append((utility_value, alpha_value, count))

    df = pd.DataFrame(plot_data, columns=["Utility", "Alpha", "Count"])
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
        metric_res.append(row['Count'])

    points_and_colors = sorted(zip(normalized_points, metric_res), key=lambda x: x[1])
    points, colors = zip(*points_and_colors)

    # Create a colormap
    colormap = plt.cm.Blues
    normalize = mcolors.Normalize(vmin=min(metric_res), vmax=max(metric_res))
    print(min(metric_res), max(metric_res))
    scalar_map = cm.ScalarMappable(norm=normalize, cmap=colormap)

    # Map the normalized accuracies to colors
    colors_mapped = [scalar_map.to_rgba(metric_res) for metric_res in colors]

    # create simplex
    plt.figure(figsize=(10, 10))
    figure, tax = ternary.figure(scale=scale)
    tax.boundary(linewidth=1.0)
    tax.gridlines(multiple=0.1, color="blue")
    tax.set_title("Word count " + eval_type, fontsize=18, pad=18)

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
    lab = 'word count'
    cbar.set_label(lab, fontsize=12)

    plt.savefig(savepath + "simplex_word_count_" + eval_type + ".png", bbox_inches='tight')
    tax.show()





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
    basedir = "src/saved_models/"+ str(settings.num_protos) + '/' + random_init_dir + settings.folder_ctx + "kl_weight" f"{settings.kl_weight}" + "/seed" + str(settings.seed) + '/'
    save_path = "Plots/" + str(settings.num_protos) + "/random_init/simplex/" if settings.random_init else "Plots/" + str(settings.num_protos) + "/anneal/simplex/"
    #simplex(basedir, "pragmatics", save_path, scale=1)
    simplex(basedir, "lexsem", save_path, scale=1)




if __name__ == '__main__':
    #settings.alphas =  [0,  0.5, 1.5, 7]  # informativeness
    #settings.utilities = [0,  0.5, 1.5, 7] # utility
    settings.alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 3.7, 5, 6, 7, 8, 10.5, 12.8, 21, 33, 88, 140, 200]
    settings.utilities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.5, 3.7, 5, 6, 7, 8, 10.5, 12.8, 21, 33, 88, 140, 200] 
    settings.complexities = 1.0
    settings.with_ctx_representation = False
    settings.random_init = False
    settings.kl_weight = 1.0
    settings.folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
    settings.seed = 0
    settings.num_protos = 3000
    settings.n_epochs = 3000

    random_init_dir = "random_init/" if settings.random_init else ""

    run()


#    file_path = 'logs/count.out' if settings.random_init else 'logs/count.out'
#    c_values = parse_results(file_path)
#    print(c_values)
#    modes = ['PRAG', 'LEXSEM']

#    fig, axes = plt.subplots(len(settings.alphas), len(settings.utilities), figsize=(20, 10))
#    fig.suptitle('Word count')

#    all_c_values = [value for sublist in c_values.values() for value in sublist]
#    min_c = min(all_c_values)
#    max_c = max(all_c_values)

#    buffer = 0.1 * (max_c - min_c)
#    y_min, y_max = min_c - buffer, max_c + buffer
    
#    bar_colors = ['orange', 'blue']

#    for i, utility in enumerate(settings.utilities):
#        for j, alpha in enumerate(settings.alphas):
#            norm_alpha, norm_ut, norm_compl = normalize_and_adjust([alpha, utility, complexity])
#            ax = axes[i, j]
#            values = c_values[(norm_ut, norm_alpha, norm_compl)]
#            print(values)
#            x = np.arange(len(modes))
#            ax.bar(x, values, color=bar_colors)
#            ax.set_title('Utility:' + str(round(norm_ut, 3)) + ', Informativeness:' + str(round(norm_alpha,3)) + ', Complexity:' + str(round(norm_compl, 3)))
#            ax.set_xticks(x)
#            ax.set_xticklabels(modes)
#            ax.set_ylabel('N words')
#            ax.set_ylim(y_min, y_max)  
#            ax.axhline(0, color='grey', linewidth=0.8)

#    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#    plt.show()
#    savedir = "Plots/3000/random_init/count_vectors.png" if settings.random_init else "Plots/3000/anneal/count_vectors.png"
#    plt.savefig(savedir, bbox_inches='tight')



