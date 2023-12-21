import numpy as np
from matplotlib import pyplot as plt

import src.settings as settings


def normalize(values):
    total = sum(values)
    return tuple(round(value / total, 2) for value in values)


def parse_results(file_path):
    r_values = {}
    current_utility = None
    current_alpha = None

    with open(file_path, 'r') as file:
        for line in file:
            if 'utility:' in line:
                current_utility = float(line.split(':')[1].strip())
            elif 'alpha:' in line:
                current_alpha = float(line.split(':')[1].strip())
                if (current_utility, current_alpha) not in r_values:
                    r_values[(current_utility, current_alpha)] = []
            elif 'r ' in line and current_utility is not None and current_alpha is not None:
                r_value = float(line.split(':')[1].strip())
                r_values[(current_utility, current_alpha)].append(r_value)

    return r_values


if __name__ == '__main__':
    settings.alphas = [0, 1, 7]  # informativeness
    settings.utilities = [0, 1, 7] # utility
    complexity = 1.0
    settings.random_init = True

    file_path = 'logs/someRE_rand.out' if settings.random_init else 'logs/someRE.out'
    r_values = parse_results(file_path)
    modes = ['L on L data', 'L on P data', 'P on L data', 'P on P data']

    fig, axes = plt.subplots(len(settings.alphas), len(settings.utilities), figsize=(20, 10))
    fig.suptitle('Evaluation on someRE - PRODUCTION')

    all_r_values = [value for sublist in r_values.values() for value in sublist]
    min_r = min(all_r_values)
    max_r = max(all_r_values)

    buffer = 0.1 * (max_r - min_r)
    y_min, y_max = min_r - buffer, max_r + buffer
    
    bar_colors = ['green', 'red', 'red', 'green']

    for i, utility in enumerate(settings.utilities):
        for j, alpha in enumerate(settings.alphas):
            norm_alpha, norm_ut, norm_compl = normalize([alpha, utility, complexity])
            ax = axes[i, j]
            values = r_values[(utility, alpha)]
            x = np.arange(len(modes))
            ax.bar(x, values, color=bar_colors)
            ax.set_title(f'Utility: {norm_ut}, Informativeness: {norm_alpha}, Complexity: {norm_compl}')
            ax.set_xticks(x)
            ax.set_xticklabels(modes)
            ax.set_ylabel('r Value')
            ax.set_ylim(y_min, y_max)  
            ax.axhline(0, color='grey', linewidth=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    savedir = "Plots/3000/random_init/someRE_production.png" if settings.random_init else "Plots/3000/anneal/someRE_production.png"
    plt.savefig(savedir, bbox_inches='tight')



