import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker

def heatmap(jsonfile, savepath, metric, zoom=False):
    with open(jsonfile) as f:
        dic = json.load(f)

    plot_data = []
    for utility, inner_dict in dic.items():
        for inf_weight, metrics in inner_dict.items():
            to_plot = metrics[metric]
            plot_data.append((float(utility.replace("utility", "")), float(inf_weight.replace("inf_weight", "")), to_plot))

    df = pd.DataFrame(plot_data, columns=["Utility", "Inf_Weight", metric])
    if zoom:
        df = df.loc[(df['Utility'] <= zoom_par) & (df['Inf_Weight'] <= zoom_par)]
    
    # Plotting
    plt.figure(figsize=(10, 8))
    #cmap = sns.cubehelix_palette(as_cmap=True)
    cmap = "viridis"
    print(min(df['Utility']))
    print(min(df['Inf_Weight']))
    sc = plt.scatter(df["Utility"], df["Inf_Weight"], c=df[metric], cmap=cmap, s=40)
    plt.title('Heatmap training', fontsize=20)
    plt.xlabel('utility weight', fontsize=20)
    plt.ylabel('informativeness weight', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.colorbar(sc, label=metric)
    
    plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.xaxis.set_major_locator(ticker.FixedLocator(ax.get_xticks()))
    ax.yaxis.set_major_locator(ticker.FixedLocator(ax.get_yticks()))
    
    plt.show()
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if zoom:
        plt.savefig(savepath + metric + "_zoom.png")
    else:
        plt.savefig(savepath + metric + ".png")

def run():
    num_protos = 442
    save_path = "Plots/" + str(num_protos) + '/heatmaps/'
    json_file = "src/saved_models/442/without_ctx/kl_weight1.0/seed0/objective.json"
    heatmap(json_file, save_path, "objective")
    heatmap(json_file, save_path, "speaker loss")
    heatmap(json_file, save_path, "recons loss")
    heatmap(json_file, save_path, "utility loss")
    
    heatmap(json_file, save_path, "objective", zoom=True)
    heatmap(json_file, save_path, "speaker loss", zoom=True)
    heatmap(json_file, save_path, "recons loss", zoom=True)
    heatmap(json_file, save_path, "utility loss", zoom=True)

zoom_par = 15
run()

