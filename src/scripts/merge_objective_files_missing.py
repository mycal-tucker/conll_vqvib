import os
import json
import glob
import src.settings as settings

def merge_json_files(file_pattern):
    merged_data = {}

    # List all files that match the pattern
    files = glob.glob(file_pattern)

    for file_name in files:
        with open(file_name, 'r') as file:
            data = json.load(file)

            # Merge data
            for utility_key, utility_values in data.items():
                
                if utility_key not in merged_data:
                    merged_data[utility_key] = {}
                # Iterate over each dictionary in the list
                for inf_weight, inf_values in utility_values.items():
                    merged_data[utility_key][inf_weight] = inf_values

    return merged_data


file_pattern = 'objective*.json'

def run():
    settings.random_init = True
    random_init_dir = "random_init/" if settings.random_init else ""
    files_dir = "src/saved_models/3000/" + random_init_dir + "without_ctx/missing/seed0"
    savedir = "src/saved_models/3000/" + random_init_dir + "without_ctx/missing/seed0/objective_missing_merged.json"
    savedir_nans = "src/saved_models/3000/" + random_init_dir + "without_ctx/missing/seed0/objective_missing_merged_nans.json"
    file_pattern = os.path.join(files_dir, 'objective*.json')
    
    # merge the multiple files
    merged_data = merge_json_files(file_pattern)
    with open(savedir, 'w') as file:
        json.dump(merged_data, file, indent=4)
    # merge with objective.json
    if not os.path.exists("src/saved_models/3000/" + random_init_dir + "without_ctx/missing/seed0/objective.json"):
        d = {}
        with open("src/saved_models/3000/" + random_init_dir + "without_ctx/missing/seed0/objective.json", 'w') as file:
            json.dump(d, file, indent=4)
    with open("src/saved_models/3000/" + random_init_dir + "without_ctx/missing/seed0/objective.json", 'r') as file:
        data1 = json.load(file)
    with open(savedir, 'r') as file:
        data2 = json.load(file)
    merged_data = {**data1, **data2}

    # save two files: one with NaNs and one with well trained models
    nan_dict = {}
    non_nan_dict = {}

    for utility_key, inf_weights in merged_data.items():
        for weight_key, metrics in inf_weights.items():
            if metrics['objective'] != metrics['objective']:  
                if utility_key not in nan_dict:
                    nan_dict[utility_key] = {}
                nan_dict[utility_key][weight_key] = metrics
            else:
                if utility_key not in non_nan_dict:
                    non_nan_dict[utility_key] = {}
                non_nan_dict[utility_key][weight_key] = metrics

    with open(savedir, 'w') as file:
        json.dump(non_nan_dict, file, indent=4)
    with open(savedir_nans, 'w') as file:
        json.dump(nan_dict, file, indent=4)
run()


