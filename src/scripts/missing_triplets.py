import json
import os
import ast
import src.settings as settings
import random
random.seed(19)


def generate_triplets(resolution=0.1):
    # Generates all possible triplets with the given resolution
    return [(x, y, z) for x in frange(0, 1, resolution) 
                      for y in frange(0, 1, resolution) 
                      for z in frange(0, 1, resolution) 
                      if x + y + z == 1]

def frange(start, stop, step):
    # A range function for floating-point numbers
    while start < stop:
        yield round(start, 10)  # rounding to avoid floating-point arithmetic issues
        start += step


def run():
    folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
    json_file_path = "src/saved_models/" + str(settings.num_protos) + "/random_init/" + folder_ctx + 'kl_weight' + str(settings.kl_weight_unnorm) + '/seed' + str(seed) + '/'
    with open(json_file_path+"done_weights.json") as f:
        done = json.load(f)
    done_triplets = [ast.literal_eval(i) for i in done.values()]
    for num in range(20):
        if os.path.exists(json_file_path+"done_weights" + str(num) + ".json"):
            with open(json_file_path+"done_weights" + str(num) + ".json") as f:
                done = json.load(f)
                done_triplets_num = [ast.literal_eval(i) for i in done.values()]
                done_triplets = done_triplets_num + done_triplets
    print(len(done_triplets))
    all_triplets = generate_triplets(resolution=0.01)
    missing = [i for i in all_triplets if i not in done_triplets]
    sampled = random.sample(missing, 100)
    dic = dict(zip([str(i) for i in sampled], [str(i) for i in sampled]))
    with open(json_file_path+"done_weights" + str(settings.save_num) + ".json", "w") as f2:
        json.dump(dic, f2)

if __name__ == '__main__':
    settings.num_protos = 3000
    settings.kl_weight_unnorm = 1.0
    seed = 0
    settings.with_ctx_representation = False
    settings.save_num = 3
    run()




