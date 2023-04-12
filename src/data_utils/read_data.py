import csv
from operator import itemgetter
import os
import shutil
import sys
import time
from multiprocessing.pool import ThreadPool
from torchvision import transforms

import numpy as np
import pandas as pd
import requests
from scipy import spatial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from sklearn.decomposition import PCA
from src.data_utils.helper_fns import get_unique_labels


# %% ---- FUNCTION TO LOAD MANYNAMES.TSV

def load_cleaned_results(filename="data/manynames.tsv", sep="\t",
                         index_col=None):
    # read tsv
    resdf = pd.read_csv(filename, sep=sep, index_col=index_col)

    # remove any old index columns
    columns = [col for col in resdf.columns if not col.startswith("Unnamed")]
    resdf = resdf[columns]

    # run eval on nested lists/dictionaries
    evcols = ['vg_same_object', 'vg_inadequacy_type',
              'bbox_xywh', 'clusters', 'responses', 'singletons',
              'same_object', 'adequacy_mean', 'inadequacy_type']
    for icol in evcols:
        if icol in resdf:
            resdf[icol] = resdf[icol].apply(lambda x: eval(x))
            
    resdf['vg_image_id'] = [str(i) for i in resdf['vg_image_id']] # added otherwise it won't work with get_feature_data
    
    return resdf


def download_url(args):
    t0 = time.time()
    url, filename = args[0], args[1]
    try:
        print("Downloading url", url, "to", filename)
        res = requests.get(url, stream=True)
        if res.status_code == 200:
            with open(filename, 'wb') as f:
                shutil.copyfileobj(res.raw, f)
        return(url, time.time() - t0)
    except Exception as e:
        print('Exception in download_url():', e)


def download_parallel(args):
    cpus = 64
    results = ThreadPool(cpus - 1).imap_unordered(download_url, args)
    for result in results:
        print('url:', result[0], 'time (s):', result[1])


def download_img():
    existing_ids = []
    for existing_img in os.listdir(image_directory):
        existing_ids.append(existing_img.split('.')[0])
    inputs = []
    id_to_url = {}
    for url, img_id in zip(manynames[url_fieldname], manynames['vg_image_id']):
        id_to_url[img_id] = url
        if str(img_id) in existing_ids:
            # print("Skipping", img_id)
            continue
        inputs.append((url, image_directory + str(img_id) + suffix))
    print("Total number to do", len(inputs))
    download_parallel(inputs)
    return id_to_url


def intersection_over_union(boxA, boxB):
    # needs boxes of format [x1, x2, y1, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def rank_distractors(target_features, tar_xyxy, target_size, 
                     img_id, image_size, detections, 
                     threshold_iou, threshold_size):
    """
    Ranks distractors based on size and IoU
    Returns: first competitor bbox xyxy, its features,
    and the bboxes of the other candidates
    """
    # store candidate distractors, based on thresholds
    candidates = []
    for d_xyxy in detections:
        d_w = d_xyxy[2]-d_xyxy[0]
        d_h = d_xyxy[3]-d_xyxy[1]
        dist_size = d_w * d_h
        iou = intersection_over_union(tar_xyxy, d_xyxy)
        if iou <= threshold_iou and dist_size/image_size > threshold_size:
            candidates.append(d_xyxy)
    # choose distractor based on similarity
    similarities = []
    if len(candidates) > 0:
        for d_xyxy in candidates:
            distractor_features, _ = obj_features(img_id, d_xyxy)
            similarities.append([1-spatial.distance.cosine(distractor_features, 
                                target_features), d_xyxy, distractor_features])
    competitors = sorted(similarities, key=itemgetter(0), reverse=True)
    return competitors[0][1], competitors[0][2], [j[1] for j in competitors[1:]]


def img_features_from_id(img_id):
    """
    Extracts visual features from images.
    :param img_id: img_id
    :return: image features, size 
    """
    # Get a pretrained model
    feature_extractor = models.resnet18(pretrained=True)
    feature_extractor.eval()
    # feature_extractor = models.resnet50(pretrained=True)
    modules = list(feature_extractor.children())[:-1]
    feature_extractor = nn.Sequential(*modules)
    for p in feature_extractor.parameters():
        p.requires_grad = False
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pil_image = Image.open(image_directory + img_id + suffix)
    #array_version = np.array(pil_image)
    #if array_version.shape[-1] != 3:
    #    print("Skipping")
    #    continue
    input_tensor = preprocess(pil_image)
    img_tensor = input_tensor
    img_tensor = torch.unsqueeze(img_tensor, 0)  # Batch size 1
    all_features = feature_extractor(img_tensor)
    features = all_features[0, :, 0, 0]
    
    return features, pil_image.size  
 
   
def obj_features(img_id, bbox_xyxy):
    """
    Extracts visual features for objects.
    :param img_id: img_id
    :param bbox_xyxy: list, bounding box [x1, x2, y1, y2]
    :return: object features, size 
    """
    # Get a pretrained model
    feature_extractor = models.resnet18(pretrained=True)
    feature_extractor.eval()
    # feature_extractor = models.resnet50(pretrained=True)
    modules = list(feature_extractor.children())[:-1]
    feature_extractor = nn.Sequential(*modules)
    for p in feature_extractor.parameters():
        p.requires_grad = False
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pil_image = Image.open(image_directory + img_id + suffix)
    cropped = pil_image.crop((bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]))
    input_tensor = preprocess(cropped)
    img_tensor = input_tensor
    img_tensor = torch.unsqueeze(img_tensor, 0)  # Batch size 1
    all_features = feature_extractor(img_tensor)
    features = all_features[0, :, 0, 0]
    
    return features, cropped.size 
    
    
def save_input_representations(filename='data/manynames.tsv', filename_detections='data/manynames_detections.tsv'):
    """
    Extracts and saves visual features for targets, distractors, and context. 
    """
    
    manynames = load_cleaned_results(filename) 
    detdf = pd.read_csv('data/manynames_detections.tsv', sep='\t') 
    detdf['detected_xyxy'] = detdf['detected_xyxy'].apply(lambda x: eval(x))
    del detdf['tar_xywh'] 
    del detdf['classes']
    del detdf['scores'] 
    # merge MN with detections	
    merged_df = manynames.merge(detdf, left_on='link_vg', right_on='link_vg')
    for img_id, tar_xywh, detections in zip(merged_df['vg_image_id'], 
                                            merged_df['bbox_xywh'],
                                            merged_df['detected_xyxy']):
        img_name = image_directory + str(img_id) + suffix
        image_features, image_size = img_features_from_id(img_id)
        tar_x1 = tar_xywh[0]
        tar_y1 = tar_xywh[1]
        tar_x2 = tar_xywh[0] + tar_xywh[2]
        tar_y2 = tar_xywh[1] + tar_xywh[3]
        tar_xyxy = [tar_x1, tar_y1, tar_x2, tar_y2]
        target_features, target_size = obj_features(img_id, tar_xyxy)        
        dist_xyxy, distractor_features, ctx_objects = rank_distractors(target_features, 
                                                    tar_xyxy, target_size, 
                                                    img_id, image_size, detections, 
                                                    threshold_iou, threshold_size)   
        # save target features
        with open(t_features_filename, 'a') as f:
            f.write(img_name.split('.')[0] + ', ')
            f.write(', '.join([str(e) for e in target_features.cpu().detach().numpy()]))
            f.write('\n')     
        # save distractor features
        with open(d_features_filename, 'a') as f:
            f.write(img_name.split('.')[0] + ', ')
            f.write(', '.join([str(e) for e in distractor_features.cpu().detach().numpy()]))
            f.write('\n')
        # save context features
        ctx_feature_list = []
        for ctx_obj_xyxy in ctx_objects:
            ctx_obj_features, _ = obj_features(img_id, ctx_obj_xyxy)
            # normalize features before computing average
            norm_feat = F.normalize(ctx_obj_features, p=2, dim=0)
            ctx_feature_list.append(norm_feat)
        # the image context repreentation is the average
        ctx_features = sum(ctx_feature_list) / len(ctx_feature_list)
        with open(ctx_features_filename, 'a') as f:
            f.write(img_name.split('.')[0] + ', ')
            f.write(', '.join([str(e) for e in ctx_features.cpu().detach().numpy()]))
            f.write('\n')

        
def img_features(id_to_url):
    # Get a pretrained model
    feature_extractor = models.resnet18(pretrained=True)
    feature_extractor.eval()
    # feature_extractor = models.resnet50(pretrained=True)
    modules = list(feature_extractor.children())[:-1]
    feature_extractor = nn.Sequential(*modules)
    for p in feature_extractor.parameters():
        p.requires_grad = False
    count = 0
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for img in sorted(os.listdir(image_directory)):
        print("reading", img, "number", count, "of", len(manynames))
        count += 1
        # if count == 1000:
        #     break
        pil_image = Image.open(image_directory + img)
        array_version = np.array(pil_image)
        if array_version.shape[-1] != 3:
            print("Skipping")
            continue
        try:
            input_tensor = preprocess(pil_image)
        except OSError:
            print("Downloading replacement for truncated image")
            img_id = int(img.split('.')[0])
            download_url((id_to_url.get(img_id), image_directory + str(img_id) + suffix))
            pil_image = Image.open(image_directory + img)
            input_tensor = preprocess(pil_image)
        img_tensor = input_tensor
        img_tensor = torch.unsqueeze(img_tensor, 0)  # Batch size 1
        all_features = feature_extractor(img_tensor)
        features = all_features[0, :, 0, 0]
    
    return features


def get_feature_data(filename_features, desired_names=[], excluded_names=[], selected_fraction=None, max_per_class=None):
    # Merge the feature data with the dataset data.
    manynames = load_cleaned_results(filename='data/manynames.tsv')
    data_rows = []
    with open(filename_features, 'r') as f:
        for line in f:
            list_data = eval(line)
            data_rows.append((list_data[0], list_data[1:]))
    feature_df = pd.DataFrame(data_rows[1:], columns=['vg_image_id', 'features']) # adding [1:] to skip first empy line
    merged_df = pd.merge(feature_df, manynames, on=['vg_image_id'])
    if len(desired_names) == 0 and len(excluded_names) == 0 and selected_fraction is None:
        return merged_df
    assert len(desired_names) == 0 or len(excluded_names) == 0, "Can't specify both include and exclude"
    if len(desired_names) > 0:
        merged_df = merged_df[merged_df['topname'].isin(desired_names)]
        if max_per_class is not None:
            all_idxs = []
            for g in desired_names:
                ix = np.where(merged_df['topname'] == g)[0]
                max_len = min(max_per_class, len(ix))
                ix = ix[:max_len]
                all_idxs.extend(ix.tolist())
            merged_df = merged_df.iloc[all_idxs]
    elif len(excluded_names) > 0:  # Exclude names
        print("Original size", len(merged_df))
        merged_df = merged_df[~merged_df['topname'].isin(excluded_names)]
        print("Filtered by topnames size", len(merged_df))
        indices_to_keep = []
        merged_df.reset_index(inplace=True)
        num_discarded = 0
        for i, response in enumerate(merged_df['responses']):
            filter_out = False
            for response_name in response.keys():
                if response_name in excluded_names:
                    filter_out = True
                    break
            if not filter_out:
                indices_to_keep.append(i)
            else:
                num_discarded += 1
        print("Filtered by all responses: discarding", num_discarded, "and keeping", len(indices_to_keep))
        merged_df = merged_df.iloc[indices_to_keep]
    else:
        # Only keep a random subset of the topnames.
        unique_topnames, _ = get_unique_labels(merged_df)
        selected_names = set()
        for name in sorted(unique_topnames):  # Sorting is important for reproducibility.
            if np.random.random() < selected_fraction:
                selected_names.add(name)
        merged_df = merged_df[merged_df['topname'].isin(selected_names)]
    merged_df.reset_index(inplace=True)
    
    # Count the number of unique words used to describe items in the dataset
    unique_responses = set()
    unique_topwords = set()
    for topword, response in zip(merged_df['topname'], merged_df['responses']):
        unique_topwords.add(topword)
        for word in response.keys():
            unique_responses.add(word)
    print("Num unique topwords:\t", len(unique_topwords))
    print("Num unique response words:\t", len(unique_responses))
    print("Overall dataset size:\t", len(merged_df))
    return merged_df


def get_glove_vectors(comm_dim):
    raw_data = pd.read_table('data/glove.6B.100d.txt', sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    if comm_dim > 100:
        return raw_data
    np_data = np.array(raw_data)
    pca = PCA(n_components=comm_dim)
    new_data = pca.fit_transform(np_data)
    new_pd = pd.DataFrame(data=new_data, index=raw_data.index)
    return new_pd


# %% ---- DIRECTLY RUN
if __name__ == "__main__":
    with_bbox = False
    see_distractors = True
    image_directory = 'data/images/' if with_bbox else 'data/images_nobox/'
    url_fieldname = 'link_mn' if with_bbox else 'link_vg'
    suffix = '.png' if with_bbox else '.jpg'
    t_features_filename = 'data/t_features.csv' if with_bbox else 'data/features_nobox.csv'
    d_features_filename = 'data/d_features.csv'
    ctx_features_filename = 'data/ctx_features.csv'
    if len(sys.argv) > 1:
        fn = sys.argv[1]
    else:
        fn = "data/manynames.tsv"
    print("Loading data from", fn)
    manynames = load_cleaned_results(filename=fn)
    print(manynames.head())
    print(len(manynames))
    if see_distractors: # EG setup
        threshold_iou=0.1
        threshold_size=0.01
        url_map = download_img()
        save_input_representations()
    else: # Mycal's setup
        url_map = download_img()
        img_features(url_map)
