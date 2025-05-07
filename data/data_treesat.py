import numpy as np
import argparse
from tqdm import tqdm
import rasterio
import h5py
import os
import json
import numpy as np
import geopandas as gpd

from utils import storage_set, read_list_file

MAX_LEN_S1a = 151
MAX_LEN_S1d = 257
MAX_LEN_S2 = 146
classes = ["Abies",
            "Acer",
            "Alnus",
            "Betula",
            "Cleared",
            "Fagus",
            "Fraxinus",
            "Larix",
            "Picea",
            "Pinus",
            "Populus",
            "Prunus",
            "Pseudotsuga",
            "Quercus",
            "Tilia"]
INPUT_FEATURES = {
    "aerial": ["R", "G", "B", "NIR"], #t 0.2m spatial
    "S1asc": ["VV","VH"],  #10m spatial resolution
    "S1desc": ["VV","VH"], #10m spatial resolution
    "S2": ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"], #10m spatial resolution
    "S2mask": ["snow_prob","cloud_prob"] #10m spatial resolution
    }

def filter_labels_by_threshold(labels_dict, area_threshold = 0.07):
    filtered = {}
    
    for img in labels_dict:
        for lbl, area in labels_dict[img]:
            if area > area_threshold: # if area greater than threshold we keep the label
                if img not in filtered:
                    filtered[img] = []
                filtered[img].append(lbl)
    return filtered

def load_labels(path, data_list):
    with open(path + "TreeSatBA_v9_60m_multi_labels.json") as file:
        jfile = json.load(file)
        subsetted_dict = {file : jfile[file] for file in data_list}
        labels = filter_labels_by_threshold(subsetted_dict, 0.07)
        lines = (labels.keys())

    y = [[0 for i in range(len(classes))] for line in lines]
    for i, line in enumerate(lines):
        for u in labels[line]:
            y[i][classes.index(u)] = 1
    
    return [v.replace(".tif","") for v in lines], np.array(y).astype(np.uint8)

def extract_data(path, name, view):
    if view == "aerial":
        with rasterio.open(f"{path}/aerial/{name}.tif") as f:
            return np.asarray(f.read()).transpose(1,2,0)
    elif view =="sentinel":
        file_name = [v for v in os.listdir(f"{path}/sentinel-ts/") if name in v][0]
        year_ = int(file_name.split("_")[-1][:-3])
        with h5py.File(f"{path}/sentinel-ts/{file_name}", 'r') as h5file:
            S1_asc = h5file['sen-1-asc-data'][:].astype(np.float32).transpose(0,2,3,1)
            S1_desc = h5file['sen-1-des-data'][:].astype(np.float32).transpose(0,2,3,1)
            S2 = h5file['sen-2-data'][:].astype(np.float32).transpose(0,2,3,1) #float to allow nan storage
            S2_mask = h5file['sen-2-masks'][:].astype(np.uint8).transpose(0,2,3,1)
        return S1_asc, S1_desc, S2, S2_mask, year_
    

def create_full_views(path, split, pad_val=np.nan):
    split_list = read_list_file(f"{path}/split/{split}_filenames.lst")

    new_split, label_ohv  = load_labels(f"{path}/labels/", split_list) #list names, one hot label
    print(f"ammount of data in split={split} is {len(new_split)}")

    views_data = {"aerial":[],"S1asc": [], "S1desc": [], 
    				"S2": [], "S2mask": []}
    lens = {"S1asc": [], "S1desc": [], "S2": []}
    year_data = []
    for name in tqdm(new_split):
        views_data["aerial"].append(extract_data(path, name, view="aerial"))

        s1a, s1d, s2, s2m, y= extract_data(path, name, view="sentinel")
        lens["S1asc"].append(len(s1a))
        lens["S1desc"].append(len(s1d))
        lens["S2"].append(len(s2))
        
        views_data["S1asc"].append(np.apply_along_axis(lambda x: 
                                                    np.pad(x, (MAX_LEN_S1a-len(s1a),0), mode="constant", constant_values=pad_val), 0, s1a))
        views_data["S1desc"].append(np.apply_along_axis(lambda x:
                                                    np.pad(x, (MAX_LEN_S1d-len(s1d),0), mode="constant", constant_values=pad_val), 0, s1d))
        views_data["S2"].append(np.apply_along_axis(lambda x:
                                                    np.pad(x, (MAX_LEN_S2-len(s2),0), mode="constant", constant_values=pad_val), 0, s2))
        views_data["S2mask"].append(np.apply_along_axis(lambda x:
                                                    np.pad(x, (MAX_LEN_S2-len(s2),0), mode="constant", constant_values=0), 0, s2m)) #probability 0
        year_data.append(y)

    max_lens = {v: np.max(lens[v]) for v in lens}
    print(f"maximum lengths in split {split}", max_lens)
    
    for col in views_data.keys():
        views_data[col] = np.stack(views_data[col], axis =0)
        
    return new_split, label_ohv, views_data, np.array(year_data)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--data_dir",
        "-d",
        required=True,
        type=str,
        help="path of the data directory",
    )
    arg_parser.add_argument(
        "--out_dir",
        "-o",
        required=True,
        type=str,
        help="path of the output directory to store the data",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        required=True,
        type=str,
        help="type of data split, options are [train, val, test]",
    )
    args = arg_parser.parse_args()

    new_split, label_ohv, views_data, year_data = create_full_views(args.data_dir, args.split)

    gpd_metadata = gpd.read_file(f"{args.data_dir}/geojson/bb_60m.GeoJSON")
    gpd_metadata["centroid"] = gpd_metadata["geometry"].centroid
    gpd_metadata = gpd_metadata.to_crs(epsg=4326)
    gpd_metadata["long"] = gpd_metadata["centroid"].x
    gpd_metadata["lat"] = gpd_metadata["centroid"].y
    
    gpd_metadata_filtered = gpd_metadata[gpd_metadata["IMG_ID"].isin(new_split)]
    views_data["coords"] = gpd_metadata_filtered[["long","lat"]].values.astype(np.float32)
    views_data["year"] = year_data.astype(np.uint16)
    del gpd_metadata, gpd_metadata_filtered

    storage_set([new_split, views_data, label_ohv], args.out_dir, 
        name=f"treesat_{args.split}", mode="",target_names=classes)
