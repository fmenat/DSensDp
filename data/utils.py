import numpy as np
import xarray as xray
import sys
sys.path.insert(1, '../')

from src.datasets.views_structure import Dataset_MultiView

STATIC_FEATS = ["soil", "dem", "lidar", "landcover"]

def codify_labels(array, labels):
    labels_2_idx = {v: i for i, v in enumerate(labels)} 
    return np.vectorize(lambda x: labels_2_idx[x])(array)

def reverse_padding(array, pad_val = np.nan):
    n= len(array)
    mask_nans = np.isnan(array)
    new_array= np.pad(array[~mask_nans], (np.sum(mask_nans),0), mode="constant", constant_values=pad_val)
    return new_array

def read_list_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]
	

def storage_set(data, path_out_dir, mode = "input", name="lfmc", target_names=[]):
	index, view_data, target = data
	print({"Index": len(index), "Target": len(target)}, {v: len(view_data[v]) for v in view_data})

	data_views = Dataset_MultiView()
	for view_name in view_data:
		if mode.lower() == "input" and view_name.lower() in STATIC_FEATS: 
			data = view_data[view_name]
		else:
			if view_name.lower() in STATIC_FEATS:
				data = view_data[view_name][:,0,:]
			else:
				data = view_data[view_name]
			
		data_views.add_view(data, identifiers=index, name=view_name)
	data_views.add_target(target, identifiers=index, target_names=target_names)

	xarray_data = data_views._to_xray()
	if "test" not in name and "val" not in name:
		storage_stats(xarray_data, path_out_dir, name)
	
	print(f"data stored in {path_out_dir}/{name}")
	add_info = "_input" if mode == "input" else ""
	xarray_data.to_netcdf(f"{path_out_dir}/{name}{add_info}.nc", engine="h5netcdf")

def storage_stats(xarray_data, path_out_dir, name):
	stats_data = xray.Dataset()
	for view_n in xarray_data.view_names:
		if "year" == view_n or xarray_data[view_n].dtype == object: #store min-max values per numerical variable
			print("skipping stats calculation for", view_n)
			continue
		axis_to_norm = xarray_data[view_n].dims[:-1]
		stats_data = stats_data.assign({
			f"{view_n}-mean": xarray_data[view_n].mean(axis_to_norm, skipna=True),
			f"{view_n}-std": xarray_data[view_n].std(axis_to_norm, skipna=True) if view_n != "aerial" else 0, #aerial is too big in memory
			f"{view_n}-min": xarray_data[view_n].min(axis_to_norm, skipna=True),
			f"{view_n}-max": xarray_data[view_n].max(axis_to_norm, skipna=True),
		})
	stats_data.to_netcdf(f"{path_out_dir}/stats/stats_{name}.nc", engine="h5netcdf") 
	print("stats created")