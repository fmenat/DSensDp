import copy, gc, sys, os , pickle
import numpy as np
import pandas as pd
import xarray as xray
from pathlib import Path
from typing import List, Union, Dict
from tqdm import tqdm

from torch.utils.data import Dataset

class Dataset_MultiView(Dataset):
    """
    Example: one item, one data example, could contain several views. However, full-view scenario is considered.
    n-views: number of views
    n-examples: number of examples
    
    Attributes
    ----------
        views_data : dictionary 
            with the data {key:name}: {view name:array of data in that view}
        views_data_ident2indx : dictionary of dictionaries
            with the data {view name: dictionary {index:identifier} }
        inverted_ident : dictionary 
            with {key:indx}: {identifier:list of views name (index) that contain that identifier}
        view_names : list of string
            a list with the view names
        views_cardinality : dictionary 
            with {key:name}: {view name: n-examples that contain that view}
        identifiers_target: dictionary
            with the target corresponding ot each index example
        target_names : list of strings
            list with string of target names, indicated in the order
    """ 
    def __init__(self, views_to_add: Union[list, dict] = [], identifiers:List[int] = [], view_names: List[str] =[] , target: List[int] =[], full_view_flag=False):

        """initialization of attributes. You also could given the views to add in the init function to create the structure already, without using add_view method

        Parameters
        ----------
            views_to_add : list or dict of numpy array, torch,tensor or any
                the views to add
            identifiers : list of ints 
                each int will be the identifiers that correspond to each example in the views_to_add
            view_names : list of string 
                the name of the views being added
            target: list of int
                the target values if available (e.g. supervised data)
        """
        self.views_data = {}
        self.views_data_ident2indx = {} 
        self.view_names = []
        self.identifiers_target = {}
        self.target_names = []
        self.train_identifiers = []
        self.val_identifiers = []
        self.train_set = True

        self.stats_xarray = None
        self.set_additional_info()
        
        if len(views_to_add) != 0:
            if len(identifiers) == 0:
                identifiers = np.arange(len(views_to_add[0]))
            if len(view_names) == 0 and type(views_to_add) != dict:
                view_names = ["S"+str(v) for v in np.arange(len(views_to_add))]
            elif len(view_names) == 0 and type(views_to_add) == dict:
                view_names = list(views_to_add.keys())

            for v in range(len(views_to_add)):
                if type(views_to_add) == list or type(views_to_add) == np.ndarray:
                    self.add_view(views_to_add[v], identifiers, view_names[v])
                if type(views_to_add) == dict:
                    self.add_view(views_to_add[view_names[v]], identifiers, view_names[v])
            if len(target) != 0:
                self.add_target(target, identifiers)
            else:
                self.train_identifiers = identifiers
        
    def add_target(self, target_to_add: Union[list,np.ndarray], identifiers: List[int], target_names: List[str] = []):
        """add a target for the corresponding identifiers indicated, it also works by updating target

        Parameters
        ----------
            target_to_add : list, np.array or any structure that could be indexed as ARRAY[i]
                the target values to add 
            identifiers : list of ints 
                each int will be the identifiers that correspond to each example in the target_to_add
            target_names : list of str 
                the target names if available (e.g. categorical targets)
        """
        for i, ident in enumerate(identifiers): 
            self.identifiers_target[ident] = target_to_add[i]
        self.target_names = [f"T{i}" for i in range(len(self.identifiers_target[identifiers[-1]]))] if len(target_names) == 0 else target_names
        self.train_identifiers = list(self.identifiers_target.keys())

    def add_view(self, view_to_add, identifiers: List[int], name: str):
        """add a view array based on identifiers of list and name of the view. The identifiers is used to match the view with others views.

        Parameters
        ----------
            view_to_add : ideally xarray.DataArray, but numpy array is also possible 
                the array of the view to add (no restriction in dimensions or shape)
            identifiers : list of ints 
                each int will be the identifiers that correspond to each example in the view_to_add
            name : string 
                the name of the view being added
        """
        if name in self.view_names:
            print("This view is already saved, it will be updated")
        self.view_names.append(name)
        self.views_data[name] = view_to_add
        self.views_data_ident2indx[name] = {}
        for indx, ident in enumerate(identifiers):
            self.views_data_ident2indx[name][ident] = indx

    def set_val_mask(self, identifiers: List[int]):
        """set a binary mask to indicate the test examples

        Parameters
        ----------
            identifiers : list of identifiers that correspond to the test examples

        """
        self.train_identifiers = list( set(self.identifiers_target.keys()) - set(identifiers) )
        self.val_identifiers = identifiers

    def set_data_mode(self, train: bool):
        self.train_set = train

    def set_used_view_names(self, view_names: List[str]):
        self.used_view_names = list(view_names)
        self.extended_used_view_names = list(self.used_view_names)
        for v in view_names:
            if "_" in v:
                self.extended_used_view_names.extend(v.split("_"))

    def set_additional_info(self,form="zscore",fillnan=False,fillnan_value=0.0,flatten=False,view_names=[]):
        self.form = form
        self.fillnan = fillnan
        self.fillnan_value = fillnan_value
        self.flatten = flatten
        if len(view_names) != 0:
            self.set_used_view_names(view_names)
        else:
            self.used_view_names = view_names
            self.extended_used_view_names = view_names

    def __len__(self) -> int:
        return len(self.get_all_identifiers())

    def get_all_identifiers(self) -> list:
        """get the identifiers of all views on the corresponding set
     
        Returns
        -------
            list of identifiers
            
        """
        if len(self.val_identifiers) != 0:
            identifiers = self.train_identifiers if self.train_set else self.val_identifiers
        else:
            identifiers = self.train_identifiers
        return identifiers
    
    def get_all_labels(self):
        identifier = self.get_all_identifiers()
        labels = []
        for ident in identifier:
            labels.append(self.identifiers_target[ident])
        return np.array(labels)
    
    def get_view_data(self, name: str):
        identifier = self.get_all_identifiers()

        data_views = []
        for ident in identifier:
            data_views.append(self.views_data[name][self.views_data_ident2indx[name][ident]])
        return {"views": np.stack(data_views, axis=0), "identifiers":identifier  , "view_names": [name]}
    
    def normalize_w_stats(self, data, view_name):
        mean_ = self.stats_xarray[f"{view_name}-mean"].values
        std_ = self.stats_xarray[f"{view_name}-std"].values
        max_ = self.stats_xarray[f"{view_name}-max"].values
        min_ = self.stats_xarray[f"{view_name}-min"].values

        if self.form == "zscore":
            return (data - mean_)/std_
        elif self.form == 'max': 
            return data/max_
        elif self.form == "minmax-01":
            return (data - min_)/(max_ - min_)


    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, List[np.ndarray], List[str]]]:
        """
        Parameters
        ----------
            index : int value that correspond to the example to get (with all the views available)

        Returns
        -------
            dictionary with three values
                data : numpy array of the example indicated on 'index' arg
                views : a list of strings with the views available for that example
                train? : a mask indicated if the example is used for train or not    
        """
        identifier = self.get_all_identifiers()[index]

        target_data = self.identifiers_target[identifier] 
        if type(target_data) == np.ndarray:
            target_data = target_data
        elif type(target_data) == xray.DataArray:
            target_data = target_data.values 
        
        view_data = {}
        views_to_add = []
        for view in (self.view_names if len(self.extended_used_view_names) == 0 else self.extended_used_view_names):
            if view not in self.view_names:
                if "_" in view:
                    views_to_add.append(view)
                continue

            data_ = self.views_data[view]
            if type(data_) == np.ndarray:
                data_ = data_[self.views_data_ident2indx[view][identifier]]
            if type(data_) == xray.DataArray:
                data_ = data_.isel(identifier=self.views_data_ident2indx[view][identifier]).values
            else:
                raise Exception(f"Data type for view {view} not supported")
            if view == "year":
                view_data["year"] = data_
                continue
            
            if self.stats_xarray is not None:
                data_ = self.normalize_w_stats(data_, view)
            if self.fillnan:
                data_ = np.nan_to_num(data_, nan=self.fillnan_value)
            if self.flatten:
                data_ = data_.reshape(-1)

            view_data[view] = data_.astype("float32")
        
        for view in views_to_add:
            data_list = []
            for v in view.split("_"):
                data_list.append(view_data.pop(v))
            view_data[view] = np.concatenate(data_list, axis=-1)
                
        return {"identifier": identifier,
                "views": view_data, 
                "target": target_data
                }
        
    def get_view_names(self, indexs: List[int] = []) -> List[str]:
        """get the names of the views available in the corresponding indices"""
        return self.view_names if len(indexs) == 0 else np.asarray(self.view_names)[indexs].tolist()

    def get_view_shapes(self, view_name:str ="") -> Dict[str, tuple]:
        return_dic = {name: self.views_data[name].shape[1:] for name in self.view_names}
        if view_name in return_dic:
            return return_dic[view_name]
        elif "_" in view_name:
            return (return_dic[view_name.split("_")[0]][:-1]) +(sum([return_dic[v][-1] for v in view_name.split("_")]),)
    
    def _to_xray(self):
        data_vars = {}
        for view_n in self.get_view_names():
            data_vars[view_n] = xray.DataArray(data=self.views_data[view_n], 
                                  dims=["identifier"] +[f"{view_n}-D{str(i+1)}" for i in range (len(self.views_data[view_n].shape)-1)], 
                                 coords={"identifier":list(self.views_data_ident2indx[view_n].keys()),
                                        }, )
        
        data_vars["train_mask"] = xray.DataArray(data=np.asarray([1]*len(self.train_identifiers) + [0]*len(self.val_identifiers), dtype="bool"),  
                                        dims=["identifier"], 
                                         coords={"identifier": self.train_identifiers + self.val_identifiers })
        if len(self.identifiers_target) != 0:
            data_vars["target"] = xray.DataArray(data = np.stack(list(self.identifiers_target.values()), axis=0),
                dims=["identifier","dim_target"] , coords ={"identifier": list(self.identifiers_target.keys())} ) 

        return xray.Dataset(data_vars =  data_vars,
                        attrs = {
                                "view_names": self.view_names, 
                                 "target_names": self.target_names,
                                },
                        )
        
    def save(self, name_path, xarray = True, ind_views = False):
        """save data in name_path

        Parameters
        ----------
            name_path : path to a file to save the model, without extension, since extension is '.pkl.
            ind_views : if you want to save the individual views as csv files 
        """
        path_ = Path(name_path)
        name_path_, _, file_name_ = name_path.rpartition("/") 
        path_ = Path(name_path_)
        path_.mkdir(parents=True, exist_ok=True)

        if xarray and (not ind_views): 
            xarray_data = self._to_xray()
            path_ = path_ / (file_name_+".nc" if "nc" != file_name_.split(".")[-1] else file_name_)
            xarray_data.to_netcdf(path_, engine="h5netcdf") 
            return xarray_data
        elif (not xarray) and ind_views:  #only work with 2D arrays
            path_ = Path(name_path_ +"/"+ file_name_)
            path_.mkdir(parents=True, exist_ok=True)
            for view_name in self.get_view_names():
                view_data_aux = self.get_view_data(view_name)
                df_tosave = pd.DataFrame(view_data_aux["views"])
                df_tosave.index = view_data_aux["identifiers"]
                df_tosave.to_csv(f"{str(path_)}/{view_name}.csv", index=True)

    def load_stats(self, path:str, file_name:str ):
        self.stats_xarray = xray.open_dataset(f"{path}/stats/stats_{file_name}.nc", engine="h5netcdf").load()