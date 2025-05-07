import numpy as np
import itertools

def possible_missing_mask(n_views):
    #all possible of combination/permutation for use of views with "n_views" number of views
    return (2**n_views)-1 #or len(create_list_options(n_views))

def create_list_options(n_views):
    #create a list with all the combination/permutation of the views. To be extracted on each index i
    return list(itertools.product([0, 1], repeat=n_views))[1:][::-1]


def augment_all_missing(view_names_forward):
    #return a list of all the view names for augmentation ["S2", ["S2", "weather"], "weather", ...]
    n_views = len(view_names_forward)
    
    augmented_view_list = []
    n_missing_mask = possible_missing_mask(n_views)
    lst_missing = create_list_options(n_views)
    for i in range(n_missing_mask):
        views_mask_i = np.asarray(lst_missing[i]).astype(bool) #mask as [0,1,1]

        augmented_view_list.append(np.asarray(view_names_forward)[views_mask_i].tolist())    
    return augmented_view_list

def augment_random_missing(view_names_forward, perc= 0):
    #return a single augmentation based on missing
    n_views = len(view_names_forward)
    
    if perc != 0 and perc != 1:
        affected_views = []
        while (len(affected_views) == 0):
            for v in view_names_forward:
                if np.random.rand() > perc:
                    affected_views.append(v)

    else: #1/combinations randomness
        n_missing_mask = possible_missing_mask(n_views)
        lst_missing = create_list_options(n_views)
        
        i_rnd = np.random.randint(0, n_missing_mask)
        views_mask_i = np.asarray(lst_missing[i_rnd]).astype(bool) #mask as [0,1,1]
        affected_views =  np.asarray(view_names_forward)[views_mask_i].tolist()
        
    return affected_views

if __name__ == "__main__":
    view_names = [ "S1", "S2","weather"]
    
    print("View names =",view_names)
    print("Augmented views = ",augment_all_missing(view_names))
    print("Random augmented views = ",augment_random_missing(view_names))

    print("Random with perc 10 of augmented views = ",augment_random_missing(view_names, perc=0.10))
    print("Random with perc 30 of augmented views = ",augment_random_missing(view_names, perc=0.30))
    print("Random with perc 50 of augmented views = ",augment_random_missing(view_names, perc=0.50))
    print("Random with perc 70 of augmented views = ",augment_random_missing(view_names, perc=0.70))
    print("Random with perc 90 of augmented views = ",augment_random_missing(view_names, perc=0.90))
