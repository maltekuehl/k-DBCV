import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
import numpy.typing as npt
from typing import List, Tuple, Dict


def format_data(
    X: npt.NDArray[np.float_],
    labels: npt.NDArray[np.int_]
) -> Tuple[
    npt.NDArray[np.float_],
    List[npt.NDArray[np.float_]],
    npt.NDArray[np.int_],
    int,
    int,
    int
]:

    n_samp = X.shape[0]

    # Initial check if all data is noise
    if np.sum(labels) == -n_samp:
        return 'noise', 0, 0, 0, 0, 0
       
    d = X.shape[1] 

    # Stack labels with X and sort
    relist = [i for i in X.T]
    relist.append(labels)
    Xl = np.vstack((relist)).T
    Xl_sort = Xl[Xl[..., -1].argsort()]

    # Find where clusters are seperated
    cluster_ID_split = np.where(np.diff(Xl_sort[..., -1]))[0]

    # Checks for clusters that are single or only two points
    # Reassigns them to noise then resorts the data if necessary
    diff_arr = np.append(
        np.diff(cluster_ID_split), (n_samp - 1) - cluster_ID_split[-1]
    )
    idx1 = np.where(diff_arr == 1)[0]
    idx2 = np.where(diff_arr == 2)[0]
    renoise = len(idx1) + (2 * len(idx2))
    if renoise > 0:
        for i in range(len(idx1)):
            Xl_sort[cluster_ID_split[idx1[i]] + 1] = np.append(
                Xl_sort[cluster_ID_split[idx1[i]] + 1][..., :-1], -1
            )
        for i in range(len(idx2)):
            Xl_sort[cluster_ID_split[idx2[i]] + 1] = np.append(
                Xl_sort[cluster_ID_split[idx2[i]] + 1][..., :-1], -1
            )
            Xl_sort[cluster_ID_split[idx2[i]] + 2] = np.append(
                Xl_sort[cluster_ID_split[idx2[i]] + 2][..., :-1], -1
            )
        Xl_sort = Xl_sort[Xl_sort[..., -1].argsort()]   
        cluster_ID_split = np.where(np.diff(Xl_sort[..., -1]))[0]

    # Checks if all data is now noise
    if np.sum(Xl_sort[..., -1]) == -n_samp:
        return 'noise', 0, 0, 0, 0, 0


    if Xl_sort[..., -1][0] == -1:
        cluster_sort = Xl_sort[cluster_ID_split[0] + 1:, :]
        cluster_groups = np.split(
            cluster_sort, (cluster_ID_split - (cluster_ID_split[0]))[1:]
        )
        cluster_ind = np.concatenate(
            [cluster_ID_split - (cluster_ID_split[0]), [len(cluster_sort) - 1]]
        )
    else:
        cluster_sort = Xl_sort
        cluster_groups = np.split(cluster_sort, cluster_ID_split + 1)
        cluster_ind = np.concatenate(
            [[0], cluster_ID_split + 1, [len(cluster_sort) - 1]]
        )

    N_clust = len(cluster_groups)
    if N_clust < 2:
        return 'not enough clusters', 0, 0, 0, 0, 0

    return cluster_sort, cluster_groups, cluster_ind, n_samp, d, N_clust


# populate the dictionaries that store key values -> sparseness,mst etc.
def intracluster_analysis(
    N_clust: int,
    cluster_groups: List[npt.NDArray[np.float_]],
    d: int,
) -> Tuple[
    Dict[int, float],
    npt.NDArray[np.float_],
    Dict[int, npt.NDArray[np.float_]],
    Dict[int, npt.NDArray[np.int_]]
]:

    core_dists_arr = []
    core_dists_dict = {}
    core_pts = {}
    sparseness = {}
    for i in range(N_clust):
        cluster = cluster_groups[i][:, :-1]

        intraclustmatrix_condensed = pdist(cluster, metric='euclidean') 
        all_pts_core_dists = all_points_core_distance(intraclustmatrix_condensed, d)
        all_core_dists_matrix = np.tile(
            all_pts_core_dists, (all_pts_core_dists.shape[0], 1)
        )
        max_core_dist_matrix  = np.maximum(
            all_core_dists_matrix, all_core_dists_matrix.T
        )

        intraclustmatrix = squareform(intraclustmatrix_condensed)
        intraclust_MRD_matrix = np.maximum(max_core_dist_matrix, intraclustmatrix) 
        sparseness[i], core_pts[i] = MST_builder(intraclust_MRD_matrix)
        core_dists_dict[i] = all_pts_core_dists[core_pts[i]]
        core_dists_arr.append(core_dists_dict[i])

    return sparseness, np.hstack(core_dists_arr), core_dists_dict, core_pts


def all_points_core_distance(
    distance_matrix_condensed: npt.NDArray[np.float_],
    d: int
) -> npt.NDArray[np.float_]:

    distance_matrix_condensed[distance_matrix_condensed == 0] = np.inf
    distance_matrix_condensed = (1 / distance_matrix_condensed)**d
    distance_matrix = squareform(distance_matrix_condensed)
    all_pts_core_dists = (
        distance_matrix.sum(axis=1) / (distance_matrix.shape[0] - 1)
    )
    
    if np.sum(all_pts_core_dists) > 0:
        return all_pts_core_dists**(-1 / d)
    else:
        return all_pts_core_dists
  

def MST_builder(
    MRD_matrix: npt.NDArray[np.float_]
) -> Tuple[float, npt.NDArray[np.int_]]:

    MST_arr = minimum_spanning_tree(MRD_matrix).toarray()
    if np.sum(MST_arr) == 0:
        sparseness = 0
        core_pts = np.array([0], dtype='int64')
    else:
        check_MST = np.hstack(np.where(MST_arr > 0))
        unique_vals, index, count = np.unique(
            check_MST, return_counts=True, return_index=True
        )
        core_pts = unique_vals[count > 1]
        sparseness = MST_arr[core_pts][:, core_pts].max()
        if sparseness == 0:
            sparseness = MST_arr.max()

    return sparseness, core_pts
    

def core_points_analysis(
    cluster_sort: npt.NDArray[np.float_],
    cluster_ind: npt.NDArray[np.int_],
    core_pts: npt.NDArray[np.int_]  
) -> Tuple[
    npt.NDArray[np.float_],
    List[npt.NDArray[np.float_]],
    npt.NDArray[np.int_]
]:

    core_pts_arr = np.array(list(core_pts.values()), dtype=object)

    if len(core_pts_arr.shape) > 1:
        core_clust_ind = np.hstack(core_pts_arr + (cluster_ind[:-1])[..., None])
    else:
        core_clust_ind = np.hstack(core_pts_arr + (cluster_ind[:-1]))

    cols = np.arange(cluster_sort.shape[1]) # np.arange() is faster than range() here
    rows = core_clust_ind.astype(int)

    core_X = cluster_sort.ravel()[
        (cols + (rows * cluster_sort.shape[1]).reshape((-1, 1))).ravel()
    ].reshape(rows.size, cols.size)
    
    cluster_ID_split = np.where(np.diff(core_X[..., -1]))[0] + 1 
    core_cluster_groups = np.split(core_X[..., :-1], cluster_ID_split)
    core_X_ind = np.concatenate([[0], cluster_ID_split, [len(core_X)]]) 
    
    return core_X, core_cluster_groups, core_X_ind


def intercluster_analysis(
    core_X: npt.NDArray[np.float_],
    core_cluster_groups: List[npt.NDArray[np.float_]],
    core_X_ind: npt.NDArray[np.int_],
    core_dists_arr: npt.NDArray[np.float_],
    core_dists_dict: Dict[int, npt.NDArray[np.float_]]
) -> Dict[int, float]:

    separation = {}

    Tree = cKDTree(core_X[:, :-1])

    for i in range(len(core_cluster_groups)):
        cluster = core_cluster_groups[i]
        cluster_size = len(cluster)
        
        NN_array = Tree.query(cluster, k=cluster_size + 1)
        NN_array_min = []
        for j in range(cluster_size):
            NN_array_j = np.vstack((NN_array[0][j], NN_array[1][j])).T 
            NN_array_j = NN_array_j[
                np.where(
                    np.logical_or(
                        NN_array_j[:, 1].astype(int) < core_X_ind[i],
                        NN_array_j[:, 1].astype(int) >= core_X_ind[i + 1]
                    )
                )
            ]
            if len(NN_array_j) > 1:
                NN_array_j = NN_array_j[NN_array_j[:, 0] == np.min(NN_array_j[:, 0])]

            NN_array_min.append(np.hstack((NN_array_j[0], j + core_X_ind[i])))  

        NN_array_min = np.vstack(NN_array_min)
        
        min_Edists = NN_array_min[:, 0]
        outer_core_pts =  NN_array_min[:, 1].astype(int)
        outer_core_dists = core_dists_arr[outer_core_pts]
        inner_core_dists = core_dists_dict[i]
        
        MRD_arr = np.vstack((min_Edists, inner_core_dists, outer_core_dists)).T
        MRD_init = MRD_arr.max(axis=1)
        init_min_MRD = np.min(MRD_init)
        MRD_det = MRD_arr.argmax(axis=1)
        
        check_radially = np.where(
            np.logical_and(
                MRD_det==2,
                MRD_arr[:, 0] < init_min_MRD
            )
        )[0]

        if len(check_radially) == 0:
            separation[i] = init_min_MRD
        else:
            radial_check = Tree.query_ball_point(cluster[check_radially], init_min_MRD)
            for j in range(len(radial_check)):
                pts = np.array(radial_check[j])           
                outer_pts = pts[
                    np.where(
                        np.logical_or(
                            pts < core_X_ind[i], 
                            pts >= core_X_ind[i + 1]
                        )
                    )
                ]
                min_outer_core_dist = min(core_dists_arr[outer_core_pts])
                MRD_arr[check_radially[j]][2] = min_outer_core_dist

            MRD_fin = MRD_arr.max(axis=1)
            separation[i] = np.min(MRD_fin)


    return separation


def weighted_score(
    sparseness: Dict[int, float],
    separation: Dict[int, float],
    N_clust: int,
    cluster_groups: List[npt.NDArray[np.float_]],
    n_samp: int, 
    ind_clust_scores: bool = False
) -> float:

    # Add up all the weighted DBCV scores to get the total  
    
    cluster_score_set = []
    DBCV_val = 0
    for i in range(N_clust):
        cluster_validity = (
            (separation[i] - sparseness[i]) /
            max(separation[i], sparseness[i])
        )
        
        cluster_score_set.append(cluster_validity)
        cluster_size = len(cluster_groups[i])
        DBCV_val += (cluster_size / n_samp) * cluster_validity

    if ind_clust_scores == True:
        return DBCV_val, cluster_score_set  
    else:
        return DBCV_val  


def predict_memory_allocation(
    labels: npt.NDArray[np.int_]
) -> float:

    _, l_counts = np.unique(labels[labels >= 0], return_counts=True)
    max_cluster_size = l_counts.max() if l_counts.size > 0 else 0
    predicted_memory = (((max_cluster_size**2) * 8) / 1024**3) * 8

    return predicted_memory

            
# main function
def DBCV_score(
    X: npt.NDArray[np.float_],
    labels: npt.NDArray[np.int_],
    ind_clust_scores: bool = False, 
    mem_cutoff: float = 25.0
) -> float:
    
    # Initially formats data for later calculations
    cluster_sort, cluster_groups, cluster_ind, n_samp, d, N_clust = format_data(
        X, labels
    )

    # Handles situations where scoring can not be performed
    if type(cluster_sort) == str:
        if cluster_sort == 'noise':
            print('All points assigned to noise')
        elif cluster_sort == 'not enough clusters':
            print('Not enough clusters: must have at least two. ')
        return -1, -1

    #Handles memory cutoff
    pred_mem_alloc = predict_memory_allocation(labels=cluster_sort[...,-1])
    if pred_mem_alloc > mem_cutoff:
        print('memory cutoff reached')
        return -1,-1

    # Sparseness calculations for DBCV
    sparseness, core_dists_arr, core_dists_dict, core_pts = intracluster_analysis(
        N_clust, cluster_groups, d, 
    )
    

    # Formats core points for intercluster analysis        
    core_X, core_cluster_groups, core_X_ind = core_points_analysis(
        cluster_sort, cluster_ind, core_pts
    )

    # Separation calculations for DBCV
    separation = intercluster_analysis(
        core_X, core_cluster_groups, core_X_ind, core_dists_arr, core_dists_dict
    )

    # DBCV score
    DBCV_val = weighted_score(
        sparseness, separation, N_clust, cluster_groups, n_samp, ind_clust_scores
    )

    return DBCV_val
