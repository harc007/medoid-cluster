import pandas as pd
import gc
from itertools import permutations

def get_indexed_df(df):
    indexed_df = df
    row_ids = set(range(1, len(df)+1))
    indexed_df["row_id"] = row_ids
    return indexed_df


def feature_dist_calc(indexed_df, f):
    dist_dict = []
    for i1, r1 in indexed_df.iterrows():
        for i2, r2 in indexed_df.iterrows():
            gc.disable()
            dist_dict.append({'head_row_id':i1, 'tail_row_id':i2, 'dist':f(r1, r2)})
            gc.enable()
    return pd.DataFrame(dist_dict)


def get_initialization_score(feature_distance_df):
    proximity_obj = feature_distance_df[['head_row_id', 'dist']].groupby('head_row_id').sum()
    proximity_obj['head_row_id'] = proximity_obj.index
    joined_data = pd.merge(feature_distance_df, proximity_obj, on = ["head_row_id"], how="inner")
    joined_data['divided_value'] = joined_data['dist_x']/joined_data['dist_y']
    required_final_df = joined_data[['tail_row_id', 'divided_value']].groupby('tail_row_id').sum()
    required_final_df['head_row_id'] = required_final_df.index
    return required_final_df

def sort_initial_medoids(required_final_df, k):
    return list(required_final_df.sort('divided_value')['head_row_id'][:k])

def nearest_seedpoint_finder(feature_distance_df, seed_list):
    filtered_df = feature_distance_df[feature_distance_df['tail_row_id'].isin(seed_list)]
    min_score_df_renamed = filtered_df[['head_row_id', 'dist']].groupby('head_row_id').min()
    min_score_df_renamed['head_row_id'] = min_score_df_renamed.index
    points_with_seed = pd.merge(filtered_df, min_score_df_renamed, how='inner', on = ['head_row_id', 'dist'])
    points_with_seed.columns = ['dist', 'head_row_id', 'cluster_id']
    return points_with_seed

def get_sum_distance(points_with_seed):
    return float(points_with_seed[['dist']].sum())


def get_permutations_for_each_cluster(points_with_seed):
    grp = points_with_seed.groupby('cluster_id')
    permutations_dict = []
    for i in grp:
        for j in list(permutations(i[1]['head_row_id'], 2)) + [(e, e) for e in i[1]['head_row_id']]:
            permutations_dict.append({'cluster_id':i[0], 'head_row_id':j[0], 'tail_row_id':j[1]})

    return pd.DataFrame(permutations_dict)

def recompute_medoids(permutations_df, feature_distance_df):
    joined_data = pd.merge(feature_distance_df, permutations_df, how = 'inner', on = ['head_row_id', 'tail_row_id'])
    distance_per_id = joined_data[['cluster_id', 'head_row_id', 'dist']].groupby(['cluster_id', 'head_row_id']).sum()
    cluster_list = []
    head_id_list = []
    for i in distance_per_id.index:
        gc.disable()
        cluster_list.append(i[0])
        head_id_list.append(i[1])
        gc.enable()
    distance_per_id['cluster_id'] = cluster_list
    distance_per_id['head_row_id'] = head_id_list
    distance_per_id = distance_per_id.reset_index(drop=True)
    min_cluster_dist = distance_per_id[['cluster_id', 'dist']].groupby('cluster_id').min()
    min_cluster_dist['cluster_id'] = min_cluster_id.index
    return pd.merge(min_cluster_dist, distance_per_id, how = "inner", on = ['cluster_id', 'dist'])

def get_single_medoid(recomputed_medoids):
    recomputed_medoids_single = recomputed_medoids.groupby('cluster_id').first()
    recomputed_medoids_single['cluster_id'] = recomputed_medoids_single.index
    return recomputed_medoids_single

def get_new_medoids(recomputed_medoids_single):
    return list(recomputed_medoids_single['cluster_id'])

def get_new_cluster_assignment(feature_distance_df, seed_list, latest_dist=float("inf")):
    while True:
        df_new = nearest_seedpoints_finder(feature_distance_df, seed_list)
        dist_new = get_sum_distance(df_new)
        if(dist_new == latest_dist):
            return df_new
        else:
            permutations_df_new = get_permutations_for_each_cluster(df_new)
            new_medoids_df = recompute_midiods(permutations_df_new, feature_distance_df)
            unique_medoids_df = get_single_medoid(new_medoids_df)
            seed_list = get_new_medoids(unique_medoids_df)
            latest_dist = dist_new

def k_medoids(df, f, k):
    print "INPUT DATAFRAME : "
    indexed_df = get_indexed_df(df)
    feature_dist_calc = feature_dist_calc(df, f)
    seedpoints_score = get_initialization_score(feature_dist_calc)
    data_with_initial_clusters = sort_initial_medoids(seedpoints_score, k)
    final_df = nearest_seedpoint_finder(feature_dist_calc, data_with_initial_clusters)


def main():
    no_cluster = 5
    import random
    import math
    import time
    data_size = 1000
    A = [random.normalvariate(0, 1) for i in range(data_size)]
    B = [random.normalvariate(1, 1) for i in range(data_size)]
    C = [random.normalvariate(-1, 0.5) for i in range(data_size)]
    df = pd.DataFrame({'A':A, 'B':B, 'C':C})
    f = lambda x, y : math.pow((x.A - y.A), 2) + math.pow((x.B - y.B), 2) + math.pow((x.C - y.C), 2)
    start_time = time.time()
    model = k_medoids(df, f, no_cluster)
    end_time = time.time()
    print "TIME TAKEN : " + str(end_time - start_time)
    print "======================================"
