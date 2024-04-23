#!/usr/bin/env python
import math
import os.path
import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns


####################
# DRAW CLUSTERS    #
####################

# Plot the data points in Euclidean space, color-code by cluster
def DrawClusters(dataframe):
    sns.relplot(data=dataframe, x=dataframe.columns[0], y=dataframe.columns[1], hue='clusters', aspect=1.61,
                palette="tab10")
    plt.show()


####################
# LOAD DATA        #
####################
def LoadData(DB_PATH):
    # Load the input file into a Pandas DataFrame object
    dataframe = pandas.read_csv(DATA_PATH, sep=';', encoding='cp1252')

    # Check how many rows and columns are in the loaded data
    assert dataframe.shape[1] == 22, "Unexpected input data shape."

    # TODO: Perform a PROJECT operation to filter down to the following attributes:
    #       - latitude
    #       - longitude
    dataframe = dataframe[['latitude', 'longitude']]
    assert dataframe.shape[1] == 2, "Unexpected projected data shape."

    return dataframe


####################
# GET NUM CLUSTERS #
####################
def GetNumClusters(dataframe):
    # TODO: Get the number of unique clusters
    num_clusters = dataframe['clusters'].nunique()
    return num_clusters


####################
# GET CLUSTER IDS  #
####################
def GetClusterIds(dataframe):
    # TODO: Get the unique IDs of each cluster
    cluster_ids = dataframe['clusters'].unique()
    return cluster_ids


####################
# GET CLUSTER      #
####################
def GetCluster(dataframe, cluster_id):
    # TODO: Perform a SELECT operation to return only rows in the specified cluster
    cluster = dataframe[['latitude', 'longitude']][dataframe['clusters'] == cluster_id].to_numpy()
    return cluster


####################
# DISTANCE         #
####################
def Distance(lhs, rhs):
    # TODO: Calculate the Euclidean distance between two rows
    dist = np.linalg.norm(lhs - rhs)
    return dist


####################
# SINGLE LINK DIST #
####################
def SingleLinkDistance(lhs, rhs):
    # TODO: Calculate the single-link distance between two clusters
    min1 = 1000000
    for x in lhs:
        for y in rhs:
            dist = Distance(x, y)
            if dist < min1:
                min1 = dist
    dist = min1
    return dist


######################
# COMPLETE LINK DIST #
######################
def CompleteLinkDistance(lhs, rhs):
    # TODO: Calculate the complete-link distance between two clusters
    max1 = 0
    for x in lhs:
        for y in rhs:
            dist = Distance(x, y)
            if dist > max1:
                max1 = dist
    dist = max1
    return dist


#######################
# RECURSIVELY CLUSTER #
#######################
def RecursivelyCluster(dataframe, K, M):
    # TODO: Check if we have reached the desired number of clusters
    global othercluster, onecluster
    if GetNumClusters(dataframe) == K:
        return dataframe

    # TODO: Find the closest 2 clusters
    x = GetClusterIds(dataframe)
    min1 = 100000000
    for y in range(len(x)):
        lhs = GetCluster(dataframe, x[y])
        for z in range(y+1, len(x)):
            rhs = GetCluster(dataframe, x[z])
            # if M == SingleLinkDistance:
            dis = M(lhs, rhs)
            if min1 > dis:
                min1 = dis
                onecluster = x[y]
                othercluster = x[z]



    # TODO: Merge the closest 2 clusters
    dataframe.loc[dataframe['clusters'] == othercluster, 'clusters'] = onecluster
    result = RecursivelyCluster(dataframe, K, M)

    return result


####################
# AGNES            #
####################
def Agnes(db_path, K, M):
    # Load the data in and select the features/attributes to work with (lat, lon)
    dataframe = LoadData(db_path)
    assert dataframe.shape[1] == 2, "Unexpected input data shape (lat, lon)."

    # TODO: Add each datum to its own cluster (as a new column)
    dataframe['clusters'] = dataframe.index.tolist()
    assert dataframe.shape[1] == 3, "Unexpected input data shape (lat, lon, cluster)."

    # Generate clusters from all points and recursively merge
    results = RecursivelyCluster(dataframe, K, M)

    return results


####################
# MAIN             #
####################
if __name__ == "__main__":

    RUN_UNIT_TEST = True
    if RUN_UNIT_TEST:
        # Path where you downloaded the data
        DATA_PATH = './unit_test_data.csv'
        K = 2  # The number of output clusters.
        M = SingleLinkDistance  # The cluster similarity measure M to be used.

        # Run the AGNES algorithm with the unit test data
        results = Agnes(DATA_PATH, K, M)
        assert results.shape == (5, 3), "Unexpected output data shape. {}".format(results.shape)

        # Write results to file
        f = open("agnes_unit_test.txt", "w")
        f.write(results.to_csv(header=False))
        f.close()
        DrawClusters(results)

    # TODO: When you are ready to run with the full dataset, modify the following line to True
    RUN_FULL_SINGLE_LINK = False
    if RUN_FULL_SINGLE_LINK:
        # Path where you downloaded the data
        DATA_PATH = './apartments_for_rent_classified_100.csv'
        K = 6  # The number of output clusters.
        M = SingleLinkDistance  # The cluster similarity measure M to be used.

        # Run the AGNES algorithm using single-link
        results = Agnes(DATA_PATH, K, M)
        assert results.shape == (97, 3), "Unexpected output data shape. {}".format(results.shape)

        # Write results to file
        f = open("agnes_single_link.txt", "w")
        f.write(results.to_csv(header=False))
        f.close()
        DrawClusters(results)

    # TODO: When you are ready to run with the full dataset, modify the following line to True
    RUN_FULL_COMPLETE_LINK = False
    if RUN_FULL_COMPLETE_LINK:
        # Path where you downloaded the data
        DATA_PATH = './apartments_for_rent_classified_100.csv'
        K = 6  # The number of output clusters.
        M = CompleteLinkDistance  # The cluster similarity measure M to be used.

        # Run the AGNES algorithm using complete-link
        results = Agnes(DATA_PATH, K, M)
        assert results.shape == (97, 3), "Unexpected output data shape. {}".format(results.shape)

        # Write results to file
        f = open("agnes_complete_link.txt", "w")
        f.write(results.to_csv(header=False))
        f.close()
        DrawClusters(results)
