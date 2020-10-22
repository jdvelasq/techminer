import pandas as pd
from sklearn.cluster import (
    DBSCAN,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    FeatureAgglomeration,
    KMeans,
    MeanShift,
)

from techminer.core.sort_axis import sort_axis
from techminer.plots import expand_ax_limits
from operator import itemgetter


def repair_labels(labels):

    n_labels = len([w for w in labels if w > 0])

    if n_labels == 0:
        return [0] * len(labels)

    for i_item, item in enumerate(labels):
        if item == -1:
            labels[i_item] = n_labels
            n_labels += 1

    return labels


def clustering(
    X,
    method,
    n_clusters,
    affinity,
    linkage,
    random_state,
    top_n,
    name_prefix="Cluster {}",
    documents=False,
):

    X = X.copy()

    ##
    ## Compute cluster labels
    ##
    labels = None

    if method == "Affinity Propagation":
        try:
            labels = AffinityPropagation(random_state=int(random_state)).fit_predict(X)
            labels = repair_labels(labels)
            n_clusters = len(set([w for w in labels if w >= 0]))

        except:
            n_clusters = 1
            labels = [0] * len(X)
            ## raise Exception("Affinity Propagation did not converge")

    if method == "Agglomerative Clustering":
        labels = AgglomerativeClustering(
            n_clusters=n_clusters, affinity=affinity, linkage=linkage
        ).fit_predict(X)

    if method == "Birch":
        labels = Birch(n_clusters=n_clusters).fit_predict(X)

    if method == "DBSCAN":
        labels = DBSCAN().fit_predict(X)
        labels = repair_labels(labels)
        n_clusters = len(set([w for w in labels if w >= 0]))

    if method == "KMeans":
        labels = KMeans(
            n_clusters=n_clusters, random_state=int(random_state)
        ).fit_predict(X)

    if method == "Mean Shift":
        labels = MeanShift().fit_predict(X)
        labels = repair_labels(labels)
        n_clusters = len(set([w for w in labels if w >= 0]))

    ##
    ## Cluster memberships
    ##
    M = pd.DataFrame({"K": X.index.tolist(), "Cluster": labels})
    M = M.groupby(["Cluster"], as_index=False).agg({"K": list})
    if documents is False:
        M["K"] = M.K.map(
            lambda w: sorted(
                w, key=lambda m: m.split(" ")[-1].split(":")[0], reverse=True
            )
        )
    n = M.K.map(len).max()
    M["K"] = M.K.map(lambda w: w + [pd.NA] * (n - len(w)))
    dict_ = {}
    for cluster in M.Cluster.tolist():
        dict_[cluster] = M[M.Cluster == cluster]["K"].tolist()[0]
    cluster_members = pd.DataFrame(dict_)
    ###
    cluster_members.columns = ["CLUST_{}".format(i) for i in cluster_members.columns]
    ###
    cluster_members = cluster_members.applymap(lambda w: "" if pd.isna(w) else w)

    ##
    ## Cluster centres
    ##
    X["CLUSTER"] = labels
    cluster_centers = X.groupby("CLUSTER").mean()
    X.pop("CLUSTER")

    ##
    ## Cluster names
    ##
    cluster_names = cluster_members.loc[cluster_members.index[0], :].tolist()

    return n_clusters, labels, cluster_members, cluster_centers, cluster_names
