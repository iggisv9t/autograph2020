"""Final solution for autograph https://www.automl.ai/competitions/3
based on template class provided by organizers
"""

import numpy as np
import pandas as pd
import os
from collections import Counter
# It's was one of the easiest way to install packages on this platform
os.system("pip install scipy")
os.system("pip install scikit-learn")

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy import sparse
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


class Model:
    def __init__(self):
        pass

    def prepare_data(self, inp_data):
        # features
        data = inp_data["fea_table"]
        # target labels
        target = inp_data["train_label"]
        # edge list
        edges = inp_data["edge_file"]
        # features without indices
        dataf = data.drop(columns=["node_index"])
        # number of the nodes
        nnodes = data.shape[0]

        # names of the feature columns
        fcols = list(dataf.columns)
        todrop = []
        for col in fcols:
            # check if column has no sense
            if dataf[col].nunique() == 1:
                todrop.append(col)
        if len(todrop) > 0:
            dataf.drop(columns=todrop, inplace=True)

        # the simpliest handcrafted features
        # nodes degrees
        out_degree = Counter(edges["src_idx"].values)
        in_degree = Counter(edges["dst_idx"].values)

        degs = np.empty((data.shape[0], 2))
        for i in range(data.shape[0]):
            degs[i][0] = out_degree.get(i, 0)
            degs[i][1] = in_degree.get(i, 0)
        dataf["in_degree"] = degs[:, 1]
        dataf["out_degree"] = degs[:, 0]
        dataf["degree"] = dataf["in_degree"] + dataf["out_degree"]

        # unweighted adjacency matrix
        adj_ones = sparse.csc_matrix(
            (
                np.ones(edges.shape[0]),
                (edges["src_idx"].values, edges["dst_idx"].values),
            ),
            shape=(nnodes, nnodes),
        )
        adj_ones = adj_ones.tocsr()
        # make it simmetric to handle graph as undirected
        adj_ones += adj_ones.T
        # some weights might be doubled, so clip it to [0, 1]
        adj_ones.data = np.clip(adj_ones.data, 0, 1)

        # approx calculation of the nubmer of triangles every node is in
        # I'm not sure if it's right way, but it fast and produces good feature
        triags = adj_ones @ adj_ones
        triags = adj_ones.multiply(triags).sum(axis=0)

        dataf["triags"] = np.array(triags)[0]

        # whighted directed adjacency matrix
        adj = sparse.csc_matrix(
            (
                edges["edge_weight"].values,
                (edges["src_idx"].values, edges["dst_idx"].values),
            ),
            shape=(nnodes, nnodes),
        )
        adj = adj.tocsr()

        # normalized graph laplacian
        # matrix multiplication of normalized laplacian by the feature matrix 
        # is the same as one graph convolution, as they do in GNN
        S = sparse.csgraph.laplacian(adj + adj.T, normed=True)
        
        # convolutions with identity matrix as representation of mutual nodes proximity
        # mostly useful for graphs with no given features
        eyefeats1 = S @ sparse.identity(nnodes)
        eyefeats2 = S @ eyefeats1
        # convolutions of given features and handcrafted features
        ffeats = S @ dataf.values
        ffeats2 = S @ ffeats

        # label propagation. It leaks on validation, but it also improves score
        fulltarget = data[["node_index"]].merge(target, on="node_index", how="left")
        ohe_target = sparse.csr_matrix(pd.get_dummies(fulltarget["label"]).values)

        ohe_conv1 = adj @ ohe_target
        ohe_conv2 = adj @ ohe_conv1

        # stack everything together
        X = sparse.hstack(
            [
                sparse.csr_matrix(dataf.values),
                ffeats,
                ffeats2,
                eyefeats2,
                ohe_conv1,
                ohe_conv2,
            ]
        ).tocsr()

        # split between train and test
        train_mask = data["node_index"].isin(target["node_index"])
        test_mask = np.invert(train_mask)

        print(X.shape)
        # scaling needed only for VarienceThreshold feature selector
        ss = StandardScaler(with_mean=False)
        X = ss.fit_transform(X)
        selector = VarianceThreshold(0.2)
        X = selector.fit_transform(X)
        print(X.shape)

        return X, train_mask, target, test_mask

    def train_predict(self, data, time_budget, n_class, schema):
        # this method was called by platform in order to get predictions
        X, train_mask, target, test_mask = self.prepare_data(data)
        model = ExtraTreesClassifier(
            n_estimators=400, max_depth=70, n_jobs=4, random_state=1
        )
        model.fit(X[np.where(train_mask)], target["label"].values)

        preds = model.predict(X[np.where(test_mask)])

        return preds
