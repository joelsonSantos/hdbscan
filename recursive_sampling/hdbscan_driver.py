import hdbscan
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd


class RecursiveSampling:

    def __init__(self, X, mpts, capacity, n_samples):
        self.X = X
        self.mpts = mpts
        self.capacity = capacity
        self.n_samples = n_samples

    # return MST as pandas DataFrame
    def get_local_extended_mst(self, X, mpts):
        model = hdbscan.HDBSCAN(min_cluster_size=4, algorithm='generic', gen_min_span_tree=True)
        return model.fit(X).minimum_spanning_tree_.to_pandas()
    
    def get_data_bubbles_hierarchy(self, X, n_samples):
        seeds = X.sample(n=min(n_samples, X.shape[0]))
        return seeds

if __name__ == '__main__':
    dataset = load_iris()
    X = pd.DataFrame(data=dataset.data)
    y = dataset.target
    
    # Recursive Sampling calls
    model = RecursiveSampling(X=X, mpts=4, capacity=50, n_samples=10)
    print(model.get_data_bubbles_hierarchy(X=X, n_samples=10))
    print(model.get_local_extended_mst(X=X, mpts=4))