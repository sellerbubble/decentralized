from sklearn.decomposition import PCA
import copy
import numpy as np

def pca_weights(n_components, weights):
      pca = PCA(n_components = n_components)
      weights = copy.deepcopy(weights).flatten
      pca_weights = pca.transform(np.array(weights))
      return pca_weights

