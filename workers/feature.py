from sklearn.decomposition import PCA
import copy
import numpy as np

def pca_weights(n_components, weights):
      pca = PCA(n_components = n_components)
      weights = copy.deepcopy(weights)
      pca_weights = pca.fit_transform(np.array(weights).flatten().reshape(-1, 1))
      return pca_weights

# 
#pca[0] =  (pca.transform(np.array(self.flatten(copy.deepcopy(self.model.state_dict()))).reshape(1, -1)))[0]