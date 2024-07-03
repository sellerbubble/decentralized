from sklearn.decomposition import PCA

n_components = 2
pca = PCA(n_components=n_components)

import copy
import numpy as np

weights = np.array([[1, 2, 3, 4, 5, 6]])
weights = copy.deepcopy(weights)
print(weights)  # 输出: [1 2 3 4 5 6]

from sklearn.decomposition import PCA
import copy
import numpy as np

pca.fit(np.array(weights))
pca_weights = pca.transform(np.array(weights))

print(pca_weights)
