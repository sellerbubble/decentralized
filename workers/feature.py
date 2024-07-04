from sklearn.decomposition import PCA
import numpy as np
import torch
import time

def pca_weights(n_components, weights):
      # weights 是 state dict
      for value in weights.values():
    
            if len(value.shape) == 4 and value.shape[-2] == 3 and value.shape[-1] == 3:
                  value = value.view(value.size(0), value.size(1), -1)
                  # print(value.shape)
                  value = value.permute(2, 0 ,1)
                  value = value.view(value.size(0), -1)
                  # print(value.shape)
                  if min_shape == 0:
                        min_shape = value.shape[-1]
                  else:
                        min_shape = min(min_shape, value.shape[-1])
                  # print(min_shape)
                  if value.shape[-1] > min_shape:
                        value = value.view(value.size(0), min_shape, -1)
                        value = value.mean(dim=-1)
                  # print(value.shape)
                  if result == None:
                        result = value
                  else:
                        result = torch.cat((result, value), dim=0)

                  # 
                  
            # print(result.shape)
            n_components = result.shape[0]
            pca = PCA(n_components=n_components)
            time1 = time.time()
            # weights = np.array(result)
            # print(pca.fit_transform(result))
            print(time.time()-time1)
            return pca.fit_transform(result)

                  

            # 
            #pca[0] =  (pca.transform(np.array(self.flatten(copy.deepcopy(self.model.state_dict()))).reshape(1, -1)))[0]