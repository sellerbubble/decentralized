from networks import load_model
from sklearn.decomposition import PCA



import copy
import numpy as np
import time
import torch

model = load_model("ResNet18_M", 10, pretrained=True).to(
            "cpu"
        )

dic = model.state_dict()

min_shape = 0
result = None 

for value in dic.values():
    # print(len(value.shape))
    
    if len(value.shape) == 4 and value.shape[-2] == 3 and value.shape[-1] == 3:
        value = value.view(value.size(0), value.size(1), -1)
        # print(value.shape)
        value = value.permute(2, 0 ,1)
        value = value.view(value.size(0), -1)
        print(value.shape)
        if min_shape == 0:
            min_shape = value.shape[-1]
        else:
            min_shape = min(min_shape, value.shape[-1])
        print(min_shape)
        if value.shape[-1] > min_shape:
            value = value.view(value.size(0), min_shape, -1)
            value = value.mean(dim=-1)
        print(value.shape)
        if result == None:
            result = value
        else:
            result = torch.cat((result, value), dim=0)

        # 
        
print(result.shape)
n_components = result.shape[0]
pca = PCA(n_components=n_components)
time1 = time.time()
weights = np.array(result)
print(pca.fit_transform(weights))
print(time.time()-time1)

        