from sklearn.decomposition import PCA
import copy
import numpy as np

def pca_weights(n_components, weights):
      pca = PCA(n_components = n_components)
      weights = copy.deepcopy(weights).flatten
      pca_weights = pca.transform(np.array(weights))
      return pca_weights

'''
pca = PCA(n_components = len(self.list_clients))


client_w_for_first_iteration = client.get_model()
                
weight_list_for_iteration.append(self.flatten(client_w_for_first_iteration))

# 得到weight list
weight_list_for_iteration_pca = pca.fit_transform(weight_list_for_iteration)

state_list = []

        for cpt in range(0, len(self.list_clients)):
            client_state = []
            
            client_state.append(list(weight_list_for_iteration_pca[cpt]))
            client_state.append(numbersamples_for_iteration[cpt])
            client_state.append(numbercores_for_iteration[cpt])
            client_state.append(frequencies_for_iteration[cpt])
            client_state.append(bandwidth_for_iteration[cpt])
            
            state_list.append(client_state)  
    
        # State is a concatenation of the different reduced weights
        state = self.flatten_state(state_list)

state_list[client_index][0] =  list((pca.transform(np.array(self.flatten(copy.deepcopy(client_w))).reshape(1, -1)))[0])
'''