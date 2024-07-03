import numpy as np
def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    n_classes = train_labels.max()+1 
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例 K 是 10个类别， N代表N个clients 每个clients拥有该类别数据的比例
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合 10个类别 每个类别对应的所有索引
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # k idcs 是每个类别的所有索引 fracs是每个类别针对所有clients的比例
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        # 这里(np.cumsum(fracs)[:-1]*len(k_idcs))生成每个类别的对不同clients的划分点，然后用划分点来划分每个类别的所有索引
                                          
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]
    # 将每个clients不同类别的索引列表拼成一个列表
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs