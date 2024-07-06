6.15
1:main.py 添加101行用于在每个epoch训练前保存上一个epoch训练完的模型权重

2:main.py 添加103行用于给每个worker传入上一轮的所有模型权重（可以优化为在知道action后传入目标模型权重）

3:main.py 添加106和109行用于计算模型的前后accuracy 用于计算奖励

4: worker_vision.py 添加281行 在每次作出action前先进行 action sample

5:worker目录下添加 feature.py 用于计算模型的压缩权重作为state

   在worker_vision.py 里用feature.py中的方法定义了160行的self.feature（） 函数用于返回压缩权重

6:在worker_vision.py添加166行的select_action_sample()方法 用于对action进行采样并存储到buffer中

7:main.py 添加 152行用于wandb记录

8:main.py 添加61行用于IID数据分布，但是只关于cifar数据集
