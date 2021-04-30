# 简介
## 特点
### RoBERTa
> - 更大的训练集，更大的batch
> - 不需要使用NSP Loss
> - 使用更长的训练Sequence
> - 动态mask
### ALBERT
> - 分解Embedding矩阵，减少维度。
> - 所有Transformer层参数共享
> - 使用SOP(Sentence Order Predict)代替NSP
# RoBERTa  
RoBERTa主要试验了bert中的一些训练设置(NSP loss 是否有意义，batch的大小等)，并找出最好的设置，然后在更大的数据集上训练bert。  
## 更大的数据集  
bert：16G，RoBERTa：160G：
> - BOOKCORPUS，16G，原来的BERT训练数据集
> - CC-NEWS 76G
> - OPENWEBTEXT，38G
> - STORIES，31G
## 去掉NSP loss
