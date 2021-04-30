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
BERT在训练的过程中采用了NSP loss，原本用意是为了让模型能够更好的捕捉文本的语义，在给定两段语句X= [x1,x2,x3,.....,xn]和 y = [y1,y2,y3,......ym]，bert中的nsp任务需要预测y是不是出现在x的后面。  

但是不少papers对nsp loss持质疑态度，例如XLNet和RoBERTa，其采用了一系列实验验证nsp损失的实用性。RoBERTa实验中的四种组合如下：
> - Segment-Pair+NSP:这个是原来 BERT 的训练方法，使用 NSP Loss，输入的两段文字 X 和 Y 可以包含多个句子，但是 X + Y 的长度要小于 512。
> - Sentence-Pair + NSP：与上一个基本类似，也使用 NSP Loss，但是输入的两段文字 X 和 Y 都分别是一个句子，因此一个输入包含的 token 通常比 Segment-Pair 少，所以要增大 batch，使总的 token 数量和 Sentence-Pair 差不多。
> - Full-Sentences：不使用 NSP，直接从一个或者多个文档中采样多个句子，直到总长度到达 512。当采样到一个文档末尾时，会在序列中添加一个文档分隔符 token，然后再从下一个文档采样。
> - Doc-Sentences：与 Full-Sentences 类似，不使用 NSP，但是只能从一个文档中采样句子，所以输入的长度可能会少于 512。Doc-Sentences 也需要动态调整 batch 大小，使其包含的 token 数量和 Full-Sentences 差不多。  
![avatar](https://escalader.github.io/pictures/nlpmodel/nspzy.png)  
上图是实验结果，最上面的两行是使用 NSP 的，可以看到使用 Segment-Pair (多个句子) 要好于 Sentence-Pair (单个句子)，实验结果显示使用单个句子会使 BERT 在下游任务的性能下降，主要原因可能是使用单个句子导致模型不能很好地学习长期的依赖关系。  

上图是实验结果，最上面的两行是使用 NSP 的，可以看到使用 Segment-Pair (多个句子) 要好于 Sentence-Pair (单个句子)，实验结果显示使用单个句子会使 BERT 在下游任务的性能下降，主要原因可能是使用单个句子导致模型不能很好地学习长期的依赖关系。

## 动态Mask  
原始的BERT在训练之前就把数据Mask了，然后在整个训练过程中都是保持数据不变的，称为Static Mask。即同一个句子在整个训练过程中，Mask掉的单词都是一样的。 

RoBERTa使用了一种Dynamic Mask的策略，将整个数据集复制10次，然后在10个数据集上都Mask一次，也就是每一个句子都会有10种Mask结果。使用10个数据集训练BERT。

下图是实验结果，可以看到使用 Dynamic Mask 的结果会比原来的 Static Mask 稍微好一点，所以 RoBERTa 也使用了 Dynamic Mask。  
![avatar](https://escalader.github.io/pictures/nlpmodel/dynamicmask.png)  
## 更大的batch  
之前的一些关于神经网络翻译的研究显示了使用一个大的batch并相应地增大学习率，可以加速优化并且提升性能。RoBERTa也对batch大小进行了实验，原始的BERT使用的batch=256，训练步数为1M，这与batch=2K，训练步数125K的计算量是一样的，与batch=8K和训练步数为31K也是一样的。下图是使用不同batch的实验结果，不同batch学习率是不同的，可以看到使用batch=2K时的效果最好。  
![avatar](https://escalader.github.io/pictures/nlpmodel/yzbatch.png)  
# ALBERT  
BERT的预训练模型参数量很多，训练时候的时间也比较久。ALBERT是一个对BERT进行压缩后的模型，降低了BERT的参数量，减少了训练所需的时间。  

<b>ALBERT只是减少BERT的参数量，而不减少其计算量。ALBERT能减少训练时间，这是因为减少了参数之后可以降低分布式训练时候的通讯量；ALBERT不能减少inference的时间，因为inference的时候经过的 Transformer计算量和BERT还是一样的。</b>  
## Factorized embedding parameterization  
对embedding进行分解，从而减少参数。在bert中，embedding的维度和transformer隐藏层维度是一样的，都是H。假设词库的大小为V，则单词的embedding矩阵参数量就有VH，如果词库很大，则参数量会很多。ALBER使用了一种基于Factorized的方法，不是直接把单词的one-hot矩阵映射到H维的向量，而是先映射到一个低维空间(E维)，再映射到H维的空间，这个过程类似做了一次矩阵分解。  
> - bert embedding参数量 = O(V*H)
> - albert embedding参数量 = O(V*E+E*H)

