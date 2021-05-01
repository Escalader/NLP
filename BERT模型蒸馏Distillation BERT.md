# 简介  
模型蒸馏主要用于模型压缩。
## transformer历史模型参数(单位百万)  
![avatar](https://escalader.github.io/pictures/nlpmodel/hispara.png)  
模型的体积也限制其在现实世界中的使用，有一下因素：  
> - 这种模型的训练花费大量的金钱，需要使用昂贵的 GPU 服务器才能提供大规模的服务。
> - 模型太大导致 inference 的时间也变长，不能用于一些实时性要求高的任务中。
> - 现在有不少机器学习任务需要运行在终端上，例如智能手机，这种情况也必须使用轻量级的模型。  

常见的模型压缩的方法有一下几种：
> - 模型蒸馏Distillation，使用大模型学到的只是训练小模型，从而让小模型具有大模型的泛化能力。
> - 量化Quantization，降低大模型的精度，减小模型。
> - 剪枝Pruning，去掉模型汇总作用比较小的连接。
> - 参数共享，共享网络中部分参数，降低模型参数数量。  

# 模型蒸馏Distillation  
蒸馏是一种模型压缩的方法，有Hinton在论文《Distilling the konwledge in neural network》中提出，蒸馏的要点：  
> - 首先训练一个大的模型，大模型称为teacher模型。
> - 利用teacher模型输出的概率分布训练小模型，小模型称为student模型。
> - 训练student模型时，包含两种label，soft label对应了teacher模型输出的概率分布，而hard label是原来的one-hot label。
> - 模型蒸馏训练<b>小模型会学到大模型的表现以及泛化能力</b>  

在有监督训练中，模型会有一个预测的目标，通常是one-hot label，模型在训练时需要最大化对应label的概率(softmax或者log)。
在模型预测时的概率分布中，应该是正确的类别概率最大，其他类别的概率比较小，但是那些<b>小概率类别之间的概率大小应该也
是存在差异的</b>，例如将一只狗预测成浪的概率要比预测成树木的概率要大。这种概率之间的差异在一定程度上表明了模型的泛
化能力，一个好的模型不是拟合训练集好的模型，而是具有泛化能力的模型。  

模型蒸馏希望student模型学习到teacher模型的泛化能力，因此在训练模型时采用的target是teacher模型输出的概率分布，这也称为
soft target。有些模型蒸馏的方法在训练的时候也会使用原来的one-hot label，称为hard target。  

为了更好的学习到teacher模型的泛化能力，Hinton提出了softmax-temperature，公式如下:  
$$ p_i = \frac{exp(Z_i/T)}{\sum_{j}exp(Z_j/T)}$$  
softmax-temperature在softmax基础上添加了一个参数T，T越接近0，则分布越接近ont-hot，T越接近无穷大，则分布越接近平均分布。即T越大，分布会越平滑，选择合适的T可以让student模型更能观察到teacher模型的类别分布多样性。在训练的过程中，teacher模型和student模型使用同样的参数T，而后续使用student模型推断时，T设置回1，变成标准的softmax。
# DistilBERT
DistilBERT是huggingFace发布的，论文是《distilbert，啊distilled version of bert：small，faster，cheaper and lighter》，DistilBERT模型与BERT模型类似，但是DistilBERT只有6层，而BERT-base有12层，DistilBERT只有6600万参数，而 BERT-base有1.1亿参数。DistilBERT在减少BERT参数和层数的情况下，仍然可以保持比较好的性能，在GLUE上保留了BERT 95%的性能。
## DistilBERT训练
DistilBERT使用KL散度作为损失函数，q表示student模型的分布，而p表示teacher模型输出的分布，损失函数如下：  
$$ KL(p||q) = E_p(\log(\frac{p}{q})) = \sum_{i}p_i\times\log(p_i)-\sum_{i}p_i\times\log(q_i)$$

















