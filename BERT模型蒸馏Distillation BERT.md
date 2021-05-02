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

DistilBERT最终的损失函数有KL散度(蒸馏损失)和MLM(遮蔽语言建模)损失两部分线性组合得到。DistillBERT一出来BERT模型的token类型embedding和NSP(下一句预测任务)，保留了BERT的其他机制，然后把BERT的层数减少为原来的1/2。  

此外DistilBERT还是用了一些优化的trick，例如，使用teacher模型的参数对DistilBERT模型进行初始化；采用RoBEERTa中的一些训练方法，例如：大的batch，动态mask等。  
## DistillBERT实验结果  

![avatar](https://escalader.github.io/pictures/nlpmodel/distires.png)  

上图是DistilBERT在GLUE基准开发集上的实验结果，可以看到在所有的数据集上，DIstBERT的效果逗比ELMO好，在一些数据上
效果甚至比BERT还好，整体性能也达到了BERT的97%，但是DistilBERT的参数量只有BERT的60%。如下图：  
![avatar](https://escalader.github.io/pictures/nlpmodel/distipa.png)  

上图是不同模型参数的以及推算时间的对比，可以看到DistiBERT的参数比ELMo和BERT-base都少很多，而且推算时间也大大缩短。  
# 将BERT蒸馏到BiLSTM
出自文章《distilling task-specific knowledge FORM bert into simple neural networks》，将BERT模型蒸馏到bilstm中，称为Distiled BiLSTM。即taecher模型时bert，而student模型时BiLSTM。文章提出来两种模型，其中一个是针对单个句子的分类；另一个是针对两个句子做匹配。  
## Distilled BiLSTM模型  
![avatar](https://escalader.github.io/pictures/nlpmodel/distibilstm.png)  

上图是第一种BiLSTM模型，用于单个句子分类，讲句子中所有单词的词向量输入一个BiLSTM，然后将前向和后向LSTM的隐藏向量拼接在一起，传图全链接网络中进行分类。  

![avatar](https://escalader.github.io/pictures/nlpmodel/distibilstm2.png)  

上面是第二种BiLSTM模型，用于两个两个句子进行匹配，两个BiLSTM输出的隐藏向量分别为h1和h2，则需要将两个向量拼接在一起，在进行分类。h1和h2拼接公式如下：
$$f(h_1,h_2) = [h_1,h_2,h_1\odot h_2,|h_1-h_2|]$$  

⊙表示点乘
## Distiled BiLSTM训练  
将bert蒸馏到BiLSTM模型，使用的损失函数包括两个部分：  

- 一部分是hard target，直接使用one-hot类别与BiLSTM输出的概率值计算交叉熵。
- 一部分是soft target，使用teacher模型(BERT)输出的概率值与BiLSTM输出的概率值计算均方误差MSE。  

$$ y = softmax(Z) $$
$$ l_{distil} = {||Z^{(B)}-Z^{(S)}||}_2^2 $$

$$ L = \alpha\cdot L_{CE}+(1-\alpha)\cdot L_{distil}$$

$Z^{(B)}$是teacher模型的输出；$Z^{(B)}$是student模型的输出  
在训练过程中，太小的数据集不足以rangstudent模型学习到teacher模型的所有知识，所以作者提出了三种数据增强的方法扩充数据：  

- Masking，使用[Mask]标签随机替换一个单词，例如"i have a cat"，替换成"i [mask] a cat"
- POS-guided word replacement，将一个单词替换成另一个具有相同POS的单词，例如将"i have a cat"替换成" i have a dog"
- n-gram，在1-5中随机采样得到n，然后采用n-gram，去掉其他词。
## Distilled BiLSTM实验结果  
![avatar](https://escalader.github.io/pictures/nlpmodel/disbilprere.png)  

上面是DIstiled BilSTM模型的结果，可以看到比单纯使用BiLSTM模型的效果好很多，在SST和QQP数据集上的效果甚至比ELMO好，说明模型能够学习到一部分BERT的泛化能力。但是Distilled BiLSTM的效果还是比BERT差了很多，说明还是有很多知识不能迁移到BiLSTM中。  

![avatar](https://escalader.github.io/pictures/nlpmodel/distibilstmpa2.png)  

上面是Distilled BiLSTM的参数和推断时间，BiLSTM的参数要远远少于BERT-large，比BERT-large少了335倍，推断时间比BERT-large快乐434倍。压缩效果还是比较明显的。

# 不同蒸馏模型总结  
DistilBERT模型的效果相对较好，而Distiled BiLSTM压缩的更小。  

DistilBERT模型使用了KL散度计算soft target，而Distilled BiLSTM使用MSE计算。HuggingFace在博客中给出的原因是，DistilBERT训练的是语言模型，而Distilled BiLSTM针对下游分类任务，语言模型的输出空间维度要大很多，这种时候使用
MSE可能不同logit之间会相互抵消。












