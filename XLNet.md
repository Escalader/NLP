# 简介
BERT训练时将部分单词mask起来，是模型能够利用句子双向的信息。在很多NLU任务上获得很好的效果。
但是BERT忽略了mask单词之间的关系，并且微调过程与
预训练过程不一致(微调时没有mask的单词)。XLNet采用了PLM，将句子随机排列，然后用子回归的方法训练，
从而获得双向信息并且可以学习token之间的依赖关系。另外XLNet说那个Transformer-XL，使用了更广阔的上下文信息。  

XLNet将当前预训练模型分为了两类AR(Auto Regression，自回归)，和AE(Auto Encoder，自编码)。  

GPT就是一种AR方法，不断地使用当前得到的信息预测一下个输出(自回归)。而BERT是一种AE方法，将输入句子的某些单词mask掉，
然后再通过BERT还原数据，这一过程类似去噪自编码器(DAE)。  

AR的方法可以更好的学习token之间的依赖关系，而AE的方法可以更好的利用深层的双向信息。因此XLNet希望将AR和AE两种方法
的有点结合起来，XLNet使用了Permutation Language Model(PLM)实现这一目的。  

Permutation指排列组合的意思，XLNet将句子中的token随机排列，然后采用AR的方式预测末尾的几个token。这样一来，
在预测token的时候就可以同时利用该token双向的信息，并且能学到token间的依赖，如下图所示。  
![avatar](https://escalader.github.io/pictures/nlpmodel/1.png)  

XLNet为了实现PLM，提出了Two-StreamSelf-attention 和Partial Prediction。另外XLNet还使用了Transformer-XL中的
Segment Recurrence Mechanism和Relative Position Encoding。  

# Permutation Language Model
PLM是XLNet的核心思想，首先将句子的token随机排列，然后采用AR的方式预测句子末尾的单词，这样XLNet即可同时拥有AE和AR的
优势。
## PLM
XLNet中通过Attention Mask实现PLM，而无需真正修改句子token的顺序。例如原来的句子是[1,2,3,4]，如果随机生成序列是
[3,2,4,1]，则输入的句子仍然是[1,2,3,4]，但是掩码需要修改成下图。
![avatar](https://escalader.github.io/pictures/nlpmodel/2.png)  
图中的掩码矩阵，红色表示不遮掩，白色表示遮掩。第1行表示token 1 的编码，可以看到，1是句子最后一个token，
因此可以看到之前所有token(3,2,4)。3是句子的第一个token，看不到句子的任何信息，因此第三行都是白色的(表示遮掩)。

## Two-Stream Self-attention  
### two-stream
XLNet打乱了句子的顺序，这时在预测的时候token的位置信息会非常重要，同时在预测的时候必须将token的内容信息遮掩起来
(否则输入包含了要预测的内容信息，模型就无法学习到知识)。<b>也就是说XLNet需要看到token的位置信息，但是又不能看到
  token的内容信息</b>，因此XLNet采用了两个Stream实现这一目的：
  
  - Query Stream，对于每一个token，其对应的Query Stream只包含了该token的位置信息，注意是token在原始句子的位置信息，
  不是重新排列的位置信息。
  - Content Stream，对于每一个token，其对应的Content Strean包含了该token的内容信息。  
  
### query stream计算
query stream用g表示，ContentStream用h表示使用query stream对要预测的位置进行预测的时候，Q(Query)向量是用g计算得到的，包含
包含该位置的位置信息，而K(Key)和V(Value)使用h计算的，包含其他token的内容信息。下图展示了如何通过当前层的g计算下一层g的过层
，涂总的排列是[3,2,4,1]，计算的token是1。  
![avatar](https://escalader.github.io/pictures/nlpmodel/3.png)  
可以看到在计算token 1 的Q向量时，只使用了token 1的QueryStream g，即模型只得到token 1的位置信息。而向量K,V使用token 3，2
4进行计算，所以模型可以得到token 3，2，4的内容信息。因为token 1是排列[3，2，4，1]的最后一位。这个过程的掩码矩阵
和上一节是一样，对角线上都是白色，即遮掩当前预测位置的内容信息h。  
![avatar](https://escalader.github.io/pictures/nlpmodel/4.png)  
### Content Stream 计算
Content Stream包含了token的内容信息，因为XLNet的层数很多，需要将token的内容传递到下一层。这一层的Q、K、V都是利用h计算
的。Content Stream的计算如下图：  
![avatar](https://escalader.github.io/pictures/nlpmodel/5.png)  

可以看到，在计算下一层的h1时，也会利用token 1当前的内容信息，这样就可以将token的内容传递到一下层，但是注意XLNet
在预测时只是用g(Query Stream)。计算Content Stream时候的掩码矩阵下图：  
![avatar](https://escalader.github.io/pictures/nlpmodel/6.png)  
和Query Stream的掩码矩阵区别在于对角线，Content Stream 不遮掩对角线，使得当前token的信息可以传递到下一层。
### Query Stream和Content  Stream组合  
XLNet将Query Stream和Content Stream组合在一起，如下图：  
![avatar](https://escalader.github.io/pictures/nlpmodel/7.png)   
图中最下面的一层是输入层，其中e(x)是单词的词向量，表示输入的Content Stream，而W表示输入的位置信息，即Query Stream.
## Partial Prediction
XLNet 将句子重新排列，然后根据排列后的顺序使用 AR 方式预测，但是由于句子是随机排列的，会导致优化比较困难且收敛速度慢。因此 XLNet 采用了 Partial Prediction (部分预测) 
的方式进行训练，对于排列后的句子，只预测句子末尾的1/K个token。  

例如 K=4，就是只预测最后1/4的token。给定句子 [1,2,3,4,5,6,7,8] 和一种随机排列 [2,8,3,4,5,1,7,6]，
则只预测7和6。论文中训练 XLNet-Large 时使用的 K 为 6，大约是预测末尾14.3%的token。

# XLNet优化技巧
## Tramsformer-XL
XLNet使用了Transformer—XL中的SegmentRecurrenceMechanism(段循环)和Relative Positional Encoding(相对位置编码)
进行优化。  
Segment Recurrence Mechanism 段循环的机制会将上一段文本输出的信息保存下来，用于当前文本的计算，使模型可以拥有更广阔的上下文信息。  
在引入上一段信息后，可能会有两个 token 拥有相同的位置信息，例如上一段的第一个单词和当前段的第一个单词位置信息都是一样的。因此 Transformer-XL 采用了 Relative Positional Encoding (相对位置编码) ，
不使用固定的位置，而是采用单词之间的相对位置进行编码。  
XLNet 使用了 Transformer-XL 后如下图所示。mem 表示的就是前一个 XLNet 段的内容信息，而 XLNet 中输入的 Query Stream 为 w，保存位置信息，采用的是 Relative Positional Encoding。  
![avatar](https://escalader.github.io/pictures/nlpmodel/8.png)    
## Relative Segment Encoding  
XLNet 希望像 BERT 一样采用 [A, SEP, B, SEP, CLS] 的形式处理句子任务，在 BERT 中有两个表征向量 EA 和 EB 分别表示句子 A 和 B。但是 XLNet 采用 Transformer-XL 的段循环机制后会出现问题，两个段都有句子 A 和 B，则两个句子 A 属于不同的段，但是却会有相同的 Segment 向量。  

XLNet 提出了 Relative Segment Encodings，对于每一个 attention head 都添加 3 个可训练的向量 s+, s-, b，然后利用以下公式计算 attention score。
![avatar](https://escalader.github.io/pictures/nlpmodel/9.png)  
其中 q 就是 Query 向量，这个计算得到的 attention score 会加到原来的 attention score 上，再计算 softmax。Relative Segment Encodings 加上了一个偏置向量 b，同时 Relative Segment Encodings 也可以用于一些超过两段输入句子的任务上。  

# 小结
XLNet 的核心思想是 PLM，排列原来的句子，然后预测末尾的单词。这样可以学习到单词之间的依赖关系，而且可以利用 token 前后向的信息。  

XLNet PLM 的实现需要用到 Two-Stream Self-Attention，包含两个 Stream，Query Stream 用于预测，只包含当前位置的位置信息。而 Content Stream 保存了 token 的内容。  

XLNet 还使用了 Transformer-XL 的优化方式。
  
  
