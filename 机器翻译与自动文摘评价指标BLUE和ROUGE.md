    在机器翻译任务中，BLEU和ROUGE是两个常用的评价指标，BLEU根据精确率(Precision)衡量翻译的质量，而ROUGE根据召回率(Recall)衡量翻译的质量  
    
    
    
# 机器翻译评价指标 
使用机器学习的方法生成文本的翻译之后，需要评价模型翻译的性能，常见的评价指标有BLEU(2002)和ROUGE(2003)。一般用C表示机器翻译的译文，另外需要提供m个参考的翻译S1,S2..Sm。评价指标就可以
    衡量机器翻译的C和参考翻译S1....Sm的匹配程度。  
    
# BLEU(Bilingual evaluation understudy)  
分数取值范围0~1，分数越高，说明翻译质量越高。BLEU主要是基于精确率，整体公式如下  
![avatar](https://escalader.github.io/pictures/nlpmodel/b1.png)  
        BLEU需要计算译文1-gram，2-gram...N-gram的精确率，一般N设为4即可，Pn指n-gram的精确率。
        Wn：n-gram的权重，一般设为均匀权重，即对于任意n都有Wn=1/N。
        BP：是惩罚因子，如果译文的长度小于最短的参考疑问，则BP小于1。
        BLEU的1-gram精确率表示译文忠于原文的程度，而其他n-gram表示翻译的流畅程度。  
        
 ## n-gram精确率计算 
 假设机器翻译的译文C和一个参考翻译S1如下：
        C: a cat is on the table
        S1: there is a cat on the table   
        
 则可以计算出1-gram，2-gram...的精确率   
 
 ![avatar](https://escalader.github.io/pictures/nlpmodel/b2.png)  
 
 直接这样直接计算precision会存在问题，例如：
        C:there there there there
        S1：there is a cat on the table

这时翻译结果是不正确的，但是1-gram的precision=1，因此BLEU会使用修正的方法。给定参考译文s1,s2...sm，可以计算C里面n元祖的precision，计算公式如下：  

![avatar](https://escalader.github.io/pictures/nlpmodel/b3.png)  

## 惩罚因子
BLEU计算n-gram精确率的方法，仍然存在一些问题，当机器翻译的长度比较短时，BLEU得分也会比较高，但是翻译会损失很多的信息，例如：
         C:a cat
         S1：there is a cat on the table  
         
因此需要在BLEU分数乘上惩罚因子   

![avatar](https://escalader.github.io/pictures/nlpmodel/b4.png)  

# ROUGE(Recall-Oriented Understudy for Gisting Evaluation) 
主要基于召回率。ROUGE 是一种常用的机器翻译和文章摘要评价指标，由 Chin-Yew Lin 提出，其在论文中提出了 4 种 ROUGE 方法：
- ROUGE-N: 在 N-gram 上计算召回率
- ROUGE-L: 考虑了机器译文和参考译文之间的最长公共子序列
- ROUGE-W: 改进了ROUGE-L，用加权的方法计算最长公共子序列
- ROUGE-S: 采用了 Skip 的 N-gram，可以允许 N-gram 的单词不连续
## ROUGE-N
ROUGE-N 主要统计 N-gram 上的召回率，对于 N-gram，可以计算得到 ROUGE-N 分数，计算公式如下：  

![avatar](https://escalader.github.io/pictures/nlpmodel/b5.png)  

分母统计在参考译文中N-gram个数，分子统计参考译文与机器译文共有的N-gram个数。  
        C:a cat is on the table
        S1: there is a cat on the table  
        
上面例子中ROUGE-1和ROUGE-2分数如下：  

![avatar](https://escalader.github.io/pictures/nlpmodel/b6.png) 

如果给定多个 参考译文 Si，Chin-Yew Lin 也给出了一种计算方法，假设有 M 个译文 S1, ..., SM。ROUGE-N 会分别计算机器译文和这些参考译文的 ROUGE-N 分数，并取其最大值，公式如下。这个方法也可以用于 ROUGE-L，ROUGE-W 和 ROUGE-S。  
![avatar](https://escalader.github.io/pictures/nlpmodel/b7.png)  

## ROUGE-L 
ROUGE-L 中的 L 指最长公共子序列 (longest common subsequence, LCS)，ROUGE-L 计算的时候使用了机器译文 C 和参考译文 S 的最长公共子序列，计算公式如下：  

![avatar](https://escalader.github.io/pictures/nlpmodel/b8.png)   

公式中的 R_LCS 表示召回率，而 P_LCS 表示精确率，F_LCS 就是 ROUGE-L。一般 beta 会设置为很大的数，因此 F_LCS 几乎只考虑了 R_LCS (即召回率)。注意这里 beta 大，则 F 会更加关注 R，而不是 P，可以看下面的公式。如果 beta 很大，则 P_LCS 那一项可以忽略不计。  

![avatar](https://escalader.github.io/pictures/nlpmodel/b9.png)  

## ROUGE-W
ROUGE-W 是 ROUGE-L 的改进版，考虑下面的例子，X 表示参考译文，而 Y1，Y2 表示两种机器译文。 

![avatar](https://escalader.github.io/pictures/nlpmodel/b10.png)  

Y1 的翻译质量更高，因为 Y1 有更多连续匹配的翻译。但是采用 ROUGE-L 计算得到的分数确实一样的，即 ROUGE-L(X, Y1)=ROUGE-L(X, Y2)。因此作者提出了一种加权最长公共子序列方法 (WLCS)，给连续翻译正确的更高的分数，具体做法可以阅读原论文《ROUGE: A Package for Automatic Evaluation of Summaries》。
## ROUGE-S
ROUGE-S 也是对 N-gram 进行统计，但是其采用的 N-gram 允许"跳词 (Skip)"，即单词不需要连续出现。例如句子 "I have a cat" 的 Skip 2-gram 包括 (I, have)，(I, a)，(I, cat)，(have, a)，(have, cat)，(a, cat)。
    
