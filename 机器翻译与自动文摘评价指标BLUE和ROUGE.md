    在机器翻译任务中，BLEU和ROUGE是两个常用的评价指标，BLEU根据精确率(Precision)衡量翻译的质量，而ROUGE根据召回率(Recall)衡量翻译的质量  
    
    
    
# 机器翻译评价指标 
使用机器学习的方法生成文本的翻译之后，需要评价模型翻译的性能，常见的评价指标有BLEU(2002)和ROUGE(2003)。一般用C表示机器翻译的译文，另外需要提供m个参考的翻译S1,S2..Sm。评价指标就可以
    衡量机器翻译的C和参考翻译S1....Sm的匹配程度。  
    
# BLEU(Bilingual evaluation understudy)  
分数取值范围0~1，分数越高，说明翻译质量越高。BLEU主要是基于精确率，整体公式如下  
![avatar](https://escalader.github.io/pictures/nlpmodel/hispara.png)
    
    
