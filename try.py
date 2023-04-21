import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import os
import re
import jieba
from sklearn.metrics import accuracy_score

# # 下载nltk资源
# nltk.download('punkt')
# nltk.download('stopwords')

# # 示例数据：包含200个段落的列表
# data = ['paragraph_1', 'paragraph_2', ..., 'paragraph_200']

# 预处理文本
# def preprocess(text):
#     tokens = word_tokenize(text.lower())
#     tokens = [t for t in tokens if t.isalpha()]
#     stop_words = set(stopwords.words("english"))
#     tokens = [t for t in tokens if t not in stop_words]
#     return tokens

# processed_data = [preprocess(paragraph) for paragraph in data]
# print(processed_data)
def stopwordslst(addr):#获得停词表，返回一个被‘|’隔开的str
    stop_sum = 0  # 总的中文字符数
    stop_num_dic = {}  # 存储中文字符和出现次数的字典
    with open(addr, "r", encoding='utf-8', errors='ignore') as file1:
        fop = file1.readlines()
        for line in fop:
            wd = line.strip()#清除掉空格之类的乱七八糟的东西
            stop_sum += 1  # 总的中文字符数+1
            stop_num_dic[wd] = stop_num_dic.get(wd, 0) + 1
        del fop, line
    print('一共有：', stop_sum, ' 个停词')
    dic_lst = list(stop_num_dic.keys())
    return '|'.join(dic_lst)
'''得到停词表'''
stopstr = stopwordslst("cn_stopwords.txt")
print('停词表为：', stopstr)
'''获得一个文本前para_num个符合字数大于500的段落，去除无用字符和停词，返回一个list以及列表的实际尺寸num，list的内容为num个字符串'''
def words(txtfile, para_num):
    artic = []
    para = ''
    num = 0
    with open(txtfile, "r", encoding='ANSI', errors='ignore') as file:
        fp = file.readlines()
        for line in fp:
            line = line.replace(' ', '')
            line = line.replace('\u3000', '')
            line = line.replace('\t', '')
            para += line
            if line[-1] == '\n':
                para = para[:-1]
                para = re.sub(stopstr, '', para)
                if len(para) >= 500:
                    artic.append(para)
                    num += 1
                para = ''
                if num == para_num:
                    break
    del file
    return artic, num
'''去除无用字符和停词，获得一个文本前para_num个符合分词数大于350的段落，返回一个list以及列表的实际尺寸num，list的内容为num个字符串，分词被空格隔开'''
def fenci(txtfile, para_num):
    artic = []
    words = []
    para = ''
    num = 0
    with open(txtfile, "r", encoding='ANSI', errors='ignore') as file:
        fp = file.readlines()
        for line in fp:
            line = line.replace(' ', '')
            line = line.replace('\u3000', '')
            line = line.replace('\t', '')
            para += line
            if line[-1] == '\n':
                para = para[:-1]
                para = re.sub(stopstr, '', para)
                for x in jieba.cut(para):
                    words.append(x)
                if len(words) >= 350:
                    wordsstr = words
                    artic.append(wordsstr)
                    num += 1
                para = ''
                words = []
                if num == para_num:
                    break
    del file
    return artic, num

filepath = './ch/'#需要遍历的文件夹
txt_num = 10#文章个数
para_sum = 200#需要的段落数
artic_para = []
# 输出主题词的文件路径
top_words_csv_path = 'top-topic-words.csv'
# 输出各文档所属主题的文件路径
predict_topic_csv_path = 'words-distribution.csv'
html_path = 'visiual.html'
labels = []
lb = 0
for root, path, fil in os.walk(filepath):
    #将段落写入一个list
    for txt_file in fil:
        para_num = round(para_sum/txt_num)
        # artic,num = words(root+txt_file, para_num)
        artic,num = fenci(root+txt_file, para_num)
        for i in range(num):
            labels.append(lb)
        print('文件名称为：%s，获得的段落数为：%d'%(txt_file, num))
        para_sum -= num
        txt_num -= 1
        artic_para.extend(artic)
        lb += 1
print(labels)
processed_data = artic_para
true_labels = labels
# 构建词汇表
dictionary = Dictionary(processed_data)

# 构建语料库
corpus = [dictionary.doc2bow(text) for text in processed_data]

# 设置主题数量
num_topics = 10

# 训练LDA模型
lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# 获取文档的主题分布
doc_topics = [lda.get_document_topics(bow) for bow in corpus]

# 为每个文档分配最可能的主题
assigned_topics = []
for dt in doc_topics:
    topic_probs = [0] * num_topics
    for t, p in dt:
        topic_probs[t] = p
    assigned_topics.append(np.argmax(topic_probs))

# 计算分类准确率（如果有已知的主题标签）
print(assigned_topics)
accuracy = accuracy_score(true_labels, assigned_topics)
print("分类准确率：", accuracy)

# 打印主题关键词
for i in range(num_topics):
    print(f"主题 {i}:")
    print(" ".join([dictionary[word_id] for (word_id, _) in lda.get_topic_terms(i)]))
