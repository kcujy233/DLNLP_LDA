import os
import re
import time
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn

def top_words_data_frame(model: LatentDirichletAllocation,
                         tf_idf_vectorizer: TfidfVectorizer,
                         n_top_words: int) -> pd.DataFrame:
    '''
    求出每个主题的前 n_top_words 个词

    Parameters
    ----------
    model : sklearn 的 LatentDirichletAllocation 
    tf_idf_vectorizer : sklearn 的 TfidfVectorizer
    n_top_words :前 n_top_words 个主题词

    Return
    ------
    DataFrame: 包含主题词分布情况
    '''
    rows = []
    feature_names = tf_idf_vectorizer.get_feature_names_out()
    for topic in model.components_:
        top_words = [feature_names[i]
                     for i in topic.argsort()[:-n_top_words - 1:-1]]
        rows.append(top_words)
    columns = [f'topic word {i+1}' for i in range(n_top_words)]
    df = pd.DataFrame(rows, columns=columns)

    return df
def predict_to_data_frame(model: LatentDirichletAllocation, X: np.ndarray) -> pd.DataFrame:
    '''
    求出文档主题概率分布情况

    Parameters
    ----------
    model : sklearn 的 LatentDirichletAllocation 
    X : 词向量矩阵

    Return
    ------
    DataFrame: 包含主题词分布情况
    '''
    matrix = model.transform(X)
    columns = [f'P(topic {i+1})' for i in range(len(model.components_))]
    df = pd.DataFrame(matrix, columns=columns)
    return df
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

start_time = time.time()
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
                    wordsstr = ' '.join(words)
                    artic.append(wordsstr)
                    num += 1
                para = ''
                words.clear()
                if num == para_num:
                    break
    del file
    return artic, num


'''不分词，将每个字独立看待'''
filepath = './ch/'#需要遍历的文件夹
txt_num = 10#文章个数
para_sum = 200#需要的段落数
artic_para = []
# 输出主题词的文件路径
top_words_csv_path = 'top-topic-words.csv'
# 输出各文档所属主题的文件路径
predict_topic_csv_path = 'document-distribution.csv'
html_path = 'visiual.html'
for root, path, fil in os.walk(filepath):
    for txt_file in fil:
        para_num = round(para_sum/txt_num)
        # artic,num = words(root+txt_file, para_num)
        artic,num = fenci(root+txt_file, para_num)
        print('文件名称为：%s，获得的段落数为：%d'%(txt_file, num))
        para_sum -= num
        txt_num -= 1
        artic_para.extend(artic)
        print(len(artic_para))

        #使用tf_idf方法
    tf_idf_vectorizer = TfidfVectorizer()
    tf_idf = tf_idf_vectorizer.fit_transform(artic_para)
    # feature_names = tf_idf_vectorizer.get_feature_names_out()
    # print(type(feature_names))
    # matric = tf_idf.toarray()
    # df = pd.DataFrame(matric, columns=feature_names)
    # print(df)
    # #构造词频特征实例化
    # count_vectorizer = CountVectorizer()
    # cv = count_vectorizer.fit_transform(artic_para)
    # feature_names = count_vectorizer.get_feature_names_out()
    # matric = cv.toarray()
    # df = pd.DataFrame(matric, columns=feature_names)
    # print(df)
    topic_num = 10
    lda = LatentDirichletAllocation(
        n_components=topic_num, max_iter=200,
        learning_method='online',
        learning_offset=50,
        random_state=0,
        batch_size=4)
    lda.fit(tf_idf)
    # lda.fit(cv)
    top_words_df = top_words_data_frame(lda, tf_idf_vectorizer, 20)
    # top_words_df = top_words_data_frame(lda, count_vectorizer, 20)
    print(top_words_df)
    top_words_df.to_csv(top_words_csv_path, encoding='utf-8-sig', index=None)
    X = tf_idf.toarray()
    # X = cv.toarray()
    predict_df = predict_to_data_frame(lda, X)
    print(predict_df)
    predict_df.to_csv(predict_topic_csv_path, encoding='utf-8-sig', index=None)
    # data = pyLDAvis.sklearn.prepare(lda, cv, count_vectorizer,mds='mmds')
    # data = pyLDAvis.sklearn.prepare(lda, tf_idf, tf_idf_vectorizer)
    data = pyLDAvis.sklearn.prepare(lda, tf_idf, tf_idf_vectorizer, mds='mmds')
    pyLDAvis.save_html(data, html_path)
    break
    

#可视化
# 使用 pyLDAvis 进行可视化
data = pyLDAvis.sklearn.prepare(lda, tf_idf, tf_idf_vectorizer)
pyLDAvis.save_html(data, html_path)
# 清屏
os.system('clear')
# 浏览器打开 html 文件以查看可视化结果
os.system(f'start {html_path}')

# print('本次生成了文件：',
#       top_words_csv_path,
#       predict_topic_csv_path,
#       html_path)
#构造模型
topic_num = 2
lda = LatentDirichletAllocation(
    n_components=topic_num, max_iter=50,
    learning_method='online',
    learning_offset=50,
    random_state=0)

