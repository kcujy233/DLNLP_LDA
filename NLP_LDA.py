import os
import re
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.metrics import accuracy_score

class LDA_DATA():
    def __init__(self, filepath, txtnum, parasum=200) -> None:
        self.filepath = filepath
        self.txtnum = txtnum
        self.parasum = parasum
        self.stop_addr = "cn_stopwords.txt"
        self.para_num = 0
        self.words_stop = ''
        self.truelabels = []
        self.artic_para = []
    def stopwordslst(self):#获得停词表，返回一个被‘|’隔开的str
        stop_sum = 0  # 总的中文字符数
        stop_num_dic = {}  # 存储中文字符和出现次数的字典
        with open(self.stop_addr, "r", encoding='utf-8', errors='ignore') as file1:
            fop = file1.readlines()
            for line in fop:
                wd = line.strip()#清除掉空格之类的乱七八糟的东西
                stop_sum += 1  # 总的中文字符数+1
                stop_num_dic[wd] = stop_num_dic.get(wd, 0) + 1
            del fop, line
        print('一共有：', stop_sum, ' 个停词')
        dic_lst = list(stop_num_dic.keys())
        self.words_stop = '|'.join(dic_lst)
        return self.words_stop
    def words(self,txtpath):
        artic = []
        para = ''
        num = 0
        with open(txtpath, "r", encoding='ANSI', errors='ignore') as file:
            fp = file.readlines()
            for line in fp:
                line = line.replace(' ', '')
                line = line.replace('\u3000', '')
                line = line.replace('\t', '')
                para += line
                if line[-1] == '\n':
                    para = para[:-1]
                    para = re.sub(self.words_stop, '', para)
                    parastr = ' '.join(para)
                    if len(para) >= 500:
                        artic.append(parastr)
                        parastr = ''
                        num += 1
                    para = ''
                    if num == self.para_num:
                        break
        del file
        return artic, num
    def fenci(self,txtpath):
        artic = []
        words = []
        para = ''
        num = 0
        with open(txtpath, "r", encoding='ANSI', errors='ignore') as file:
            fp = file.readlines()
            for line in fp:
                line = line.replace(' ', '')
                line = line.replace('\u3000', '')
                line = line.replace('\t', '')
                para += line
                if line[-1] == '\n':
                    para = para[:-1]
                    para = re.sub(self.words_stop, '', para)
                    for x in jieba.cut(para):
                        words.append(x)
                    if len(words) >= 350:
                        wordsstr = ' '.join(words)
                        artic.append(wordsstr)
                        num += 1
                    para = ''
                    words.clear()
                    if num == self.para_num:
                        break
        del file
        return artic, num
    def data_load_fenci(self):
        lb = 0
        for root, path, fil in os.walk(self.filepath):
    #将段落写入一个list
            for txt_file in fil:
                self.para_num = round(self.parasum/self.txtnum)
                artic,num = self.fenci(root+txt_file)
                for i in range(num):
                    self.truelabels.append(lb)
                print('文件名称为：%s，获得的段落数为：%d'%(txt_file, num))
                self.parasum -= num
                self.txtnum -= 1
                self.artic_para.extend(artic)
                lb += 1
        print('最后的总段落数为：', len(self.artic_para))
        return self.artic_para, self.truelabels
    def data_load_words(self):
        lb = 0
        for root, path, fil in os.walk(self.filepath):
    #将段落写入一个list
            for txt_file in fil:
                self.para_num = round(self.parasum/self.txtnum)
                artic,num = self.words(root+txt_file)
                for i in range(num):
                    self.truelabels.append(lb)
                print('文件名称为：%s，获得的段落数为：%d'%(txt_file, num))
                self.parasum -= num
                self.txtnum -= 1
                self.artic_para.extend(artic)
                lb += 1
        print('最后的总段落数为：', len(self.artic_para))
        return self.artic_para, self.truelabels

def top_words_data_frame(model: LatentDirichletAllocation,
                         tf_idf_vectorizer: TfidfVectorizer,
                         n_top_words: int) -> pd.DataFrame:
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
    prelst = []
    matrix = model.transform(X)
    for i in range(matrix.shape[0]):
        max_index = matrix[i].argmax()
        prelst.append(max_index)
    columns = [f'P(topic {i+1})' for i in range(len(model.components_))]
    df = pd.DataFrame(matrix, columns=columns)
    return df, prelst

def TF_IDF(ldainput, topic_num, iternum, top_words_csv_path, predict_topic_csv_path, html_path):
    tf_idf_vectorizer = TfidfVectorizer()
    # tf_idf_vectorizer = TfidfVectorizer(analyzer='char')
    tf_idf = tf_idf_vectorizer.fit_transform(ldainput)
    # feature_names = tf_idf_vectorizer.get_feature_names_out()
    # print(type(feature_names))
    # matric = tf_idf.toarray()
    # df = pd.DataFrame(matric, columns=feature_names)
    lda = LatentDirichletAllocation(
        n_components=topic_num, max_iter=iternum,
        learning_method='online',
        learning_offset=50,
        random_state=0)
    lda.fit(tf_idf)
    top_words_df = top_words_data_frame(lda, tf_idf_vectorizer, 10)
    top_words_df.to_csv(top_words_csv_path, encoding='utf-8-sig', index=None)
    X = tf_idf.toarray()
    predict_df, gen_labels = predict_to_data_frame(lda, X)
    predict_df.to_csv(predict_topic_csv_path, encoding='utf-8-sig', index=None)
    data = pyLDAvis.sklearn.prepare(lda, tf_idf, tf_idf_vectorizer)
    # data = pyLDAvis.sklearn.prepare(lda, tf_idf, tf_idf_vectorizer, mds='mmds')
    pyLDAvis.save_html(data, html_path)
    return gen_labels

def CV(ldainput, topic_num, iternum, top_words_csv_path, predict_topic_csv_path, html_path):
    count_vectorizer = CountVectorizer()
    # count_vectorizer = CountVectorizer(analyzer='char')
    cv = count_vectorizer.fit_transform(ldainput)
    # feature_names = count_vectorizer.get_feature_names_out()
    # matric = cv.toarray()
    # df = pd.DataFrame(matric, columns=feature_names)
    # df.to_csv('词频数据——单一向.csv', encoding='utf-8-sig')
    lda = LatentDirichletAllocation(
        n_components=topic_num, max_iter=iternum,
        learning_method='online',
        learning_offset=50,
        random_state=0)
    lda.fit(cv)
    top_words_df = top_words_data_frame(lda, count_vectorizer, 10)
    top_words_df.to_csv(top_words_csv_path, encoding='utf-8-sig', index=None)
    X = cv.toarray()
    predict_df, gen_labels = predict_to_data_frame(lda, X)
    predict_df.to_csv(predict_topic_csv_path, encoding='utf-8-sig', index=None)
    data = pyLDAvis.sklearn.prepare(lda, cv, count_vectorizer,mds='mmds')
    pyLDAvis.save_html(data, html_path)
    return gen_labels

if __name__ == '__main__':
    filepath = './ch/'#需要遍历的文件夹
    txt_num = 10#文章个数
    para_sum = 200#需要的段落数
    stop_addr = "cn_stopwords.txt"#停词表文件
    ldadata = LDA_DATA(filepath, txt_num, para_sum)
    ldadata.stopwordslst()
    lda_artic, lda_labels = ldadata.data_load_fenci()
    # lda_artic, lda_labels = ldadata.data_load_words()

    top_words_csv_path = 'top-topic-words.csv'# 输出主题词的文件路径
    predict_topic_csv_path = 'words-distribution.csv'# 输出各文档所属主题的文件路径
    html_path = 'visiual.html'#输出html文件
    topic_num = 10#主题数
    iter_num = 10#迭代次数
    gen_labels = TF_IDF(lda_artic, topic_num, iter_num, top_words_csv_path, predict_topic_csv_path, html_path)
    # gen_labels = CV(lda_artic, topic_num, iter_num, top_words_csv_path, predict_topic_csv_path, html_path)

    print('真实标签：', lda_labels)
    print('生成的标签：', gen_labels)
    accuracy = accuracy_score(lda_labels, gen_labels)
    print("分类准确率：", accuracy)