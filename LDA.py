import os
import re
import math
import time
import jieba

def stopwordslst(addr):#获得停词表
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
                if len(words) >= 500:
                    wordsstr = ' '.join(words)
                    artic.append(wordsstr)
                    num += 1
                para = ''
                words.clear()
                if num == para_num:
                    break
    del file
    return artic, num


'''不分词，将每个字独立看待进行运算'''
filepath = './ch/'#需要遍历的文件夹
word_num = 0
wordlst = ''#最终将所有中文放到一个str里
char_sum = 0#记录总字符数，包括停词以及非中文字符
words_sum = 0#记录除了停词表中意外的字符出现个数
txt_num = 16
artic_para = []
for root, path, fil in os.walk(filepath):
    for txt_file in fil:
        words(root+txt_file, 10)
        fenci(root+txt_file, 10)


'''应用分词后的结果'''
words_lst = []
words_sum = 0#记录除了停词表中意外的字符出现个数
for root, path, fil in os.walk(filepath):
    for txt_file in fil:
        # if txt_file != spe:#用来进行特定文章查找，全部遍历请删除这个条件判断
        #     continue
        with open(root+txt_file, "r", encoding='ANSI', errors='ignore') as file:
            fp = file.readlines()
            for line in fp:
                line = line.replace('\n', '')
                line = line.replace(' ', '')
                line = line.replace('　　', '')
                line = line.replace('\t', '')
                wordstr = re.sub(stopstr, '', line)
                for x in jieba.cut(wordstr):
                    words_lst.append(x)
                    words_sum += 1
print('分词后的词组总数为： ', words_sum)