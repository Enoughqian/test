
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
import jieba
import json
import re


# In[2]:


train_data_path = '.../data/atec_nlp_sim_train.csv'  # 训练数据
train_add_data_path = '../data/atec_nlp_sim_train_add.csv'  # 添加训练数据
stop_words_path = '../data/stop_words.txt'  # 停用词路径
tokenize_dict_path = '../data/dict_all.txt'  # jieba分词新自定义字典
spelling_corrections_path = '../data/spelling_corrections.json'


# In[3]:


train_data_df = pd.read_csv(train_data_path, sep='\t', header=None,names=["index", "s1", "s2", "label"])
train_add_data_df = pd.read_csv(train_add_data_path, sep='\t', header=None, names=["index", "s1", "s2", "label"])
train_all = pd.concat([train_data_df, train_add_data_df])


# In[4]:


train_all.reset_index(drop=True, inplace=True)


# In[5]:


train_all.head()


# ### 分词及处理

# In[6]:


jieba.load_userdict(tokenize_dict_path)


# In[7]:


# 停用词表
stopwords = [line.strip() for line in open(stop_words_path, 'r', encoding='utf-8').readlines()]


# In[8]:


# 拼错词替换表
with open(spelling_corrections_path,"r",encoding="utf-8") as file:
    spelling_corrections = json.load(file)


# In[9]:


def transform_other_word(str_text,reg_dict):
    """
    替换词
    :param str_text:待替换的句子
    :param reg_dict:替换词字典
    :return:
    """
    for token_str,replac_str in reg_dict.items():
        str_text = str_text.replace(token_str, replac_str)
    return str_text


# In[10]:


def seg_sentence(sentence, stop_words):
    """
    对句子进行分词
    :param sentence:句子，停用词
    """
    sentence_seged = jieba.cut(sentence.strip())
    word_list = [i for i in sentence_seged if i not in stop_words and i != ' ']
    return word_list


# In[11]:


def preprocessing_word(s1_train, s2_train, stopwords, spelling_corrections):

    # 去除句子中的脱敏数字***，替换成一
    re_object = re.compile(r'\*+')

    s1_all = []
    s2_all = []
    all = []

    for s1_,s2_ in zip(s1_train, s2_train):
        s1 = re_object.sub(u"十一", s1_)
        s2 = re_object.sub(u"十一", s2_)
        spell_corr_s1 = transform_other_word(s1, spelling_corrections)
        spell_corr_s2 = transform_other_word(s2, spelling_corrections)

        # 分词
        seg_s1 = seg_sentence(spell_corr_s1, stopwords)
        seg_s2 = seg_sentence(spell_corr_s2, stopwords)

        all.extend(seg_s1)
        all.extend(seg_s2)
        s1_all.append(seg_s1)
        s2_all.append(seg_s2)
    source_list = []
    # source_list = list(set(all))
    source_list.append('<UNK>')
    source_list.append('<PAD>')
    source_list.extend(list(set(all)))
    word2id = {}
    id2word = {}
    for index, char in enumerate(source_list):
        word2id[char] = index
        id2word[index] = char

    return s1_all, s2_all, word2id, id2word


# In[12]:


s1_train = train_all["s1"].tolist()
s2_train = train_all["s2"].tolist()
y_train = train_all["label"].tolist()


# In[13]:


s1_word_all, s2_word_all, word2id, id2word = preprocessing_word(s1_train, s2_train, stopwords, spelling_corrections)


# In[14]:


def make_word2id(data, word2id):
    data2id = []
    for word_list in data:
        id_list = [word2id.get(i) if word2id.get(i) is not None else word2id.get('<PAD>') for i in word_list]
        data2id.append(id_list)
    return data2id


# In[15]:


def all_data_set(s1_all, s2_all, word2id, y_train, max_l=15):
    pad = word2id['<PAD>']
    all_data = []
    s1_data_id = make_word2id(s1_all, word2id)
    s2_data_id = make_word2id(s2_all, word2id)
    s1_all_new = []
    s2_all_new = []
    y = []
    for i in range(len(s1_data_id)):
        if len(s1_data_id[i]) > max_l:
            s1_set = s1_data_id[i][:max_l]
        else:
            s1_set = np.concatenate((s1_data_id[i], np.tile(pad, max_l - len(s1_data_id[i]))), axis=0)
        if len(s2_data_id[i]) > max_l:
            s2_set = s2_data_id[i][:max_l]
        else:
            s2_set = np.concatenate((s2_data_id[i], np.tile(pad, max_l - len(s2_data_id[i]))), axis=0)
        y_set = [1,0] if y_train[i] == 0 else [0,1]
        s1_all_new.append(s1_set)
        s2_all_new.append(s2_set)
        y.append(y_set)
    return s1_all_new, s2_all_new, y


# In[16]:


s1_word_id_all, s2_word_id_all, y_set = all_data_set(s1_word_all, s2_word_all, word2id, y_train, max_l=15)


# In[17]:


train_all["s1_word_all"] = s1_word_all


# In[18]:


train_all["s2_word_all"] = s2_word_all


# In[19]:


train_all["s1_word_id_all"] = s1_word_id_all


# In[20]:


train_all["s2_word_id_all"] = s2_word_id_all


# In[21]:


train_all["y_set"] = y_set


# In[22]:


train_all.tail()


# In[23]:


# 将数据存到一个大列表里面，格式是[[s1,s2,y],[s1,s2,y],[s1,s2,y].......]
all_data = []
for i in range(len(s1_word_id_all)):
    all_data.append([s1_word_id_all[i],s2_word_id_all[i],y_set[i]])


# In[24]:


# 将数据存入pickle中
with open("../processed_data/word_data.pk", 'wb') as f1:
    pickle.dump((all_data,word2id,id2word), f1)

