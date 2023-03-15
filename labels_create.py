import os
import pickle
import pandas as pd
import unicodedata
from transformers import BertJapaneseTokenizer

def index_func(token,pick_word_tokens):
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
    pick_word_tokens_flatten=flatten(pick_word_tokens)    

    label=[0 for i in range(len(token))]
    label_index=[i for i in range(len(token)) if token[i] in pick_word_tokens_flatten]
    label=[1 if i in label_index  else 0 for i in range(len(label))]
    index=[]
    subindex=[]
    for i in range(len(label)):
        if label[i]==1:
            subindex.append(i)
        if label[i]==0:
            index.append(subindex)
            subindex=[]
    index= list ( filter ( None , index))    

    # remove label_index中の'が' 'の' などのsingle index sublist   e.g.[['筋炎','の','可能性'].['の']]
    remove=['が','の'] 
    index_filter=[]
    for subindex in index:
        if len(subindex)==1 and token[subindex[0]] in remove:
            continue
        else:
            index_filter.append(subindex)
    
    return index_filter

def label_func(index_list, tokens, pick_word):
    label=[0 for i in range(len(tokens))]
    for index in index_list:
            for i, idx in enumerate(index):
                if i == 0:
                    label[idx] = 1
                else:
                    label[idx] = 2      
    return label

def label_func_2(row, tokens, pick_word_index_list):

    df = pd.DataFrame({'tokens':tokens})
    df.tokens = df.tokens.apply(lambda x: unicodedata.normalize("NFKC", x.replace('##', '')))
    tokens_index = []
    for t in df.tokens:
        tokens_index.append((row.find(t), row.find(t)+len(t)))
        row = row.replace(t, ''.join(['@' for i in range(len(t))]), 1)

    df['index'] = tokens_index
    for word_idx in pick_word_index_list:
        df['label'] = df.apply(lambda x: 2 if (x['index'][1]>word_idx[0]) and (x['index'][0]<word_idx[1]) else 0, axis=1)
        df['label'] = df.apply(lambda x: 1 if (x['label']==2) and (x['index'][0]<=word_idx[0]) and (word_idx[0]<x['index'][1]) else x['label'], axis=1)
    return df


#if __name__ == "__main__":
def labels_create(config):
    
    # load data
    with open(os.path.join(config.input_data_path,config.create_labels_input_data), 'rb') as file:
        df=pickle.load(file)
        #df=pd.read_pickle(file)
        
    #tokenizer
    tokenizer = BertJapaneseTokenizer.from_pretrained(config.tokenizer_path) #'cl-tohoku/bert-large-japanese'
    
    df['id']=[i for i in range(len(df))]
    
    # get tokens
    df['tokens'] = df['row'].apply(lambda x: tokenizer.tokenize(x))
    df['pick_word_tokens'] = df['pick_words'].apply(lambda x: [tokenizer.tokenize(p) for p in x])
   
    # create labels
    """
    label_list=[0,1,2]
    0==> "0" indicates the token doesn’t correspond to any entity.
    1==> "B" indicates the beginning of an entity.
    2==> "I" indicates a token is contained inside the same entity    
    """
    df['index'] = df.apply(lambda x:  index_func(x['tokens'],x['pick_word_tokens']) if x['pick_word_tokens']!=['記載なし'] else 0 , axis=1)
    df['labels'] = df.apply(lambda x: label_func(x['index'],x['tokens'],x['pick_words']), axis=1)
    
    
    """
    問題点:
    row token != pick_work_token
    G3薬剤性大腸炎発現==>['G', '3', '薬剤', '性', '大腸', '炎', '発現', '。']
    大腸炎==>['大', '腸', '##炎']
    '大腸', '炎' != '大', '腸', '##炎'    
    
    この問題がある行を抽出
    """
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
    index_list=[]
    for index,row in df.iterrows():
        #print(row['row'])
        #if row['pick_word']!= ['記載なし']:
            pick_word_token=row['pick_word_tokens']
            token=row['tokens']
            pick_word_token_flatten=flatten(pick_word_token) 
            for i in pick_word_token_flatten:
                u=set([i]) & set(token)
                if len(u) == 0:
                    #print(i)
                    #print(index)
                    index_list.append(row['id'])
    index_list=list(set(index_list))    
    
    df_ = df[df['id'].isin(index_list)]
    df= df[~df['id'].isin(index_list)]
    
    """
    この問題を解決
    """
    df_t = df_
    df_t['pick_word_index_list'] = df_.apply(lambda x: [(x['row'].find(w), x['row'].find(w)+len(w)) for w in x['pick_words']], axis=1)
    df_t['df_label'] = df_t.apply(lambda x: label_func_2(x['row'], x['tokens'], x['pick_word_index_list']), axis=1)
    df_t['labels'] = df_t['df_label'].apply(lambda x: x['label'].to_list())    
    
    #元のデータにマージする
    df=pd.concat([df, df_t],axis=0)    
    
    # bert input
    MAX_LENGTH =128  #512

    # remove length>MAX_LENGTH
    df['input_lengths']=df['tokens'].apply(lambda x:len(x))
    df=df[df['input_lengths']<MAX_LENGTH-1]

    # input token ids
    ## tokens add '[CLS]' [SEP]' 
    df['tokens']=df['tokens'].apply(lambda x: ['[CLS]']+x+['[SEP]'])
    df['input_ids']=df['tokens'].apply(lambda x:tokenizer.convert_tokens_to_ids(x))

    # labels
    # [CLS],[SEP] ==> -100 
    df['labels']=df['labels'].apply(lambda x: [-100]+x+[-100])

    # attention_mask
    df['attention_mask']=df['tokens'].apply(lambda x:[1 for i in range(len(x))])   
    #print(df.columns)
    # save
    with open(os.path.join(config.input_data_path,config.create_labels_output_data), 'wb') as file:
        pickle.dump(df, file)
