import os
import pickle
import logging
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,train_test_split

# log create
def create_log(config):

    # Log & output path
    now = datetime.datetime.now()
    config.now=now
    run_id = np.random.randint(10000, 99999)
    config.run_id=run_id
    print("run_id",run_id)
    if config.output_path == "":
        config.output_path = os.getcwd()
        
    config.output_path = os.path.join(config.output_path, "run_" + str(now.year) + "." + str(now.month) + "." + str(now.day) + "_" + str(run_id))
    config.input_traindata_path=os.path.join(config.input_data_path, "run_" + str(now.year) + "." + str(now.month) + "." + str(now.day) + "_" + str(run_id))
        
    os.makedirs(os.path.join(config.output_path, "model"))
    config.output_model_path=os.path.join(config.output_path, "model")
    
    logging.basicConfig(
        filename=os.path.join(config.output_path, "log_" + str(run_id) + ".txt"), 
        filemode='w',
        level=logging.INFO, 
        format='[%(levelname)s]%(message)s', force=True)
    logging.info("PARAMETERS:")
    for arg in vars(config):
        logging.info("{0}: {1}".format(arg, getattr(config, arg)))
        print("{0}: {1}".format(arg, getattr(config, arg)))
    logging.info("----------")
    return config
    
# Split Data
def split_data(config,cross_validation=False):
    with open(os.path.join(config.input_data_path, config.input_train_data), 'rb') as file:
            df=pickle.load(file)   
    df=df[:1000]
    #否定語処理
    df=negate_process(df)
    
    #os.makedirs(os.path.join(config.input_data_path,"run_" + str(config.now.year) + "." + str(config.now.month) + "." + str(config.now.day) + "_" + str(config.run_id)))
    os.makedirs(os.path.join(config.input_traindata_path))
    
    # Cross vaildation
    if cross_validation==True:
        
        kf = KFold(n_splits = 5, shuffle = True, random_state = 1)
        id=1

        print("-----------split data--------------")
        for train, test in kf.split(df):
            print("traindata:",len(train),"testdata:",len(test))
            df_train=df.loc[train,:]
            df_test=df.loc[test,:]
            with open(os.path.join(config.input_traindata_path,"train_split{}.pkl".format(id)), 'wb') as file:
                    pickle.dump(df_train, file)    
            with open(os.path.join(config.input_traindata_path,"test_split{}.pkl".format(id)), 'wb') as file:
                    pickle.dump(df_test, file)                
            id+=1    
    
    # NO cross vaildation        
    if cross_validation==False:
        print("-----------split data--------------")
        train_df,test_df = train_test_split(df, test_size=0.2, random_state=1) 
        print("traindata:",len(train_df),"testdata:",len(test_df))
        with open(os.path.join(config.input_traindata_path,"train_df.pkl"), 'wb') as file:
                pickle.dump(train_df, file)    
        with open(os.path.join(config.input_traindata_path,"test_df.pkl"), 'wb') as file:
                pickle.dump(test_df, file)    
                
    
#否定語処理
def negate_ae(token,etracted_ae,negate_list):
    if token[-1]!='。':
        token=token+['。']
    token=['。']+token
    # コンマによって区切り
    split_index=[]
    split_comma=[]
    for i in range(len(token)):
        if token[i]=='。':
            split_index.append(i) 
            split_comma.append(True)
        if token[i]=='、'or token[i]=='（' or token[i]=='）' :
            split_index.append(i) 
            split_comma.append(False)

    #　コンマ間の文字数が5文字以下であれば"　消す
    delete_index=[]
    for i in range(len(split_index)-1):
        if split_index[i+1]-split_index[i]<6 and not split_comma[i+1]:
            delete_index.append(i+1)
    for i in reversed(delete_index):
        del split_index[i]    
    #print(split_index)
    # 分割された文で、否定語が文末[-2:]或は[-3:]にあるか判断　=>否定されたAE (重篤ではない　の場合は省略  '重篤'==>'重', '##篤')
    negate_ae=[]
    for s in range(len(split_index)-1):
        #print(token[split_index[s]+1:split_index[s+1]])
        for negate in negate_list:
            if negate in token[split_index[s+1]-3:split_index[s+1]] and not '##篤' in token[split_index[s]+1:split_index[s+1]]:
                #print('negate in this splited text:',negate)
                for i in range(split_index[s],split_index[s+1]):
                    if token[i] in etracted_ae:
                        negate_ae.append(token[i])
    negate_ae=list(set(negate_ae))
    return negate_ae       

#　被っているケースを処理　e.g. pick_words:貧血 & negate_ae:貧血進行
def filter_fun(pick_words,negate_ae):
    _=[] 
    for i in pick_words:
        for j in negate_ae:
            if j in i or i in j:
                _.append(i)    
    return _

def negate_process(df):
    
    negate_list=['なし','なく','ない','否定','なかっ','ず','消失','無し']
    
    df['negate_ae']=df.apply(lambda x:negate_ae(x['tokens'],x['pick_words'],negate_list)    ,axis=1)
    
    df['negate_ae']=df.apply(lambda x:[i for i in x['pick_words'] if i in filter_fun(x['pick_words'],x['negate_ae'])]  ,axis=1)
    df['negate_result']=df['negate_ae'].apply(lambda x: True if len(x)>0 else False  )
    # 否定されたAEを削除
    df['pick_words']=df.apply(lambda x:[i for i in x['pick_words'] if i not in x['negate_ae']] if x['negate_result']==True else x['pick_words'],axis=1)
    # pick_wordsでAEがない行を削除
    df['remove_row']=df['pick_words'].apply(lambda x: True if len(x)==0 else False)
    df=df.drop(df[df['remove_row']==True].index)
    df=df.drop(columns=['negate_ae','negate_result','remove_row']).reset_index() 
   
    return df
    
