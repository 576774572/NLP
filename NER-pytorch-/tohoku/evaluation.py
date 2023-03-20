import os
import json
import pickle
import jaconv
import logging
import pandas as pd

# pred_pick
# pred_pick：モデルによって検出された有害事象
# label_words：pick_wordsによってannotationしたデータ、プログラミングによってラベル付けたので、ノイズが存在している、元のpick_wordsと間違うところがある
def get_result_df(x,t):
    if t=='preds':
        df=pd.DataFrame({
        'token': [s.replace('##', '') for s in x['tokens']],
        'preds':[1 if x['preds'][i]==2 else x['preds'][i] for i in range(len(x['preds']))],
        'th': [i== 1 or i==2 for i in x['preds']]
    })
    if t=='labels':
        df=pd.DataFrame({
        'token': [s.replace('##', '') for s in x['tokens']],
        'labels':[1 if x['labels'][i]==2 else x['labels'][i] for i in range(len(x['labels']))],
        'th': [i== 1 or i==2 for i in x['labels']]
    })
    return df

def join_words(df):
    df['group'] = df['th'].diff(1).cumsum().fillna(0)
    df_g = df[df['th']].groupby('group').agg({
        'token':lambda x: ''.join(x.to_list())
        })
    df_g = df_g.drop_duplicates('token')
    elminated_tokens = ['。', '、', '(', ')', 'なし', 'いる', ':', '-']
    df_g = df_g[~df_g['token'].isin(elminated_tokens)]
    return df_g

def pred_pick(result_df):

    # labels連続するトークンを結合する
    result_df['df']=result_df.apply(lambda x:get_result_df(x,'labels'),axis=1 )
    result_df['df_label_words'] = result_df['df'].apply(lambda x: join_words(x))
    result_df['label_words'] = result_df['df_label_words'].apply(lambda x: x['token'].to_list())
    result_df['label_words'] = result_df['label_words'].apply(lambda x: x if len(x)>0 else ['検出なし'])
    #半角・全角ローマ字の相互変換 半角==>全角
    result_df['label_words'] = result_df['label_words'].apply(lambda x: [(jaconv.h2z(i, kana=True, digit=True, ascii=True)) for i in x])
    
    # preds連続するトークンを結合する
    result_df['df']=result_df.apply(lambda x:get_result_df(x,'preds'),axis=1 )
    result_df['df_pred_pick'] = result_df['df'].apply(lambda x: join_words(x))
    result_df['pred_pick'] = result_df['df_pred_pick'].apply(lambda x: x['token'].to_list())
    result_df['pred_pick'] = result_df['pred_pick'].apply(lambda x: x if len(x)>0 else ['検出なし'])
    #半角・全角ローマ字の相互変換 半角==>全角
    result_df['pred_pick'] = result_df['pred_pick'].apply(lambda x: [(jaconv.h2z(i, kana=True, digit=True, ascii=True)) for i in x])
        
    result_df=result_df.drop(columns=['df','df_label_words','df_pred_pick'])
    return result_df

# result_pick
# モデルが検出した有害事象pred_pickの中で、正解データ以外のAEが存在しているので、正解データ以外のAEを除外した
def result_pick(result_df):
    # pick_wordが含まれるor含んでいる場合True
    ##半角・全角ローマ字の相互変換 全角==>半角
    #result_df['pick_words'] = result_df['pick_words'].apply(lambda x: [(jaconv.z2h(i, kana=False, digit=True, ascii=True)) for i in x])
    #半角・全角の相互変換 半角==>全角
    result_df['pick_words'] = result_df['pick_words'].apply(lambda x: [(jaconv.h2z(i, kana=True, digit=True, ascii=True)) for i in x])

    result_df['result_pick'] = result_df.apply(lambda x: [], axis=1)
    for i in range(100):
        result_df['result_pick'] = result_df.apply(
            lambda x: [w for w in x['pred_pick'] if (w in x['pick_words'][i]) or (x['pick_words'][i] in w)] + x['result_pick'] if len(x['pick_words'])> i else x['result_pick'], axis=1)
    return result_df

# miss_pick
# 被っている場合を考えて、検出されないAE
def miss_pick_filter_fun(result_pick,missed_pick):
    _=[] 
    for i in result_pick:
        for j in missed_pick:
            if j in i or i in j:
                _.append(j)    
    return _

def miss_pick(result_df,filter_fun):
    result_df['missed_pick']=result_df.apply(lambda x:[ i for i in x['pick_words'] if i not in x['result_pick'] or i not in x['pred_pick']]  ,axis=1)
    result_df['missed_pick']=result_df.apply(lambda x:[i for i in x['missed_pick'] if i not in filter_fun(x['result_pick'],x['missed_pick'])]  ,axis=1)
    return result_df

# extra_pick
# 正解データpick_words以外の検出したAEを抽出した(extra_pick ＝ pred_pick ― result_pick)
def extra_pick(result_df):
    result_df['extra_pick']=result_df.apply(lambda x:[ i for i in x['pred_pick'] if i not in x['result_pick'] and i!='検出なし' and len(i)!=1 and i!='[UNK]'] ,axis=1)
    return result_df


# 集計
def process_test_df(config,cross_validation=False):
    
    #Cross vaildation
    if cross_validation==True:
        # Load cross_validation_result_labels saved in json
        with open(os.path.join(config.output_path,'cross_result_labels.json'), 'r') as f:
            cross_result_labels = json.load(f)

        test_df=pd.DataFrame({})
        cross=[1,2,3,4,5]

        for cross_id in cross:
            print('cross :', cross_id)
            with open(os.path.join(config.input_traindata_path,"test_split{}.pkl".format(cross_id)), 'rb') as file:
                test_result_df=pickle.load(file)

            test_result_df['tokens']=test_result_df['tokens'].apply(lambda x:[i for i in x if (i!='[CLS]'and i!='[SEP]' and i!='[PAD]')==True ])

            true_labels=cross_result_labels['cross_true_labels']['cross_true_labels_{}'.format(cross_id)]
            pred_labels=cross_result_labels['cross_pred_labels']['cross_pred_labels_{}'.format(cross_id)]

            test_result_df['labels']=true_labels
            test_result_df['preds']=pred_labels

            for i,row in test_result_df.iterrows():
                assert len(row['tokens'])==len(row['labels'])==len(row['preds'])      

            # pred_pick   
            test_result_df=pred_pick(test_result_df)   
            print('cross_', cross_id,' pred_pick complete')

            # result_pick   
            test_result_df=result_pick(test_result_df)   
            print('cross_', cross_id,' result_pick complete')

            # miss_pick
            test_result_df=miss_pick(test_result_df,miss_pick_filter_fun) 
            print('cross_', cross_id,' miss_pick complete')

            # extra_pick
            test_result_df=extra_pick(test_result_df)  
            print('cross_', cross_id,' extra_pick complete')

            test_df=pd.concat([test_df,test_result_df],ignore_index=False)

        return test_df
    
    # NO cross vaildation        
    if cross_validation==False:
        # Load validation_result_labels saved in json
        with open(os.path.join(config.output_path,'result_labels.json'), 'r') as f:
            result_labels = json.load(f)
        # Load test df    
        with open(os.path.join(config.input_traindata_path,"test_df.pkl"), 'rb') as file:
            test_result_df=pickle.load(file)
        test_result_df['tokens']=test_result_df['tokens'].apply(lambda x:[i for i in x if (i!='[CLS]'and i!='[SEP]' and i!='[PAD]')==True ])

        true_labels=result_labels['true_labels']
        pred_labels=result_labels['pred_labels']
        
        test_result_df['labels']=true_labels
        test_result_df['preds']=pred_labels   
        
        for i,row in test_result_df.iterrows():
            assert len(row['tokens'])==len(row['labels'])==len(row['preds'])             
        # pred_pick   
        test_result_df=pred_pick(test_result_df)   
        print('pred_pick complete')

        # result_pick   
        test_result_df=result_pick(test_result_df)   
        print('result_pick complete')

        # miss_pick
        test_result_df=miss_pick(test_result_df,miss_pick_filter_fun) 
        print('miss_pick complete')

        # extra_pick
        test_result_df=extra_pick(test_result_df)  
        print('extra_pick complete')

        return test_result_df        
            
            
            
def evaluation(config,cross_validation=False):
    
    logging.info("--------Start Evaluation--------")
    print("--------Start Evaluation--------")
    
    test_result_df=process_test_df(config,cross_validation=cross_validation)
    
    test_result_df['result'] = test_result_df.apply(lambda x: True if len(x['result_pick']) >= len(x['pick_words']) else False, axis=1)
    # 検出がある行を抽出
    df_pick_true = test_result_df[test_result_df['pred_pick'].apply(lambda x: x != ['検出なし'])]
    print()
    print('AE recall:', len(test_result_df[test_result_df['result']==True]) / len(test_result_df))
    print('AE precision:', len(df_pick_true[df_pick_true['result']]) / len(df_pick_true))
    print()
    logging.info(f"AE recall: {len(test_result_df[test_result_df['result']==True]) / len(test_result_df)}")
    logging.info(f"AE precision: {len(df_pick_true[df_pick_true['result']]) / len(df_pick_true)}")
    
    # output test_result_df
    with open(os.path.join(config.output_path,'test_result_df.pkl'), 'wb') as file:
        pickle.dump(test_result_df, file)
              
    logging.info("Cross_validation_test_result_df saving Finished")
    print("Cross_validation_test_result_df saving Finished")
        
    logging.info("Evaluation Finished")    
    print("Evaluation Finished")
