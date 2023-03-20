import os
import json
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from TorchCRF import CRF
from transformers import BertModel
from tqdm import tqdm
from utils import split_data

# Dataset
def get_encoding(df):
    encoding={}
    encoding['input_ids']=df['input_ids'].tolist()
    #encoding['token_type_ids']=df['token_type_ids'].tolist()
    encoding['attention_mask']=df['attention_mask'].tolist() 
    return encoding

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, config):
        self.encodings = encodings
        self.labels = labels
        self.config = config
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(self.config.device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(self.config.device)
        return item

    def __len__(self):
        return len(self.labels)

def collate_func(batch):
    input_ids, attention_mask,labels = [], [], []
    
    for mini_batch in batch:
        input_ids.append(mini_batch['input_ids'])
        attention_mask.append(mini_batch['attention_mask'])
        labels.append(mini_batch['labels'])
        
    item={}
    item['input_ids'] =pad_sequence(input_ids, batch_first=True, padding_value=0)
    item['attention_mask'] = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    item['labels'] =pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return item

#BERT_CRF model
class BERT_CRF_NER(torch.nn.Module):
    
    def __init__(self,config):
        super(BERT_CRF_NER, self).__init__()
        self.num_labels = config.label_num
        self.device=config.device
        self.bert =BertModel.from_pretrained(config.bert_model_path, num_labels=config.label_num).to(config.device)
        self.dropout = torch.nn.Dropout(config.last_hidden_state_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.label_num)
        self.crf = CRF(self.num_labels)
    
    def forward(self,input_ids, attention_mask ,token_type_ids=None, label=None):
        
        # get output from bert and the output will be fed into crf layer as emission
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]    #[batch_size, sequence_size, hidden_size]
        sequence_output = self.dropout(sequence_output)        
        logits = self.classifier(sequence_output) #(batch_size, sequence_size, num_labels)
        outputs = (logits,)
        
        # crf layer
        if label is not None:
            # crf input label & label_mask [CLS],[SEP] [PAD]==>0 others==>1
            zero = torch.zeros_like(label)
            one=torch.ones_like(label)
            label_ = torch.where(label ==-100, zero, label) # (batch_size. sequence_size)
            mask_ = torch.where(label >=0, one, label)
            mask_ = torch.where(label ==-100, zero, mask_).to(device=self.device, dtype=torch.uint8)# (batch_size. sequence_size)
            
            # get crf loss
            loss=self.crf.forward(logits[:,1:,:], label_[:,1:], mask_[:,1:]).sum(dim=0)* (-1)
            #loss=crf.forward(logits, label_, mask_).sum(dim=0)* (-1)
            loss/=logits.size()[0]
            outputs = (loss,) + outputs
        
        return outputs # contain: (loss), scores

# Train    
def train_epoch(model, config, epoch_idx, training_dataloader, optimizer, scheduler):
    
    # model ==> train mode
    model.train()
    train_losses = 0
    for batch_id, batch in enumerate(tqdm(training_dataloader)):    

        input_ids, attention_mask, label=batch['input_ids'], batch['attention_mask'], batch['labels']
              
        # forward pass
        loss=model(input_ids,attention_mask,label=label)[0]
        if isinstance(model,torch.nn.DataParallel):
            loss=loss.mean()
        train_losses += loss.item()      
        
        # grad update      
        model.zero_grad()
        loss.backward()
        
        # clip_grad
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
        
        # grad calculate
        optimizer.step()
        scheduler.step()
        
    train_loss = float(train_losses) / len(training_dataloader)
    logging.info(f"Epoch: {epoch_idx}, train loss: {train_loss}")
    print(f"Epoch: {epoch_idx}, train loss: {train_loss}")    
    
    
    
def train_valid(model,training_dataloader,validation_dataloader,optimizer,scheduler,config,cross_id):
    
    best_val_f1 =0
    
    true_labels=[]
    pred_labels=[]
    
    for epoch_idx in range(1, config.epoch_num + 1):    
        print("epoch {}/{}".format(epoch_idx, config.epoch_num))
        
        # Train
        train_epoch(model, config, epoch_idx, training_dataloader, optimizer,scheduler)

        # Validate
        print('Validate')
        logging.info("Start validation")
        val_metrics,true_labels_,pred_labels_,true_labels_flatten,pred_labels_flatten=validate(model,validation_dataloader,config)
        val_f1 = val_metrics['F1_B']+val_metrics['F1_I']+val_metrics['F1_B+I']
        
        # save model with highest F1 score
        if val_f1>best_val_f1:
            
            true_labels=true_labels_
            pred_labels=pred_labels_
            
            best_val_f1=val_f1
            
            # print best  classification_report
            print(classification_report(true_labels_flatten, pred_labels_flatten, target_names=['O', 'B', 'I']))
            true_labels_flatten_=[1 if true_labels_flatten[i]==2 else true_labels_flatten[i] for i in range(len(true_labels_flatten))]
            pred_labels_flatten_=[1 if pred_labels_flatten[i]==2 else pred_labels_flatten[i] for i in range(len(pred_labels_flatten))]            
            print(classification_report(true_labels_flatten_, pred_labels_flatten_, target_names=['O', 'B+I']))
            
            # save model
            if isinstance(model,torch.nn.DataParallel):
                torch.save(model.module, os.path.join(config.output_model_path,
                                                    "model_{}_{}.pt".format(config.run_id,cross_id))) 
            else:
                torch.save(model, os.path.join(config.output_model_path,
                                                    "model_{}_{}.pt".format(config.run_id,cross_id)))     
            logging.info("--------Save best model--------")            
            print("--------Save best model--------")
            
    logging.info("Training Finished")    
    print("Training Finished")

    return true_labels,pred_labels   
    
    
def validate(model,validation_dataloader,config):
    
    # model ==> eval mode
    model.eval()
    val_losses = 0
    
    true_labels=[]
    pred_labels=[]
    
    for batch_id, batch in enumerate(tqdm(validation_dataloader)):    
        
        input_ids, attention_mask, label=batch['input_ids'], batch['attention_mask'], batch['labels']
        
        with torch.no_grad():
            loss=model(input_ids,attention_mask,label=label)[0]
            if isinstance(model,torch.nn.DataParallel):
                loss=loss.mean()
            val_losses += loss.item()
            # output emission matrix 
            batch_output = model(input_ids,attention_mask)[0]
            # label_mask for crf ,[CLS],[SEP] [PAD]==>0 others==>1
            zero = torch.zeros_like(label)
            one=torch.ones_like(label)
            mask_ = torch.where(label >=0, one, label)
            mask_ = torch.where(label ==-100, zero, mask_).to(device=config.device, dtype=torch.uint8)
            
            # output predict-label sequence
            if isinstance(model,torch.nn.DataParallel):
                pred_label=model.module.crf.viterbi_decode(batch_output[:,1:,:], mask=mask_[:,1:])
            else:
                pred_label=model.crf.viterbi_decode(batch_output[:,1:,:], mask=mask_[:,1:])
            # remove ground_ture label's[CLS][SEP][PAD]=-100 ==>get label's length that is as same as predict-label's length
            true_label=[]
            label=label.cpu().numpy()
            for i in range(len(label)):
                #true_label.append(label[i][~np.isin(label[i], -100)].to('cpu').numpy().tolist())
                true_label.append(label[i][~np.isin(label[i], -100)].tolist())
                
            true_labels.extend(true_label)
            pred_labels.extend(pred_label)         
            
    assert len(pred_labels) == len(true_labels)        
    
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
    true_labels_flatten=flatten(true_labels)
    pred_labels_flatten=flatten(pred_labels)
    
    metrics={}
    f1_B,f1_I,f1_B_I = compute_metrics(true_labels_flatten,pred_labels_flatten)
    metrics['F1_B'],metrics['F1_I'],metrics['F1_B+I']=f1_B,f1_I,f1_B_I
    metrics['loss']=float(val_losses) / len(validation_dataloader)
    
    return metrics,true_labels,pred_labels,true_labels_flatten,pred_labels_flatten    
    
def compute_metrics(true_labels_flatten,pred_labels_flatten):
    
    #print(classification_report(true_labels_flatten, pred_labels_flatten, target_names=['O', 'B', 'I']))
    logging.info(classification_report(true_labels_flatten, pred_labels_flatten, target_names=['O', 'B', 'I']))
    
    #plot confusion matrix
    #c=confusion_matrix(true_labels_flatten,pred_labels_flatten)
    #df_c=pd.DataFrame(c,index=[i for i in ['O','B','I']],columns=[i for i in ['O','B','I']])
    #pp_matrix(df_c,figsize=(4,4),cmap='PuRd')
    #f1
    f1_B=precision_recall_fscore_support(true_labels_flatten, pred_labels_flatten)[2][1]
    f1_I=precision_recall_fscore_support(true_labels_flatten, pred_labels_flatten)[2][2]
    
    # If B and I are considered the same
    true_labels_flatten_=[1 if true_labels_flatten[i]==2 else true_labels_flatten[i] for i in range(len(true_labels_flatten))]
    pred_labels_flatten_=[1 if pred_labels_flatten[i]==2 else pred_labels_flatten[i] for i in range(len(pred_labels_flatten))]
    
    #print(classification_report(true_labels_flatten_, pred_labels_flatten_, target_names=['O', 'B+I']))
    logging.info(classification_report(true_labels_flatten_, pred_labels_flatten_, target_names=['O', 'B+I']))
    
    #plot confusion matrix
    #c_=confusion_matrix(true_labels_flatten_,pred_labels_flatten_)
    #df_c_=pd.DataFrame(c_,index=[i for i in ['O','B+I']],columns=[i for i in ['O','B+I']])
    #pp_matrix(df_c_,figsize=(4,4),cmap='PuRd')#    
    #f1
    f1_B_I=precision_recall_fscore_support(true_labels_flatten_, pred_labels_flatten_)[2][1]
    
    return f1_B,f1_I,f1_B_I    


def train(config,cross_validation=False):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    #Cross vaildation
    if cross_validation==True:

        # Split Data
        split_data(config,cross_validation=cross_validation)

        #Cross vaildation
        cross=[1,2,3,4,5]
        cross_true_labels={}
        cross_pred_labels={}

        print("--------Start Training--------")
        for cross_id in cross:
            print('cross :',cross_id)

            # Model
            BERT_CRFModel = BERT_CRF_NER(config).to(config.device)

            # Multi-GPU parallel
            if config.device_count>1:
                BERT_CRFModel = nn.DataParallel(BERT_CRFModel) 

            # Prepare optimizer    full_fine_tuning
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            if isinstance(BERT_CRFModel,torch.nn.DataParallel):
                # model ==> model.module
                bert_optimizer = list(BERT_CRFModel.module.bert.named_parameters())
                classifier_optimizer = list(BERT_CRFModel.module.classifier.named_parameters())                   
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
                     'weight_decay': config.weight_decay},
                    {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
                     'weight_decay': 0.0},
                    {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
                     'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
                    {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
                     'lr': config.learning_rate * 5, 'weight_decay': 0.0},
                    {'params': BERT_CRFModel.module.crf.parameters(), 'lr': config.learning_rate * 5}
                ]      
            else:
                bert_optimizer = list(BERT_CRFModel.bert.named_parameters())
                classifier_optimizer = list(BERT_CRFModel.classifier.named_parameters())
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
                     'weight_decay': config.weight_decay},
                    {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
                     'weight_decay': 0.0},
                    {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
                     'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
                    {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
                     'lr': config.learning_rate * 5, 'weight_decay': 0.0},
                    {'params': BERT_CRFModel.crf.parameters(), 'lr': config.learning_rate * 5}
                ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)       

            # Load data
            with open(os.path.join(config.input_traindata_path,"train_split{}.pkl".format(cross_id)), 'rb') as file:
                train_df=pickle.load(file)
            with open(os.path.join(config.input_traindata_path,"test_split{}.pkl".format(cross_id)), 'rb') as file:
                test_df=pickle.load(file)
            print("traindata:",len(train_df),"testdata:",len(test_df))

            train_encodings, valid_encodings=get_encoding(train_df), get_encoding(test_df)
            train_labels, valid_labels = train_df['labels'].tolist(),test_df['labels'].tolist()

            training_dataset = Dataset(train_encodings, train_labels, config)
            validation_dataset = Dataset(valid_encodings, valid_labels, config)    

            training_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, num_workers=0, shuffle=False,collate_fn=collate_func,generator=torch.Generator(device = config.device))
            validation_dataloader = DataLoader(validation_dataset, batch_size=config.batch_size, num_workers=0, shuffle=False,collate_fn=collate_func,generator=torch.Generator(device = config.device)) 

            # Dynamic learning rate adjustment
            train_steps_per_epoch = len(training_dataset) // config.batch_size
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch, 
                                                    num_training_steps=config.epoch_num * train_steps_per_epoch)

            #Train the model
            logging.info("--------Start Training--------")
            logging.info(f"---------------------cross_id: {cross_id}------------------------")

            true_labels,pred_labels=train_valid(BERT_CRFModel,training_dataloader,validation_dataloader,optimizer,scheduler,config,cross_id) 
            
            # save labels in json
            cross_true_labels['cross_true_labels_{}'.format(cross_id)]=true_labels
            cross_pred_labels['cross_pred_labels_{}'.format(cross_id)]=pred_labels

        cross_result_labels={}
        cross_result_labels['cross_true_labels']=cross_true_labels
        cross_result_labels['cross_pred_labels']=cross_pred_labels

        """
        cross_result_labels=
        {
        cross_pred_labels:{
        cross_pred_labels_1:{}
        cross_pred_labels_2:{}
        cross_pred_labels_3:{}
        cross_pred_labels_4:{}
        cross_pred_labels_5:{}
        }
        cross_true_labels:{
        cross_true_labels_1:{}
        cross_true_labels_2:{}
        cross_true_labels_3:{}
        cross_true_labels_4:{}
        cross_true_labels_5:{}    
        }}

        """
        json_file = open(os.path.join(config.output_path,'cross_result_labels.json'), mode="w")
        json.dump(cross_result_labels, json_file, indent=1, ensure_ascii=False)
        json_file.close()      
        print("Cross_validation_result_labels saving Finished")
    
    
    # NO cross vaildation        
    if cross_validation==False:    
        cross_id=0
        # Split Data
        split_data(config,cross_validation=cross_validation)    

        # Model
        BERT_CRFModel = BERT_CRF_NER(config).to(config.device)

        # Multi-GPU parallel
        if config.device_count>1:
            BERT_CRFModel = nn.DataParallel(BERT_CRFModel) 

        # Prepare optimizer    full_fine_tuning
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        if isinstance(BERT_CRFModel,torch.nn.DataParallel):
            # model ==> model.module
            bert_optimizer = list(BERT_CRFModel.module.bert.named_parameters())
            classifier_optimizer = list(BERT_CRFModel.module.classifier.named_parameters())                   
            optimizer_grouped_parameters = [
                {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': config.weight_decay},
                {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0},
                {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
                 'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
                {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
                 'lr': config.learning_rate * 5, 'weight_decay': 0.0},
                {'params': BERT_CRFModel.module.crf.parameters(), 'lr': config.learning_rate * 5}
            ]      
        else:
            bert_optimizer = list(BERT_CRFModel.bert.named_parameters())
            classifier_optimizer = list(BERT_CRFModel.classifier.named_parameters())
            optimizer_grouped_parameters = [
                {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': config.weight_decay},
                {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0},
                {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
                 'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
                {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
                 'lr': config.learning_rate * 5, 'weight_decay': 0.0},
                {'params': BERT_CRFModel.crf.parameters(), 'lr': config.learning_rate * 5}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)       

        # Load data
        with open(os.path.join(config.input_traindata_path,"train_df.pkl"), 'rb') as file:
            train_df=pickle.load(file)
        with open(os.path.join(config.input_traindata_path,"test_df.pkl"), 'rb') as file:
            test_df=pickle.load(file)
        #print("traindata:",len(train_df),"testdata:",len(test_df))

        train_encodings, valid_encodings=get_encoding(train_df), get_encoding(test_df)
        train_labels, valid_labels = train_df['labels'].tolist(),test_df['labels'].tolist()

        training_dataset = Dataset(train_encodings, train_labels, config)
        validation_dataset = Dataset(valid_encodings, valid_labels, config)    

        training_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, num_workers=0, shuffle=False,collate_fn=collate_func,generator=torch.Generator(device = config.device))
        validation_dataloader = DataLoader(validation_dataset, batch_size=config.batch_size, num_workers=0, shuffle=False,collate_fn=collate_func,generator=torch.Generator(device = config.device)) 

        # Dynamic learning rate adjustment
        train_steps_per_epoch = len(training_dataset) // config.batch_size
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch, 
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)

        #Train the model
        logging.info("--------Start Training--------")

        true_labels,pred_labels=train_valid(BERT_CRFModel,training_dataloader,validation_dataloader,optimizer,scheduler,config,cross_id) 

        # save labels in json
        result_labels={}
        result_labels['true_labels']=true_labels
        result_labels['pred_labels']=pred_labels
        """
        result_labels=
        {
        pred_labels:{}
        true_labels:{}   
        }
        """
        json_file = open(os.path.join(config.output_path,'result_labels.json'), mode="w")
        json.dump(result_labels, json_file, indent=1, ensure_ascii=False)
        json_file.close()      
        print("validation_result_labels saving Finished")
