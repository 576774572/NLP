import argparse
import torch

#Config
def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="Tohoku-BERT_CRF_Training")
    
    parser.add_argument('--bert_model_path', type=str, default='cl-tohoku/bert-large-japanese')
    parser.add_argument('--tokenizer_path', type=str, default='cl-tohoku/bert-large-japanese')
    
    parser.add_argument('--output_path', type=str, default="output")
    parser.add_argument('--input_data_path', type=str, default="input_data")  
    parser.add_argument('--create_labels_input_data', type=str, default="df_row.pkl")
    parser.add_argument('--create_labels_output_data', type=str, default="df_row_token_classification_dataset.pkl")
    parser.add_argument('--input_train_data', type=str, default="df_row_token_classification_dataset.pkl")
    
    parser.add_argument('--device', default = "cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument('--device_count', default = torch.cuda.device_count() if torch.cuda.is_available() else 0, type=int)
    
    parser.add_argument('--label_num',default=3, type=int)        
    parser.add_argument('--batch_size', default=128, type=int)    
    parser.add_argument('--epoch_num', default=1, type=int)    
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--clip_grad', default=5, type=int)  
    parser.add_argument('--last_hidden_state_dropout_prob', default=0.2, type=float)
    parser.add_argument('--hidden_size', default=1024, type=int) 

    config = parser.parse_args(args=[])

    config.device = torch.device(config.device)
    
    return config
