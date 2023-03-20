1.Tohoku-classificationBERT_AE.ipynb  
hugging face trainer (多显卡的时候自动GPU并列训练，可在notebook直接运行，nn.DataParallel？) 

2.Tohoku-classificationBERT_AE-GPUParallel.ipynb  
pytorch + BertForSequenceClassification  
model = nn.DataParallel(model)  
