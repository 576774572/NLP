1.Tohoku-classificationBERT_AE.ipynb  
hugging face trainer (多显卡的时候自动GPU并列训练，可在notebook直接运行，nn.DataParallel？) 

2.Tohoku-classificationBERT_AE-GPUParallel.ipynb  
pytorch + BertForSequenceClassification  
model = nn.DataParallel(model)  



1.文章でAEがあるかどうかの分類結果を返す　　
文章：発熱、倦怠感発現。　　
ラベル：1　　
input token: ['cls','発熱','、','倦怠感','発現','。','sep'] ==> output: 1   　

2.文章とAEを入力して、文章のAEかどうかの分類結果を返す

文章：発熱、倦怠感発現。	
AE：発熱       
ラベル：1
input token: ['cls','発熱','、','倦怠感','発現','。','sep','発熱','sep'] ==> output: 1   

文章：発熱、倦怠感発現。	
AE：富士山       
ラベル：0
input token: ['cls','発熱','、','倦怠感','発現','。','sep','富士山','sep'] ==> output: 0   
