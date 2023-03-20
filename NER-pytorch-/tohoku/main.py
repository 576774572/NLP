from config import get_config
from labels_create import labels_create
from utils import create_log
from train import train
from evaluation import evaluation

if __name__ == "__main__":
    
    # Config
    config=get_config()

    # Create Dataset Labels
    labels_create(config)  

    # Create log & output_path
    config=create_log(config)
    
    cross_validation=True # True ,False
    
    # Train
    train(config,cross_validation=cross_validation) 
    
    # Evaluation
    evaluation(config,cross_validation=cross_validation) 
