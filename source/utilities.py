import os
import torch
#import logging
from typing import Tuple
import csv

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)  
    return data
    
def create_dirs():
    directories = ['checkpoints','logs']
    for directory in directories:
        os.makedirs(directory, exist_ok= True)
# LOGGING FORSE DA IMPLEMENTARE DIRETTAMENTE IN MAIN TRAMITE LOGGING LIB
#def set_logging(model, test_dir_name, ep: int)
#    filename = f"logs/model_{test_dir_name}_epoch_{ep}.pth"
#   print(f"Checkpoint saved to {filename}")

                
def save_checkpoint(model, test_dir_name: str, ep: int, val_accuracy=None):
        filename = f"checkpoints/model_{test_dir_name}_epoch_{ep}.pth"
        torch.save({'model_state_dict': model.state_dict(),
                    'epoch': ep, 
                    'val_accuracy': val_accuracy}, filename)
        print(f" >> Checkpoint saved to: {filename}")
    
def log_epoch_stats(test_dir_name: str, epoch: int, train_loss: float, train_accuracy: float, val_loss: float, val_accuracy: float):

    log_filename = f"logs/{test_dir_name}_epoch_log.csv"
    write_header = not os.path.exists(log_filename)

    with open(log_filename, mode='a', newline='') as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])
        writer.writerow([epoch, f"{train_loss:.4f}", f"{train_accuracy:.4f}", f"{val_loss:.4f}", f"{val_accuracy:.4f}"])
        
