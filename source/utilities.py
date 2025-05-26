import os
import torch
import logging
from typing import Tuple

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)  
    return data
    
def create_dirs():
    actual_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(actual_dir)

    # create directories 
    checkpoints_dir = os.path.join(project_dir, 'checkpoints')
    logs_dir = os.path.join(project_dir, 'logs')
    os.makedirs(checkpoints_dir, exist_ok= True)
    os.makedirs(logs_dir, exist_ok= True)

    print(f"Logs and checkpoints directories created.")

# LOGGING FORSE DA IMPLEMENTARE DIRETTAMENTE IN MAIN TRAMITE LOGGING LIB
#def set_logging(model, test_dir_name, ep: int)
#   filename = f"logs/model_{test_dir_name}_epoch_{ep}.pth"
#   print(f"Checkpoint saved to {filename}")

                
def save_checkpoint(model, test_dir_name: str, ep: int):
            filename = f"checkpoints/model_{test_dir_name}_epoch_{ep}.pth"
            torch.save(model.state_dict(), filename)
            print(f"Checkpoint saved to {filename}")
