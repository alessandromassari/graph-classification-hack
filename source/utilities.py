import os
import logging
from typing import Tuple

def create_dirs():
    directories = ['checkpoints','logs']
    for dir in directories:
        os.makedirs(directory, exist_ok= True)

#def set_logging(test_path: )

                
def save_checkpoint(model, test_dir_name: str, ep: int)
                filename = f"checkpoints/model_{test_dir_name}_epoch_{ep}.pth"
                torch.save(model.state_dict(), filename)
                print(f"Checkpoint saved to {filename})
    

