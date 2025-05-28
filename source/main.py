import os
import argparse
import torch
from torch_geometric.loader import DataLoader
from loadData import GraphDataset
import pandas as pd 
from goto_the_gym import pretraining, train
from utilities import create_dirs, save_checkpoint, add_zeros
from my_model import VGAE_all, gen_node_features
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    total_loss = 0.0
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data, enable_classifier=True)
            class_logits = output[3]
            pred = class_logits.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                classification_loss = F.cross_entropy(class_logits, data.y)
                total_loss += classification_loss.item()
                
    if calculate_accuracy:
        if total > 0 and len(data_loader) > 0:
            accuracy = correct / total 
            avg_loss = total_loss / len(data_loader) 
        else: 
            accuracy = 0.0
            avg_loss = 0.0
        return accuracy, avg_loss, predictions
    return predictions

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    
    # create directories
    create_dirs()
    
    # Hyperparameters for the model (circa a ctrl+c - ctrl+v from competiton GitHub)
    in_dim  = 8             # previous val: 32 node_feature
    hid_dim = 128
    lat_dim = 16            # previous val: 8
    out_classes = 6
    edge_feat_dim = 7
    hid_edge_nn_dim = 64    # previous val: 32
    hid_dim_classifier = 64
    
    pretrain_epoches = 20  # previous val: 10 - 20
    num_epoches      = 80  # previous val: 10 - 20 - 40
    learning_rate = 0.0005 # previous val: 0.001
    bas = 32 #batch size:  # previous val: 64 
    kl_weight_max = 0.01  # previous val: 0.005
    an_ep_kl = 20
    torch.manual_seed(0)

    # early stopping parameters (MAYBE IN THE FUTURE)
    
    # Initialize the model and choose the optimizer
    model = VGAE_all(in_dim, hid_dim, lat_dim, edge_feat_dim, hid_edge_nn_dim, out_classes, hid_dim_classifier).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    node_feat_transf = gen_node_features(feat_dim = in_dim)

    # checkpoints saving threshold on training loss - if have time implement this on acc or validation
    model_loss_min = float('inf')

    # TO BE IMPLEMENTED FOR LOGS AT LEAST 10
    logs_counter = 0
    
    # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path, transform=node_feat_transf)
    test_loader = DataLoader(test_dataset, batch_size=bas, shuffle=False)

     # If train_path is provided then train on it
    if args.train_path:
        print(f">> Starting the train of the model using the following train set: {args.train_path}")
        all_train_dataset = GraphDataset(args.train_path, transform=node_feat_transf) #add_zeros
        all_train_index = list(range(len(all_train_dataset)))
        train_index, val_index = train_test_split(all_train_index,test_size=0.2,random_state=42)
        
        # split the training set in training and validation set
        train_dataset = torch.utils.data.Subset(all_train_dataset, train_index)
        val_dataset = torch.utils.data.Subset(all_train_dataset, val_index)
        
        train_loader = DataLoader(train_dataset, batch_size=bas, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=bas, shuffle=False)
        
        # ----------- pre-training loop ------------ #
        print("\n--- Starting Pre-training of VGAE model ---")
        for epoch in range(pretrain_epoches):
            train_loss = pretraining(model,train_loader, optimizer, device,kl_weight_max, epoch, an_ep_kl)
            train_accuracy, _, _ = evaluate(train_loader, model, device, calculate_accuracy=True)
            print(f"PRETRAINING: Epoch {epoch + 1}/{pretrain_epoches}, Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"--- Pre-training Completed ---")
        
        # -----------   Training loop   ------------ #
        for epoch in range(num_epoches):
            train_loss = train(model,train_loader, optimizer, device, kl_weight_max, epoch,an_ep_kl)
            train_accuracy, _, _ = evaluate(train_loader, model, device, calculate_accuracy=True)

            # validation valutation every 5 epoches
            vaL_loss = 0.0
            val_accuracy = 0.0
            # evaluate on validation set every 5 epoches
            if (epoch+1) % 5 == 0 or epoch == num_epoches - 1:
                val_accuracy, val_loss, _ = evaluate(val_loader,model,device,calculate_accuracy=True)
                print(f"VALIDATION: Epoch {epoch + 1}/{num_epoches}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                test_dir_name = os.path.basename(os.path.dirname(args.test_path))
                save_checkpoint(model, test_dir_name, epoch, val_accuracy) 
                print(f"Checkpoint saved for epoch {epoch + 1} with validation accuracy: {val_accuracy:.4f}")
                
            print(f"TRAINING: Epoch {epoch + 1}/{num_epoches}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            
       
      #  if (epoch < 5) or (train_loss < model_loss_min):
    #       model_loss_min = train_loss
    #       test_dir_name = os.path.basename(os.path.dirname(args.test_path))
     #      save_checkpoint(model, test_dir_name, epoch)

        # SAVE LOGS EACH 10 EPOCHS TO BE COMPLETED 
        #logs/: Log files for each training dataset. Include logs of accuracy and loss recorded every 10 epochs. # usare sempre test_dir_name
        
    # Else if train_path NOT provided 
    if not args.train_path:
        checkpoint_path = args.checkpoint
        # raise an error if not able to find the checkpoint model
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found! {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f">> Loading pre-training model from: {checkpoint_path}")
          
    # Evaluate and save test predictions
    predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
    test_graph_ids = list(range(len(predictions)))  # Generate IDs for graphs

    # Save predictions to CSV
    test_dir_name = os.path.dirname(args.test_path).split(os.sep)[-1]
    #output_csv_path = os.path.join(f"testset_{test_dir_name}.csv")
    output_csv_path = os.path.join('/kaggle/working/', f"testset_{test_dir_name}.csv")
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    output_df.to_csv(output_csv_path, index=False)
    print(f"Test predictions saved to {output_csv_path}")

# arguments plus call to the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a classification model on graph datasets.")
    parser.add_argument("--train_path", type=str, default=None, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint model (e.g. checkpoints/model_B_epoch_10.pth)")
    args = parser.parse_args()
    main(args)
