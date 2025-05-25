import os
import argparse
import torch
from torch_geometric.loader import DataLoader
from loadData import GraphDataset
import pandas as pd 
from goto_the_gym import train
from utilities import create_dirs, save_checkpoint, add_zeros
from my_model import VGAE_all

def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
    if calculate_accuracy:
        accuracy = correct / total
        return accuracy, predictions
    return predictions


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # create directories
    create_dirs()
    
    # Hyperparameters for the model (circa a ctrl+c - ctrl+v from competiton GitHub)
    in_dim = 300 
    hid_dim = 64
    lat_dim = 8  #16
    out_classes = 6  
    num_epoches: int = 100 # 100
    
    # Initialize the model and choose the optimizer
    model = VGAE_all(in_dim, hid_dim, lat_dim, out_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # If train_path is provided then train on it 
    if args.train_path:
        print(f">> Starting the train of the model using the following train set: {args.train_path}")
        train_dataset = GraphDataset(args.train_path, transform=add_zeros)
        train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    
        # Training
        for epoch in range(num_epoches):
            train_loss = train(model, train_loader, optimizer, device)
            train_accuracy, _ = evaluate(train_loader, model, device, calculate_accuracy=True)
            print(f"Epoch {epoch + 1}/{num_epoches}, Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")

        
        # Save the checkpoint - call external function
        test_dir_name = os.path.basename(os.path.dirname(args.test_path))
        # INSERIRE IF BEST ACCURACY OR COUNTER < 5
        save_checkpoint(model, test_dir_name, epoch)

        # SAVE LOGS EACH 10 EPOCHS TO BE COMPLETED 
        #logs/: Log files for each training dataset. Include logs of accuracy and loss recorded every 10 epochs.
        # usare sempre test_dir_name
        
    # Else if train_path NOT provided 
    if not args.train_path:
        checkpoint_path = args.checkpoint
        # raise an error if not able to find the checkpoint model
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found! {checkpoint_path}")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f">> Loading pre-training model from: {checkpoint_path}")
          
    # Evaluate and save test predictions
    predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
    test_graph_ids = list(range(len(predictions)))  # Generate IDs for graphs

    # Save predictions to CSV
    test_dir_name = os.path.dirname(args.test_path).split(os.sep)[-1]
    output_csv_path = os.path.join(f"testset_{test_dir_name}.csv")
    output_df = pd.DataFrame({
        "GraphID": test_graph_ids,
        "Pred_class": predictions
    })
    output_df.to_csv(output_csv_path, index=False)
    print(f"Test predictions saved to {output_csv_path}")

# arguments plus call to the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a classification model on graph datasets.")
    parser.add_argument("--train_path", type=str, default=None, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    args = parser.parse_args()
    main(args)
