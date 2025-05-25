import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from my_model import VGAE_all, kl_loss

def train(model, td_loader, optimizer, device, kl_weight=0.2):
    model.train()
    total_loss = 0.0

    for data in td_loader:
        data = data.to(device)
        # reset the gradients each batch 
        optimizer.zero_grad()

        adj_pred, mu, logvar, class_logits = model(data.x, data.edge_index, data.batch)

        classification_loss = F.cross_entropy(class_logits, data.y)
        kl_term_loss = kl_loss(mu, logvar)
        
        # reconstruction loss is a little bit more complex
        true_adj = to_dense_adj(data.edge_index, batch=data.batch)
        reconstruction_loss = 0.0
        first = 0
        for batch_id in range(true_adj.size(0)):
            n_nodes = (data.batch == batch_id).sum().item()
            adj_true_i = true_adj[batch_id, :n_nodes, :n_nodes]
            adj_pred_i = adj_pred[first:first+n_nodes, first:first+n_nodes]
            reconstruction_loss += F.binary_cross_entropy(adj_true_i,adj_pred_i)
            first += n_nodes
        reconstruction_loss /= true_adj.size(0)

        #total loss
        loss = classification_loss + kl_weight*kl_term_loss + reconstruction_loss
        loss.backward()
        optimizer.step()

        # accumulate total losses
        total_loss += loss.item()

    return total_loss/len(td_loader)

