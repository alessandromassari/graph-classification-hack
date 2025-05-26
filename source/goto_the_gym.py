import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, negative_sampling
from my_model import VGAE_all

# our beloved Kullback-Leibler term loss
def kl_loss(mu, logvar):
    # clip logvar to avoid extreme values 
    clip_logvar = torch.clamp(logvar, min=-5.0, max=5.0) 
    return -0.5 * torch.mean(1 + clip_logvar -mu.pow(2) - clip_logvar.exp())

# reconstruction loss function
def eval_reconstruction_loss(adj_pred, edge_index, num_nodes, num_neg_samp=1):
    positive_logits = adj_pred[edge_index[0], edge_index[1]]
    positive_labels = torch.ones_like(positive_logits)

    neg_edge_index = negative_sampling(
        edge_index,
        num_nodes = num_nodes,
        num_neg_samples = edge_index.size(1)*num_neg_samp)
    # as for positive but for negative edges
    negative_logits = adj_pred[neg_edge_index[0],neg_edge_index[1]]
    negative_labels = torch.zeros_like(negative_logits)

    # DEBUG info: use .cat instead of np.concatenate() to keep all on GPU
    all_the_logits = torch.cat([positive_logits, negative_logits])
    all_the_labels = torch.cat([positive_labels, negative_labels])

    recon_loss = F.binary_cross_entropy(all_the_logits,all_the_labels)

    return recon_loss

# Pre training procedure - no classifiers here
def pretraining(model, td_loader, optimizer, device, kl_weight_max, cur_epoch, an_ep_kl):
    model.train()
    total_loss = 0.0

    # compute dynamic KL weight
    if cur_epoch < an_ep_kl:
        kl_weight = kl_weight_max * (cur_epoch / an_ep_kl)
    else:
        kl_weight = kl_weight_max
    # DEBUG PRINT  
    print(f"PRETRAINING: Epoch {cur_epoch + 1}, KL Weight: {kl_weight:.6f}")    
    
    for data in td_loader:
        data = data.to(device)
        # reset the gradients each batch 
        optimizer.zero_grad()
        
        adj_pred,mu,logvar,class_logits = model(data.x,
                                                data.edge_index,
                                                data.batch,
                                                enable_classifier=False)
        # DEBUG: Stampa le statistiche di mu, logvar, z
        mu, logvar = model.encoder(data.x, data.edge_index)
        z = reparametrize(mu, logvar)
        adj_pred_logits_before_sigmoid = torch.mm(z, z.t()) # Calcola i logits prima della sigmoid

        # Stampa alcune statistiche
        print(f"  Shape of z: {z.shape}")
        print(f"  Mean of mu: {mu.mean().item():.4f}, Std of mu: {mu.std().item():.4f}")
        print(f"  Mean of logvar: {logvar.mean().item():.4f}, Std of logvar: {logvar.std().item():.4f}")
        print(f"  Mean of z: {z.mean().item():.4f}, Std of z: {z.std().item():.4f}")
        print(f"  Min of adj_pred_logits: {adj_pred_logits_before_sigmoid.min().item():.4f}, Max of adj_pred_logits: {adj_pred_logits_before_sigmoid.max().item():.4f}")

        adj_pred = model.decoder(z) # Questa applica la sigmoid
    
        print(f"  Min of adj_pred (probs): {adj_pred.min().item():.4f}, Max of adj_pred (probs): {adj_pred.max().item():.4f}")
        print(f"  Mean of adj_pred (probs): {adj_pred.mean().item():.4f}")
        # ------------- FINE DEBUG PRINT ------------   
        
        #KL term loss
        kl_term_loss = kl_loss(mu, logvar)
        #reconstruction loss 
        reconstruction_loss = eval_reconstruction_loss(adj_pred,data.edge_index,data.x.size(0),num_neg_samp=1)
        #total loss
        loss = kl_weight*kl_term_loss + reconstruction_loss
        loss.backward()
        optimizer.step()

        # accumulate total losses
        total_loss += loss.item() * data.num_graphs if hasattr(data, 'num_graphs') else loss.item()#weight per pesare

    return total_loss/len(td_loader)
    
# Training procedure
def train(model, td_loader, optimizer, device, kl_weight_max, cur_epoch, an_ep_kl):
    model.train()
    total_loss = 0.0

    # compute dynamic KL weight
    if cur_epoch < annealing_epoch:
        kl_weight = kl_weight_max * (cur_epoch / an_ep_kl)
    else:
        kl_weight = kl_weight_max
    # DEBUG PRINT  
    print(f"Epoch {cur_epoch + 1}, KL Weight: {kl_weight:.6f}")    
    
    for data in td_loader:
        data = data.to(device)
        # reset the gradients each batch 
        optimizer.zero_grad()
        
        adj_pred, mu, logvar, class_logits = model(data.x, data.edge_index, data.batch)
     
        #classification loss
        classification_loss = F.cross_entropy(class_logits, data.y)
        #KL term loss
        kl_term_loss = kl_loss(mu, logvar)
        #reconstruction loss 
        reconstruction_loss = eval_reconstruction_loss(adj_pred,data.edge_index,data.x.size(0),num_neg_samp=1)
        #total loss
        loss = classification_loss + kl_weight*kl_term_loss + reconstruction_loss
        loss.backward()
        optimizer.step()

        # accumulate total losses
        total_loss += loss.item() * data.num_graphs if hasattr(data, 'num_graphs') else loss.item() # provare weight per pesare

    return total_loss/len(td_loader)
