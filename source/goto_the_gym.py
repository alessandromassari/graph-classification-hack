import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, negative_sampling
from my_model import VGAE_all
from torch.nn import SmoothL1Loss
from losses import FocalLoss
# reconstruction loss weight - DA SPOSTARE DA QUI
recon_weight = 0.2 # previous value: 0.8
focal = FocalLoss(gamma=2.0)

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

    all_the_logits = torch.cat([positive_logits, negative_logits])
    all_the_labels = torch.cat([positive_labels, negative_labels])

    recon_loss = F.binary_cross_entropy(all_the_logits,all_the_labels)
    return recon_loss

# a new loss: huber loss instead of reconstruction loss - FORSE DA CANCELLARE 
def reconstruction_huber_loss(z, edge_index, model_decoder, num_nodes, num_neg_samp=1, beta=1.0):
    """
    Ricostruzione edge-wise con Huber loss tra z e all_edges (positivi + negativi).
    """
    # Campiona archi negativi
    neg_edge_index = negative_sampling(
        edge_index,
        num_nodes=num_nodes,
        num_neg_samples=edge_index.size(1) * num_neg_samp
    )

    # Concatenazione: archi positivi + negativi
    all_edges = torch.cat([edge_index, neg_edge_index], dim=1)

    # Decoder edge-wise: score per ogni edge
    adj_pred = model_decoder(z, all_edges)

    # Costruzione etichette: primi N positivi (1), poi negativi (0)
    num_positive_edges = edge_index.size(1)
    positive_labels = torch.ones(num_positive_edges, device=z.device)
    negative_labels = torch.zeros(adj_pred.size(0) - num_positive_edges, device=z.device)
    all_labels = torch.cat([positive_labels, negative_labels], dim=0)

    # Huber loss
    huber = SmoothL1Loss(beta=beta)
    return huber(adj_pred, all_labels)
    
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
    #print(f"PRETRAINING: Epoch {cur_epoch + 1}, KL Weight: {kl_weight:.6f}")    
    
    for data in td_loader:
        data = data.to(device)
        # reset the gradients each batch 
        optimizer.zero_grad()

        # new release update: pass whole data instead of .x ecc.
        adj_pred,mu,logvar,class_logits, z = model(data, enable_classifier=False) 

        # just a check
        if adj_pred is None or mu is None or logvar is None:
            print(f"Warning: Model output is None in pretraining epoch {cur_epoch+1}. Skip the batch")
            continue
            
        #KL term loss 
        kl_term_loss = kl_loss(mu, logvar)
        #reconstruction loss 
        reconstruction_loss = eval_reconstruction_loss(adj_pred, data.edge_index, data.x.size(0), num_neg_samp=1)
        #reconstruction_loss = reconstruction_huber_loss(z=z,edge_index=data.edge_index,model_decoder=model.decoder,num_nodes=data.x.size(0),num_neg_samp=1,beta=1.0)
        
        #total pretraining loss
        loss = kl_weight*kl_term_loss + reconstruction_loss
        loss.backward()
        optimizer.step()

        # accumulate total losses
        total_loss += loss.item() #* data.num_graphs #weight per pesare

    #DEBUG PRINT - KL and recon losses (KL should not to be so closer to zero)
    print(f"PRETRAINING: Epoch {cur_epoch + 1}, KL Loss: {kl_term_loss:.4f}, Recon Loss: {reconstruction_loss:.4f}")
    return total_loss/len(td_loader)
    
# Training procedure - classifier is in!
def train(model, td_loader, optimizer, device, kl_weight_max, cur_epoch, an_ep_kl):
    model.train()
    total_loss = 0.0
    total_guessed_pred = 0
    total_worked_graphs = 0
    
    # compute dynamic KL weight
    if cur_epoch < an_ep_kl:
        kl_weight = kl_weight_max * (cur_epoch / an_ep_kl)
    else:
        kl_weight = kl_weight_max
    
    for data in td_loader:
        data = data.to(device)
        # reset the gradients each batch 
        optimizer.zero_grad()
        
        adj_pred, mu, logvar, class_logits, z = model(data, enable_classifier=True)

        # just a check
        if adj_pred is None or mu is None or logvar is None:
            print(f"Warning: Model output is None in training epoch {cur_epoch+1}. Skip the batch")
            continue
            
        #classification loss
        #DEBUG instead of just data.y
        if data.y.dim() > 1 and data.y.size(1) == 1:
            target_y = data.y.squeeze(1)
        else:
            target_y = data.y
        #classification_loss = F.cross_entropy(class_logits, target_y) 
        classification_loss = focal(class_logits, target_y)
        
        #KL term loss - NOT USED IN TRAINING TRY AND THEN DELETE IF WORKS
        #kl_term_loss = kl_loss(mu, logvar)
        #reconstruction loss 
        reconstruction_loss = eval_reconstruction_loss(adj_pred, data.edge_index, data.x.size(0), num_neg_samp=1)
        #reconstruction_loss = reconstruction_huber_loss(z=z,edge_index=data.edge_index,model_decoder=model.decoder,num_nodes=data.x.size(0),num_neg_samp=1,beta=1.0)
        
        #total loss
        loss = classification_loss + recon_weight*reconstruction_loss
        loss.backward()
        optimizer.step()

        # accumulate total losses
        total_loss += loss.item() #* data.num_graphs if hasattr(data, 'num_graphs') else loss.item() # provare weight per pesare
        
        #DEBUG evaluate accuracy 
        preds = torch.argmax(class_logits, dim=1)
        total_guessed_pred += (preds == target_y).sum().item()
        total_worked_graphs += data.num_graphs 
        ep_accuracy = total_guessed_pred / total_worked_graphs # NOT USED
        
    return total_loss/len(td_loader)
