import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

bce_weight = 0.2
#bce loss 
def compute_bce_loss(z, edge_index, decoder, num_nodes, num_neg_samples=1):
    """
    Calcola una BCE loss leggera tra:
    - edge_index (positivi)
    - archi negativi campionati

    Args:
        z: embedding latenti dei nodi [num_nodes, lat_dim]
        edge_index: archi veri [2, num_edges]
        decoder: modulo decoder da usare su (z, edge_index)
        num_nodes: numero nodi (int)
        num_neg_samples: moltiplicatore negativo (di solito 1)
    """
    device = z.device
    # Score per gli archi reali (positivi)
    pos_scores = decoder(z, edge_index)
    pos_labels = torch.ones_like(pos_scores)

    # Campionamento archi negativi
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=edge_index.size(1) * num_neg_samples,
        method='sparse'
    )
    neg_scores = decoder(z, neg_edge_index)
    neg_labels = torch.zeros_like(neg_scores)

    # Concatenazione score/label
    all_scores = torch.cat([pos_scores, neg_scores], dim=0)
    all_labels = torch.cat([pos_labels, neg_labels], dim=0)

    # Calcolo Binary Cross Entropy Loss
    bce_loss = F.binary_cross_entropy(all_scores, all_labels)
    return bce_loss

def compute_recon_loss(z, edge_index, edge_attr, edge_attr_decoder, decoder):

    adj_logits = torch.matmul(z, z.t())           # [N, N]
    adj_pred = torch.sigmoid(adj_logits)
    """ TROPPO PESANTE CON LA BCE
    N = z.size(0)
    device = z.device
    adj_true = torch.zeros((N, N), device=device)
    row, col = edge_index
    adj_true[row, col] = 1.0
    bce_loss = F.binary_cross_entropy(adj_pred, adj_true)
    """
    bce_loss = compute_bce_loss(z, edge_index, decoder, num_nodes, num_neg_samples=1)
    
    row, col = edge_index
    z_pair = torch.cat([z[row], z[col]], dim=-1)           # [E, lat_dim*2]
    edge_attr_pred = edge_attr_decoder(z_pair)             # [E, edge_feat_dim]

    # 4) MSE Loss per gli attributi degli archi
    mse_loss = F.mse_loss(edge_attr_pred, edge_attr)

    # 5) Combinazione pesata delle due loss
    return mse_loss + bce_weight * bce_loss

class FocalLoss(nn.Module):
    """
    Focal Loss, come descritto in Lin et al. 2017:
    FL(p_t) = - (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma (float): focussing parameter >= 0. Quando gamma=0 coincide con CE standard.
        reduction (str): 'none' | 'mean' | 'sum'
    """
    def __init__(self, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: tensor di forma [B, C] (non normalizzati)
        target: tensor di forma [B] con classi in {0,...,C-1}
        """
        # 1) calcolo la cross-entropy per esempio (senza reduction)
        ce_loss = F.cross_entropy(logits, target, reduction='none')  # [B]

        # 2) ricavo p_t = exp(-ce)
        pt = torch.exp(-ce_loss)  # [B]

        # 3) applichiamo il fattore focusing (1-pt)^gamma
        focal_term = (1 - pt) ** self.gamma  # [B]

        loss = focal_term * ce_loss  # [B]

        # 4) reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
