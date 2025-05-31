import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_recon_loss(z, edge_index, edge_attr, edge_attr_decoder):

    adj_logits = torch.matmul(z, z.t())           # [N, N]
    adj_pred = torch.sigmoid(adj_logits)

    N = z.size(0)
    device = z.device
    adj_true = torch.zeros((N, N), device=device)
    row, col = edge_index
    adj_true[row, col] = 1.0
    bce_loss = F.binary_cross_entropy(adj_pred, adj_true)

    z_pair = torch.cat([z[row], z[col]], dim=-1)           # [E, lat_dim*2]
    edge_attr_pred = edge_attr_decoder(z_pair)             # [E, edge_feat_dim]

    # 4) MSE Loss per gli attributi degli archi
    mse_loss = F.mse_loss(edge_attr_pred, edge_attr)

    # 5) Combinazione pesata delle due loss
    return 0.1 * bce_loss + mse_loss

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
