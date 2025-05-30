import torch
import torch.nn as nn
import torch.nn.functional as F

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
