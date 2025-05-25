import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj

# Encoder class
class VGAE_encoder(nn.Module):   #- DA FARE CHECK 
    def __init__(self, in_dim, hid_dim, lat_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv_mu = GCNConv(hid_dim, lat_dim)
        self.conv_logvar = GCNConv(hid_dim, lat_dim)  

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)
        return mu, logvar

# Decoder class
class GAE_decoder(nn.Module):

    def forward(self, z):
        adj_pred = torch.sigmoid(torch.mm(z, z.t()))
        return adj_pred

# our beloved Kullback-Leibler term
def kl_loss(mu, logvar):
    # QUI POSSIAMO PENSARE DI MODIFICARE QUALCOSINA INSERENDO DI LIMITI SUL LOGVAR VALUE MA FORSE NON SERVE
    return -0.5 * torch.mean(1 + logvar -mu.pow(2) - logvar.exp())

def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

# final class all the model here   - DA FARE CHECK 
class VGAE_all(nn.Module):
    def __init__(self, in_dim, hid_dim, lat_dim, out_classes):
        super().__init__()
        self.encoder = VGAE_encoder(in_dim, hid_dim, lat_dim)
        self.decoder = VGAE_decoder()
        self.classifier = nn.Sequential(
            nn.Linear(lat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, out_classes)
        )

    def forward(self, x, edge_index, batch):
        mu, logvar = self.encoder(x, edge_index)
        z = reparametrize(mu, logvar)
        adj_pred = self.decoder(z)                # check se serve

        # pooling
        graph_embedding = global_mean_pool(z, batch)
        class_logits = self.classifier(graph_embedding)
        return adj_pred, mu, logvar, class_logits
