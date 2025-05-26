import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj

# node features gen class - mi piaceva metterla qui anche se è più "data preparation"
class gen_node_features(object):
    def __init__(self, feat_dim):
        self.feat_dim = feat_dim
    def __call__(self, data):
        num_nodes = data.num_nodes if hasattr(data, 'num_nodes') and data.num_nodes is not None else data.edge_index.max().item() + 1
        data.x = torch.zeros((num_nodes, self.feat_dim), dtype=torch.float)
        return data

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
class VGAE_decoder(nn.Module):

    def forward(self, z):
        adj_pred = torch.sigmoid(torch.mm(z, z.t()))
        return adj_pred


def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

# final class all the model here   - neew release: flag for split the model and pretrain
class VGAE_all(nn.Module):
    def __init__(self, in_dim, hid_dim, lat_dim, out_classes):
        super().__init__()
        self.encoder = VGAE_encoder(in_dim, hid_dim, lat_dim)
        self.decoder = VGAE_decoder()
        self.classifier = nn.Sequential(
            nn.Linear(lat_dim, 64),
            nn.ReLU(),
            # add a 10% dropout to avoid/mitigate overfitting - try diff values 
            nn.Dropout(0.1),
            nn.Linear(64, out_classes)
        )
    #  maybe possiamo inserire qui la parte di concatenazione in decoder invece di goto_the_gym.py
    def forward(self, x, edge_index, batch, enable_classifier=True):
        mu, logvar = self.encoder(x, edge_index)
        z = reparametrize(mu, logvar)
        adj_pred = self.decoder(z)                

        # pooling if classifier was enabled: in pre-training we work only with VGAE
        if enable_classifier:
            graph_embedding = global_mean_pool(z, batch)
            class_logits = self.classifier(graph_embedding)
        else:
            class_logits = None
            
        return adj_pred, mu, logvar, class_logits



        
