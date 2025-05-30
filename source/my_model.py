import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool
#from torch_geometric.utils import to_dense_adj

# Encoder class
class VGAE_encoder(nn.Module):
    def __init__(self, in_dim, hid_dim, lat_dim, edge_feat_dim, hid_edge_nn_dim=128):
        super().__init__()
        
        # primo conv come prima
        nn1_edge_maps = nn.Sequential(
            nn.Linear(edge_feat_dim, hid_edge_nn_dim),
            nn.ReLU(),
            nn.Linear(hid_edge_nn_dim, in_dim * hid_dim)
        )
        self.conv1 = NNConv(in_dim, hid_dim, nn1_edge_maps, aggr='mean')
        
        # NUOVO conv2
        nn2_edge_maps = nn.Sequential(
            nn.Linear(edge_feat_dim, hid_edge_nn_dim),
            nn.ReLU(),
            nn.Linear(hid_edge_nn_dim, hid_dim * hid_dim)
        )
        self.conv2 = NNConv(hid_dim, hid_dim, nn2_edge_maps, aggr='mean')
        
        # head per mu e logvar
        nn_mu_edge_maps = nn.Sequential(
            nn.Linear(edge_feat_dim, hid_edge_nn_dim),
            nn.ReLU(),
            nn.Linear(hid_edge_nn_dim, hid_dim * lat_dim)
        )
        self.conv_mu = NNConv(hid_dim, lat_dim, nn_mu_edge_maps, aggr='sum')
        
        nn_logvar_edge_maps = nn.Sequential(
            nn.Linear(edge_feat_dim, hid_edge_nn_dim),
            nn.ReLU(),
            nn.Linear(hid_edge_nn_dim, hid_dim * lat_dim)
        )
        self.conv_logvar = NNConv(hid_dim, lat_dim, nn_logvar_edge_maps, aggr='sum')
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index, edge_attr))  # nuovo layer
        h = self.dropout(h)
        # compute node level mean and logvar
        mu = self.conv_mu(h, edge_index, edge_attr)
        logvar = self.conv_logvar(h, edge_index, edge_attr)
        return mu, logvar

def reparametrize(mu, logvar):
    logvar = torch.clamp(logvar, min=-10.0, max=10.0) # after debug print
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps*std
    
# Decoder class: less is more
class VGAE_decoder(nn.Module):
    def __init__(self):
        super().__init__()

        pass
        
    def forward(self, z, edge_index):
        src, dst = edge_index
        score = (z[src] * z[dst]).sum(dim=-1)
        return torch.sigmoid(score)
        

# final class all the model here   - new release: from CGNConv to NNConv
class VGAE_all(nn.Module):
    def __init__(self, in_dim, hid_dim, lat_dim, edge_feat_dim, hid_edge_nn_dim=32, 
                 out_classes=6, hid_dim_classifier=128):
        super().__init__()
        self.encoder = VGAE_encoder(in_dim, hid_dim, lat_dim, edge_feat_dim, hid_edge_nn_dim)
        self.decoder = VGAE_decoder()
                     
        # new decoder specifically for edge_attr
        self.edge_attr_decoder = nn.Sequential(
            nn.Linear(lat_dim*2, lat_dim),
            nn.LeakyReLU(0.15),
            nn.Linear(lat_dim, edge_feat_dim),
            nn.Sigmoid()
        )
                     
        self.classifier = nn.Sequential(
            nn.Linear(lat_dim, hid_dim_classifier),
            nn.LayerNorm(hid_dim_classifier),
            nn.ReLU(),
            # add a dropout to avoid/mitigate overfitting - try diff values 
            nn.Dropout(0.3), #20% previous dropout
            
            nn.Linear(hid_dim_classifier,hid_dim_classifier//2),
            # nn.LayerNorm(hid_dim_classifier//2),    - COMMENTATO 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hid_dim_classifier//2, out_classes)
        )
                     
    def forward(self, data, enable_classifier=True):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # check on x and edge_attr != None
        if (x is None) or (edge_attr is None):
            raise ValueError("None values for features data.x or for data.edge_attr!")
            
        mu, logvar = self.encoder(x, edge_index, edge_attr)
        z = reparametrize(mu, logvar)
        
        adj_pred = self.decoder(z, data.edge_index) if z is not None else None
        
        row, col = edge_index
        z_pair = torch.cat([z[row], z[col]], dim=1)
        edge_attr_pred = self.edge_attr_decoder(z_pair)
        #edge_attr_pred = self.edge_attr_decoder(z, data.edge_index)
        
        # pooling if classifier was enabled: in pre-training we work only with VGAE
        if enable_classifier:
            if z is None:
                raise ValueError("Latent node embeddings 'z' are None! Unable to use the classifier.")
            graph_embedding = global_mean_pool(z, batch)
            class_logits = self.classifier(graph_embedding)
        # else if classifier is NOT enabled then class_logits = None
        else:
            class_logits = None
            
        return adj_pred, edge_attr_pred, mu, logvar, class_logits, z
        
