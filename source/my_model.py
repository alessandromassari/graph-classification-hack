import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj

# node features gen class - mi piaceva metterla qui anche se è più "data preparation"
class gen_node_features(object):
    def __init__(self, feat_dim):
        self.feat_dim = feat_dim
    def __call__(self, data):
        # generate node features if not exist
        if not hasattr(data, 'x'):
            if hasattr(data, 'edge_index'):
                num_nodes = data.num_nodes if hasattr(data, 'num_nodes') and data.num_nodes is not None else data.edge_index.max().item() + 1
                #data.x = torch.zeros((num_nodes, self.feat_dim), dtype=torch.float) #ALTERNATIVA
                data.x = torch.randn((num_nodes, self.feat_dim), dtype=torch.float) 
            else:
                pass
        data.x = torch.nan_to_num(data.x, nan=0.0) # avoid NaN values in every case
        return data

# Encoder class
class VGAE_encoder(nn.Module):   #- DA FARE CHECK 
    def __init__(self, in_dim, hid_dim, lat_dim, edge_feat_dim, hid_edge_nn_dim=32):
        super().__init__()
        nn1_edge_maps = nn.Sequential(
            nn.Linear(edge_feat_dim, hid_edge_nn_dim),
            nn.ReLU(),
            nn.Linear(hid_edge_nn_dim, in_dim*hid_dim)    # to check this
        )
        self.conv1 = NNConv(in_dim, hid_dim, nn1_edge_maps, aggr='mean') #try with sum

        nn_mu_edge_maps = nn.Sequential(
            nn.Linear(edge_feat_dim, hid_edge_nn_dim),
            nn.ReLU(),
            nn.Linear(hid_edge_nn_dim, in_dim*lat_dim) 
        )
        self.conv_mu = NNConv(hid_dim,lat_dim, nn_mu_edge_maps, aggr='mean')

       nn_logvar_edge_maps = nn.Sequential(
            nn.Linear(edge_feat_dim, hid_edge_nn_dim),
            nn.ReLU(),
            nn.Linear(hid_edge_nn_dim, in_dim*lat_dim)
        )
        self.conv_logvar = NNConv(hid_dim,lat_dim,nn_logvar_edge_maps, aggr='mean')

        self.dropout = nn.Dropout(0.2) # 20% dropout
        
    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = self.dropout(h)
        # compute node level mean and logvar
        mu = self.conv_mu(h, edge_index, edge_attr)
        logvar = self.conv_logvar(h, edge_index, edge_attr)
        return mu, logvar

# Decoder class
class VGAE_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, z):
        adj_pred = torch.sigmoid(torch.mm(z, z.t()))
        return adj_pred

def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

# final class all the model here   - new release: from CGNConv to NNConv
class VGAE_all(nn.Module):
    def __init__(self, in_dim, hid_dim, lat_dim, edge_feat_dim, hid_edge_nn_dim=32, 
                 out_classes, hid_dim_classifier=64):
        super().__init__()
        self.encoder = VGAE_encoder(in_dim, hid_dim, lat_dim, edge_feat_dim, hid_edge_nn_dim)
        self.decoder = VGAE_decoder()
        self.classifier = nn.Sequential(
            nn.Linear(lat_dim, hid_dim_classifier),
            nn.ReLU(),
            # add a 10% dropout to avoid/mitigate overfitting - try diff values 
            nn.Dropout(0.1),
            nn.Linear(hid_dim_classifier, out_classes)
        )
                     
    #  maybe possiamo inserire qui la parte di concatenazione in decoder invece di goto_the_gym.py
    def forward(self, data, enable_classifier=True):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # check on x and edge_attr != None
        if (x is None) or (edge_attr is None):
            raise ValueError("None values for features data.x or for data.edge_attr!")
            
        mu, logvar = self.encoder(x, edge_index, edge_attr)
        z = reparametrize(mu, logvar)
        adj_pred = self.decoder(z) if z is not None else None              

        # pooling if classifier was enabled: in pre-training we work only with VGAE
        if enable_classifier:
            if z is None:
                raise ValueError("Latent node embeddings 'z' are None! Unable to use the classifier.")
            graph_embedding = global_mean_pool(z, batch)
            class_logits = self.classifier(graph_embedding)
        # else if classifier is NOT enabled then class_logits = None
        else:
            class_logits = None
            
        return adj_pred, mu, logvar, class_logits, z
        
