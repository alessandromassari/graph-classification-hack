import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch_geometric.loader import DataLoader

@torch.no_grad()
def plot_latent_z(model, dataset, device, method='tsne', sample_size=1000):
    model.eval()
    all_z = []
    all_y = []

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    for data in loader:
        data = data.to(device)
        # Forward solo dell’encoder
        mu, logvar = model.encoder(data.x, data.edge_index, data.edge_attr)
        z = mu  # oppure usa reparametrize(mu, logvar) se preferisci campionamento
        all_z.append(z.cpu())
        all_y.append(data.y.cpu())

        if sum(d.size(0) for d in all_z) > sample_size:
            break  # limitiamoci a sample_size nodi totali

    # Concatenazione finale
    z_cat = torch.cat(all_z, dim=0)
    y_cat = torch.cat(all_y, dim=0).view(-1)

    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, init='pca', perplexity=30, random_state=42)
    else:
        reducer = PCA(n_components=2)
    z_2d = reducer.fit_transform(z_cat.numpy())

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y_cat, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar(scatter, label='Classe')
    plt.title(f'Latent space z visualizzato con {method.upper()}')
    plt.xlabel('Dimensione 1')
    plt.ylabel('Dimensione 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
