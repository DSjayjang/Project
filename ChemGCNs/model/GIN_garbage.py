import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.datasets import MoleculeNet
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd
from sklearn.metrics import roc_auc_score
print('ok?', MoleculeNet)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MLP(nn.Module):
    """
    Multi-layer Perceptron (MLP) implementation for GIN.
    This MLP is used both in the GINConv layers and for final predictions.

    Args:
        in_channels (int): Number of input features
        hidden_channels (int): Number of hidden features in intermediate layers
        out_channels (int): Number of output features
        num_layers (int, optional): Number of layers in the MLP. Default: 2

    The MLP consists of:
    - Linear layers with ReLU activation
    - Batch normalization (except for single layer case)
    - No activation on the final layer
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        if num_layers == 1:
            # If single layer, don't use batch norm
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

            self.lins.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.batch_norms[i](x)
            x = F.relu(x)
        x = self.lins[-1](x)
        return x

class GIN(nn.Module):
    """
    Graph Isomorphism Network (GIN) implementation.
    GIN is a powerful graph neural network architecture that can distinguish between different graph structures.

    Args:
        in_channels (int): Number of input node features
        hidden_channels (int): Number of hidden features
        out_channels (int): Number of output features
        num_layers (int, optional): Number of GIN layers. Default: 3
        dropout (float, optional): Dropout probability. Default: 0.5
        epsilon (float, optional): Initial value for learnable epsilon in GINConv. Default: 0

    The architecture consists of:
    1. Initial node feature projection
    2. Multiple GIN layers with MLPs
    3. Global mean pooling
    4. Final prediction MLP
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5, epsilon=0):
        super(GIN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Initial projection of node features
        self.node_encoder = nn.Linear(in_channels, hidden_channels)

        # GIN convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP(hidden_channels, hidden_channels, hidden_channels)
            # epsilon can be learned or fixed
            self.convs.append(GINConv(mlp, train_eps=True, eps=epsilon))

        # Batch normalization layers
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])

        # Prediction MLP
        self.mlp = MLP(hidden_channels, hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the GIN model.

        Args:
            x (torch.Tensor): Input node features
            edge_index (torch.Tensor): Edge index tensor
            batch (torch.Tensor): Batch tensor indicating the graph structure

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels)
        """
        # Initial embedding
        x = self.node_encoder(x.float())

        # Store representations from each layer for readout
        xs = []

        # GIN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        # Global pooling (mean)
        x = global_mean_pool(x, batch)

        # Final prediction
        x = self.mlp(x)

        return x

    def get_embeddings(self, x, edge_index, batch):
        """
        Get node embeddings from all layers of the GIN model.
        This is useful for visualization or analysis of learned representations.

        Args:
            x (Tensor): Node features of shape [num_nodes, in_channels]
            edge_index (Tensor): Graph connectivity in COO format of shape [2, num_edges]
            batch (Tensor): Batch assignment vector of shape [num_nodes]

        Returns:
            List[Tensor]: List of node embeddings from each layer
        """
        # Initial embedding
        x = self.node_encoder(x.float())

        # Store representations from each layer
        xs = []

        # GIN layers (without dropout for inference)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            xs.append(x)

        return xs  # Return embeddings from all layers

# Load a dataset from MoleculeNet (binary classification task)
print("Loading BBBP dataset (Blood-Brain Barrier Penetration)...")
dataset = MoleculeNet(root='data', name='BBBP')
print(f"Dataset loaded: {len(dataset)} molecules")

# Split the dataset
torch.manual_seed(42)
indices = torch.randperm(len(dataset))
train_idx = indices[:int(0.8 * len(dataset))]
val_idx = indices[int(0.8 * len(dataset)):int(0.9 * len(dataset))]
test_idx = indices[int(0.9 * len(dataset)):]

train_dataset = dataset[train_idx]
val_dataset = dataset[val_idx]
test_dataset = dataset[test_idx]

print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

# Create data loaders
from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train and evaluate model function
def train_and_evaluate(model, optimizer, train_loader, val_loader, test_loader, device, epochs=100):
    """Train and evaluate a model for molecular property prediction"""
    train_losses = []
    val_losses = []
    val_aucs = []
    best_val_auc = 0
    best_model = None

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        total_loss = 0
        total_samples = 0

        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)

            out = model(data.x.float(), data.edge_index, data.batch)

            # Ensure output and target have the same batch size
            if out.size(0) != data.y.size(0):
                # If sizes don't match, use the minimum size
                min_size = min(out.size(0), data.y.size(0))
                out = out[:min_size]
                data.y = data.y[:min_size]

            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            total_samples += data.num_graphs

        avg_loss = total_loss / total_samples
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_samples = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x.float(), data.edge_index, data.batch)

                # Ensure output and target have the same batch size
                if out.size(0) != data.y.size(0):
                    min_size = min(out.size(0), data.y.size(0))
                    out = out[:min_size]
                    data.y = data.y[:min_size]

                loss = criterion(out, data.y)
                val_loss += loss.item() * data.num_graphs
                val_samples += data.num_graphs

                y_true.append(data.y.cpu().numpy())
                y_pred.append(torch.sigmoid(out).cpu().numpy())

        avg_val_loss = val_loss / val_samples
        val_losses.append(avg_val_loss)

        # Calculate validation AUC
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        val_auc = roc_auc_score(y_true, y_pred)
        val_aucs.append(val_auc)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model.state_dict().copy()

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}")

    # Load best model for evaluation
    model.load_state_dict(best_model)

    # Test evaluation
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x.float(), data.edge_index, data.batch)

            # Ensure output and target have the same batch size
            if out.size(0) != data.y.size(0):
                min_size = min(out.size(0), data.y.size(0))
                out = out[:min_size]
                data.y = data.y[:min_size]

            y_true.append(data.y.cpu().numpy())
            y_pred.append(torch.sigmoid(out).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    test_auc = roc_auc_score(y_true, y_pred)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_aucs': val_aucs,
        'best_val_auc': best_val_auc,
        'test_auc': test_auc,
        'y_true': y_true,
        'y_pred': y_pred
    }

# Compare GCN, GAT, and GIN
import copy

def compare_gnn_architectures():
    """Train and compare different GNN architectures"""
    # Get input dimension
    sample = dataset[0]
    in_channels = sample.x.shape[1]

    # Define hyperparameters
    hidden_channels = 64
    out_channels = 1  # Binary classification
    num_layers = 3
    dropout = 0.5
    lr = 0.001
    weight_decay = 1e-4
    epochs = 10

    # Initialize models
    models = {
        'GIN': GIN(in_channels, hidden_channels, out_channels, num_layers, dropout=dropout, epsilon=0).to(device)
    }

    # Train and evaluate models
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name} model...")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        results[name] = train_and_evaluate(model, optimizer, train_loader, val_loader, test_loader, device, epochs)
        print(f"{name} Test AUC: {results[name]['test_auc']:.4f}")

    return results, models

# Run comparison
results, models = compare_gnn_architectures()
