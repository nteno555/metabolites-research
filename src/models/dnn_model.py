import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        return self.act(out + residual)

class MassSpecDNN(nn.Module):
    # Deep Neural Network (DNN) with residual skip-connections, batch normalization,
    # SiLU activations, and dropout for mass spectrometry toxin concentration prediction.

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.2):
        super(MassSpecDNN, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.res_block1 = ResidualBlock(hidden_dims[0], dropout_rate)
        
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.res_block2 = ResidualBlock(hidden_dims[1], dropout_rate)
        
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.SiLU()
        )
        
        self.regressor_head = nn.Linear(hidden_dims[2], 1)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.res_block1(out)
        out = self.hidden_layer1(out)
        out = self.res_block2(out)
        out = self.hidden_layer2(out)
        pred = self.regressor_head(out)
        return pred.squeeze(-1)

class DNNTrainer:
    # Trainer wrapper for MassSpecDNN model with AdamW, Cosine Annealing, early stopping,
    # and Integrated Gradients / Permutation feature saliency extraction.
    def __init__(self, input_dim, lr=1e-3, weight_decay=1e-4, epochs=150, batch_size=16, device="cpu"):
        self.device = torch.device(device if torch.cuda.is_available() or torch.backends.mps.is_available() else "cpu")
        self.model = MassSpecDNN(input_dim).to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_losses = []
        self.val_losses = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.SmoothL1Loss() # Huber loss for robust regression
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        
        best_val_loss = float('inf')
        best_weights = None
        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(batch_y)
                
            scheduler.step()
            train_loss = running_loss / len(X_train)
            self.train_losses.append(train_loss)
            
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_v = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    y_v = torch.tensor(y_val, dtype=torch.float32).to(self.device)
                    val_preds = self.model(X_v)
                    val_loss = criterion(val_preds, y_v).item()
                    self.val_losses.append(val_loss)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = self.model.state_dict().copy()
                        
        if best_weights is not None:
            self.model.load_state_dict(best_weights)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds = self.model(X_t)
            return preds.cpu().numpy()

    def get_integrated_gradients(self, X, steps=20):
    # Computes Integrated Gradients for feature importance estimation using vectorized batch operations.

        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        baseline = torch.zeros_like(X_t)
        
        attributions = torch.zeros_like(X_t)
        alphas = np.linspace(0.0, 1.0, steps)
        
        for alpha in alphas:
            interpolated = baseline + alpha * (X_t - baseline)
            interpolated = interpolated.detach().requires_grad_(True)
            output = self.model(interpolated)
            output.sum().backward()
            attributions += interpolated.grad / steps
            
        integrated_grad = (X_t - baseline) * attributions
        importance_scores = torch.abs(integrated_grad).mean(dim=0).cpu().detach().numpy()
        return importance_scores

    def get_permutation_importance(self, X_val, y_val, metric_fn):
        # Computes Permutation Feature Importance on validation set.

        baseline_score = metric_fn(y_val, self.predict(X_val))
        importances = []
        for col in range(X_val.shape[1]):
            X_perm = X_val.copy()
            np.random.shuffle(X_perm[:, col])
            perm_score = metric_fn(y_val, self.predict(X_perm))
            importances.append(baseline_score - perm_score) # Drop in score
        return np.array(importances)

if __name__ == "__main__":
    X_dummy = np.random.randn(50, 107).astype(np.float32)
    y_dummy = np.random.uniform(0, 10, size=50).astype(np.float32)
    
    trainer = DNNTrainer(input_dim=107, epochs=10)
    trainer.fit(X_dummy, y_dummy)
    preds = trainer.predict(X_dummy)
    print(f"DNN dummy predictions shape: {preds.shape}")
    ig_scores = trainer.get_integrated_gradients(X_dummy)
    print(f"Integrated Gradients scores shape: {ig_scores.shape}")
