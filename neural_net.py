import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np

class BurnoutPredictor(nn.Module):
    def __init__(self, input_size=4, hidden_sizes=[64, 32], activation='relu'):
        super(BurnoutPredictor, self).__init__()
        
        # Create layers dynamically
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        # Activation function
        self.activation_func = nn.ReLU() if activation == 'relu' else nn.Sigmoid()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):  # All layers except the last
            x = self.activation_func(layer(x))
        x = self.layers[-1](x)  # Output layer (no activation)
        return x

def prepare_data(X, y, train_split=0.7, val_split=0.15, batch_size=32):
    """Prepare data loaders for training, validation and testing"""
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Calculate splits
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


def train_and_evaluate(model, optimizer, train_loader, val_loader, test_loader, num_epochs, device='cpu'):
    """Train the model and track metrics"""
    model = model.to(device)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        # Calculate average training loss
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_actuals = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                val_predictions.extend(outputs.cpu().numpy())
                val_actuals.extend(targets.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        # Calculate R² score and MAE for validation set
        r2 = r2_score(val_actuals, val_predictions)
        mae = mean_absolute_error(val_actuals, val_predictions)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, R² Score: {r2:.4f}, MAE: {mae:.4f}")
        print("-" * 40)

    # Test phase
    model.eval()
    test_loss = 0.0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)

    # Calculate R² score and MAE for test set
    r2_test = r2_score(actuals, predictions)
    mae_test = mean_absolute_error(actuals, predictions)

    print(f"Test Loss: {test_loss:.4f}, Test R² Score: {r2_test:.4f}, Test MAE: {mae_test:.4f}")
    
    return train_losses, val_losses, test_loss, r2_test, mae_test

def plot_training_results(train_losses_sigmoid, val_losses_sigmoid,
                         train_losses_relu, val_losses_relu,
                         test_loss_sigmoid, test_loss_relu,
                         r2_sigmoid, mae_sigmoid,
                         r2_relu, mae_relu):
    """Plot training results and comparison"""
    
    # Loss Plots
    epochs = range(1, len(train_losses_sigmoid) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses_sigmoid, label='Sigmoid Train Loss')
    plt.plot(epochs, val_losses_sigmoid, label='Sigmoid Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Sigmoid Activation: Training & Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses_relu, label='ReLU Train Loss')
    plt.plot(epochs, val_losses_relu, label='ReLU Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ReLU Activation: Training & Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Metrics Comparison
    metrics = ['Test Loss', 'R² Score', 'MAE']
    sigmoid_values = [test_loss_sigmoid, r2_sigmoid, mae_sigmoid]
    relu_values = [test_loss_relu, r2_relu, mae_relu]
    
    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, sigmoid_values, width, label='Sigmoid')
    rects2 = ax.bar(x + width/2, relu_values, width, label='ReLU')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Values')
    ax.set_title('Test Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add labels on top of the bars
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()

def main():
    # Hyperparameters
    learning_rate = 0.001
    num_epochs = 25
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data (Assuming X and y are defined as per your dataset)
    
    train_loader, val_loader, test_loader = prepare_data(X, y, batch_size=batch_size)
    
    # Initialize models
    model_sigmoid = BurnoutPredictor(activation='sigmoid')
    model_relu = BurnoutPredictor(activation='relu')
    
    # Initialize optimizers
    optimizer_sigmoid = optim.Adam(model_sigmoid.parameters(), lr=learning_rate)
    optimizer_relu = optim.Adam(model_relu.parameters(), lr=learning_rate)
    
    # Train models and get results including metrics
    print("Training Sigmoid Model...")
    results_sigmoid = train_and_evaluate(
        model_sigmoid,
        optimizer_sigmoid,
        train_loader,
        val_loader,
        test_loader,
        num_epochs,
        device
    )
    
    print("\nTraining ReLU Model...")
    results_relu = train_and_evaluate(
        model_relu,
        optimizer_relu,
        train_loader,
        val_loader,
        test_loader,
        num_epochs,
        device
    )
    # Plot results
    plot_training_results(
        results_sigmoid[0], results_sigmoid[1],
        results_relu[0], results_relu[1],
        results_sigmoid[2], results_relu[2],
        results_sigmoid[3], results_sigmoid[4],
        results_relu[3], results_relu[4]
    )

if __name__ == "__main__":
   main()
