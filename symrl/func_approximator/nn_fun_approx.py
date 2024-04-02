import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
from graphviz import Digraph
try:
    from .base_approx import BaseFuncApproximator, FeatureExtractor
except ImportError:
    from base_approx import BaseFuncApproximator, FeatureExtractor
import pickle
import json
import os

class NeuralFuncApproximator(BaseFuncApproximator):
    def __init__(self, feature_extractor, num_features, num_actions, learning_rate=0.01):
        super().__init__()
        self._feature_extractor = feature_extractor
        self.num_features = num_features
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        
        # Define the neural network architecture
        self.model = nn.Sequential(
            nn.Linear(num_features, 2),   # First hidden layer
            nn.ReLU(),                    # Activation function
            nn.Linear(2, 2),              # Second hidden layer
            nn.ReLU(),                    # Activation function
            nn.Linear(2, num_actions)     # Output layer
        )
        
        # Define the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Check for GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.avg_loss = 100.0
    
    @property
    def feature_extractor(self) -> FeatureExtractor:
        return self._feature_extractor

    def predict_q(self, state, action):
        state = self._feature_extractor(state)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_values = self.model(state)
        return q_values[action].item()
    
    def update_v(self, state, target):
        state = self._feature_extractor(state)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        target = torch.tensor(target, dtype=torch.float32).to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass: compute predicted Q-values
        q_values = self.model(state)
        
        # Use the max Q-value as the predicted value (V) of the state
        predicted_v, _ = torch.max(q_values, dim=1)
        
        # Compute loss using the predicted V and the target V
        loss = nn.MSELoss()(predicted_v, target)
        
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # Perform a single optimization step (parameter update)
        self.optimizer.step()
        self.avg_loss = 0.9 * self.avg_loss + 0.1 * loss.item()

    def update_q(self, state, action, target):
        state = self._feature_extractor(state)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        target = torch.tensor(target, dtype=torch.float32).to(self.device)
        self.optimizer.zero_grad()
        q_values = self.model(state)
        loss = nn.MSELoss()(q_values[action], target)
        loss.backward()
        self.optimizer.step()
        self.avg_loss = 0.9 * self.avg_loss + 0.1 * loss.item()
    
    def predict_v(self, state):
        state = self._feature_extractor(state)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_values = self.model(state)
        return torch.max(q_values, 1)[0].item()
    
    def pretty_print_state(self, state):
        state_approx = self.feature_extractor.pretty_print_state(state)
        return state_approx
    
    def pretty_print_action(self, action):
        action_approx = self.feature_extractor.pretty_print_action(action)
        return action_approx

    def pretty_print_approximator(self):
        # # Assume 'feature_names' is a list of feature names from the feature extractor
        # feature_names = self.feature_extractor.pretty_print_feature_extractor()
        # # Convert the concatenated array style feature names to a list [a, b, c, ...]
        # feature_names = [feature_name.strip('[ ]') for feature_name in feature_names.split(",")]
        dot = Digraph()

        # Helper function to add node for a layer unit
        def add_layer_nodes(layer_name, num_nodes, biases=None):
            for i in range(num_nodes):
                node_name = f"{layer_name}_{i}"
                label = f"{layer_name}_{i}"
                if biases is not None:
                    label += f"\nbias={biases[i]:.2f}"
                dot.node(node_name, label=label)
        
        # Helper function to add edges between layer units with weights
        def add_layer_connections(prev_layer_name, layer_name, weights, prev_layer_size):
            for i in range(weights.shape[0]):  # For each unit in the current layer
                for j in range(prev_layer_size):  # Connect from all units in the previous layer
                    weight = weights[i, j]
                    dot.edge(f"{prev_layer_name}_{j}", f"{layer_name}_{i}", label=f"{weight:.2f}")

        prev_layer_name = "Input"
        input_size = None  # Will be set based on the first encountered Linear layer

        for i, layer in enumerate(self.model):
            layer_name = f"Layer{i}"
            
            if isinstance(layer, nn.Linear):
                # Set input size from the first Linear layer if not set
                if input_size is None:
                    input_size = layer.in_features
                    add_layer_nodes(prev_layer_name, input_size)
                
                # Add nodes for the current layer
                biases = layer.bias.data.cpu().numpy() if layer.bias is not None else None
                add_layer_nodes(layer_name, layer.out_features, biases=biases)

                # Add edges (connections) between the previous and current layer with weights
                weights = layer.weight.data.cpu().numpy()
                add_layer_connections(prev_layer_name, layer_name, weights, layer.in_features)

                # Update for next iteration
                prev_layer_name = layer_name
                input_size = layer.out_features

        # Return the DOT source instead of rendering
        return dot.source

    def save(self, folder):
        # Save model weights
        model_filename = f"{folder}/NNModel.pth"
        torch.save(self.model.state_dict(), model_filename)
        
        # Save feature extractor
        feature_extractor_filename = f"{folder}/FeatureExtractor.pkl"
        with open(feature_extractor_filename, 'wb') as f:
            pickle.dump(self._feature_extractor, f)
        
        # Save settings
        settings = {
            "num_features": self.num_features,
            "num_actions": self.num_actions,
            "learning_rate": self.learning_rate,
            "type": "NeuralFuncApproximator"
        }
        settings_filename = f"{folder}/Settings.json"
        with open(settings_filename, 'w') as f:
            json.dump(settings, f)

    @staticmethod
    def load(folder):
        # Load settings
        settings_filename = os.path.join(folder, "Settings.json")
        with open(settings_filename, 'r') as f:
            settings = json.load(f)
        
        # Load feature extractor
        feature_extractor_filename = os.path.join(folder, "FeatureExtractor.pkl")   #f"{folder}/FeatureExtractor.pkl"
        with open(feature_extractor_filename, 'rb') as f:
            feature_extractor = pickle.load(f)
        
        # Initialize approximator
        approximator = NeuralFuncApproximator(
            feature_extractor, 
            settings["num_features"], 
            settings["num_actions"], 
            settings["learning_rate"])
        
        # Load model weights
        model_filename = os.path.join(folder, "NNModel.pth")
        approximator.model.load_state_dict(torch.load(model_filename, map_location=approximator.device))
        
        return approximator