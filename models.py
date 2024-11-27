import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, features, input_nodes):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()  # Aplana las entradas para que sean un vector.

        # Red neuronal con capas más ligeras y Dropout
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_nodes, 256),  # Capa más pequeña
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout para regularización
            nn.Linear(256, 128),  # Otra capa reducida
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, features)  # Salida con 'features' clases
        )

    def forward(self, x):
        x = self.flatten(x)  # Aplana los datos de entrada
        logits = self.linear_relu_stack(x)  # Pasa por la red
        return logits
