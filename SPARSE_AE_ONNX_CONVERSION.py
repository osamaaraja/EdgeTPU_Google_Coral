
import torch
import torch.nn as nn
import os
cwd = os.getcwd()

class SparseAutoencoder(nn.Module):
    def __init__(self, layer_sizes, latent_dim, l1_penalty, target_sparsity, dropout_rate):
        super(SparseAutoencoder, self).__init__()

        assert len(layer_sizes) > 1, "layer_sizes list must contain at least input and one hidden layer size"

        # Encoder
        encoder_layers = []
        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            encoder_layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))  # Add BatchNorm layer
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
        # Add the latent layer
        encoder_layers.append(nn.Linear(layer_sizes[-1], latent_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = [nn.Linear(latent_dim, layer_sizes[-1]), nn.ReLU(), nn.Dropout(dropout_rate)]
        for i in range(len(layer_sizes) - 1, 0, -1):
            decoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i - 1]))
            if i > 1:  # Add BatchNorm, ReLU, and Dropout to all but the last layer
                decoder_layers.append(nn.BatchNorm1d(layer_sizes[i - 1]))  # Add BatchNorm layer
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout_rate))
        self.decoder = nn.Sequential(*decoder_layers)

        self.l1_penalty = l1_penalty
        self.target_sparsity = target_sparsity

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def kl_divergence(self, rho, rho_hat):
        return torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))

    def loss_function(self, inputs, outputs, encoded):
        mse_loss = nn.functional.mse_loss(outputs, inputs)
        rho_hat = torch.mean(encoded, dim=0)
        sparsity_loss = self.l1_penalty * self.kl_divergence(self.target_sparsity, rho_hat)
        total_loss = mse_loss + sparsity_loss
        return total_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 16
latent_dim = 4
hidden_dim_1 = 12
hidden_dim_2 = 10
hidden_dim_3 = 8
hidden_dim_4 = 6
layer_sizes = [input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, hidden_dim_4]

l1_penalty = 0.05
target_sparsity = 0.35
dropout_rate = 0.1

lr = 1e-6
num_epochs_for_fine_tuning = 100
batch_size = 64

model = SparseAutoencoder(
        layer_sizes=layer_sizes,
        latent_dim=latent_dim,
        l1_penalty=l1_penalty,
        target_sparsity=target_sparsity,
        dropout_rate=dropout_rate
    ).to(device)
# with pre-trained weights

best_model_weights = f"{cwd}/fine_tuned_weights_participant_4_0_1Hz.pth"
model.load_state_dict(torch.load(best_model_weights, map_location=device))

# Set the model to evaluation mode
model.eval()

dummy_input = torch.randn(1, 16).to(device) # [batch_size, input_dim]

# Export the model to ONNX format, setting export params to True makes sure that the model trained weights are also exported.
torch.onnx.export(model, dummy_input, "SAE.onnx", export_params=True,
                  opset_version=11, do_constant_folding=True,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print("SAE.onnx file is saved in the current working directory")