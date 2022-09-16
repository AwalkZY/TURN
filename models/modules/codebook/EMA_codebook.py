import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class MultiVectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, group_num, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        assert embedding_dim % group_num == 0
        self.group_dim = embedding_dim // group_num
        self.group_num = group_num
        self.codebooks = nn.ModuleList([VectorQuantizerEMA(num_embeddings, self.group_dim,
                                                           commitment_cost, decay, epsilon)] * group_num)

    def forward(self, x):
        # x in [B, D]
        blocks = x.split(self.group_dim, dim=-1)  # G * [B, D // G]
        final_x, final_loss, final_perplexity = [], [], []
        for (block, codebook) in zip(blocks, self.codebooks):
            quantized, loss, perplexity = codebook(block)
            final_x.append(quantized)
            final_loss.append(loss)
            final_perplexity.append(perplexity)
        if self.training:
            return (torch.cat(final_x, dim=-1), sum(final_loss) / self.group_num,
                    sum(final_perplexity) / self.group_num)
        else:
            return torch.cat(final_x, dim=-1), None, None


class ExponentialMovingAverage(nn.Module):
    """Maintains an exponential moving average for a value.

      This module keeps track of a hidden exponential moving average that is
      initialized as a vector of zeros which is then normalized to give the average.
      This gives us a moving average which isn't biased towards either zero or the
      initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)

      Initially:
          hidden_0 = 0
      Then iteratively:
          hidden_i = hidden_{i-1} - (hidden_{i-1} - value) * (1 - decay)
          average_i = hidden_i / (1 - decay^i)
    """

    def __init__(self, init_value, decay):
        super().__init__()

        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros_like(init_value))

    def forward(self, value):
        self.counter += 1
        self.hidden.sub_((self.hidden - value) * (1 - self.decay))
        average = self.hidden / (1 - self.decay ** self.counter)
        return average


class VectorQuantizerEMA(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. Use EMA to update embeddings.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
        decay (float): decay for the moving averages.
        epsilon (float): small float constant to avoid numerical instability.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon

        # initialize embeddings as buffers
        embeddings = torch.randn(self.num_embeddings, self.embedding_dim)
        # nn.init.xavier_uniform_(embeddings)
        self.register_buffer("embeddings", embeddings)
        self.ema_dw = ExponentialMovingAverage(self.embeddings, decay)

        # also maintain ema_cluster_size， which record the size of each embedding
        self.ema_cluster_size = ExponentialMovingAverage(torch.zeros((self.num_embeddings,)), decay)

    def forward(self, x):
        # x in [B, D] or [B, L, D]
        flat_x = x.reshape(-1, self.embedding_dim)

        encoding_indices, probs = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices).reshape_as(x)

        if not self.training:
            return quantized, None, None

        # update embeddings with EMA
        with torch.no_grad():
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            updated_ema_cluster_size = self.ema_cluster_size(torch.sum(encodings, dim=0))
            n = torch.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                        (n + self.num_embeddings * self.epsilon) * n)
            dw = torch.matmul(encodings.t(), flat_x)  # sum encoding vectors of each cluster
            updated_ema_dw = self.ema_dw(dw)
            normalised_updated_ema_w = (
                    updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1))
            self.embeddings.data = normalised_updated_ema_w

        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-6)))
        return quantized, loss, perplexity

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(self.embeddings ** 2, dim=1) -
                2. * torch.matmul(flat_x, self.embeddings.t())
        )  # [N, M]
        prob = (-distances).softmax(-1)
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices, prob

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.embedding(encoding_indices, self.embeddings)
