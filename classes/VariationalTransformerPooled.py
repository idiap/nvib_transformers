#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch

from classes.VariationalTransformerVariable import VariationalTransformerVariable


class VariationalTransformerPooled(VariationalTransformerVariable):
    """
    A Daughter class of Variational Transformer.
    A VAE using transformer encoder and decoder pooled to fixed dimension size
    """

    def __init__(self, vocab_size, args):
        super().__init__(vocab_size, args)

        # Layers for parameters
        self.POOLING = args.POOLING  # Pooling string ["CLS", "max", "mean"]

    def mean_pooling(self, encoder_output, src_key_padding_mask):
        """
        Mean pool over sentence length dimension Ns

        :param encoder_output: Transformer output [Ns,B,H]
        :param src_key_padding_mask: [B, Ns]
        :return: Mean pooled output [1,B,H]
        """

        # Transform mask
        mask = src_key_padding_mask.T.unsqueeze(-1)  # [Ns,B,1]

        # Mask the embeddings and sum (broadcasted across H)
        sum_over_sentences = torch.sum(encoder_output.masked_fill(mask, 0), 0).unsqueeze(
            0
        )  # [1,B,H]

        # A count of the lengths
        sentence_count = torch.sum(~mask, 0).unsqueeze(0)  # [1,B,1]

        # Mean calculation (sum/length)
        return torch.div(sum_over_sentences, sentence_count)  # [Ns,B,1]

    def max_pooling(self, encoder_output, src_key_padding_mask):
        """
        Max pool over sentence length dimension Ns

        :param encoder_output: Transformer output [Ns,B,H]
        :param src_key_padding_mask: [B, Ns]
        :return: Max pooled output [1,B,H]
        """

        # Restructure mask
        mask = src_key_padding_mask.T.unsqueeze(-1)  # [Ns,B,1]

        # mask the embeddings
        encoder_output.masked_fill_(mask, float("-inf"))

        # Get into convolution structure channels =1
        # N = B, C = 1, H = Ns, W=H
        encoder_output = encoder_output.transpose(0, 1).unsqueeze(1)  # [B,1,Ns,H]

        # Max pool (N,C,H,W) - over H which is sequence length
        m = torch.nn.MaxPool2d((encoder_output.size(2), 1))
        pooled = m(encoder_output)  # [B,1,1,H]

        return pooled.squeeze(1).transpose(0, 1)  # [1,B,H]

    def sample(self, number_samples, max_length, min_length, device, *args, **kwargs):
        """
        :param number_samples: This is like batch size
        :param max_length: Uniform samping max
        :param min_length: Uniform sampling min
        :param device: device
        :param args:
        :param kwargs: kwargs["sentence_length_distribution"]
        :return: latent_output dict
        Pools to a single sized dimension so sampling from a distribution for length is irrelevant
        """

        # Sample from a gaussian
        eps = torch.randn(size=(1, number_samples, self.args.DIM_H), device=device)
        z = self.prior_mu + (self.prior_var**0.5) * eps
        mask = torch.zeros(number_samples, dtype=torch.bool, device=device).unsqueeze(-1)  # [B,1]

        return {"z": z, "memory_key_padding_mask": mask}

    def latent_layer(self, encoder_output, src_key_padding_mask):
        """
        Latent layer for fixed dimensional VAE

        :param encoder_output: encoder bov output [Ns,B,H]
        :param src_key_padding_mask: Trues where to mask [B,Ns] (typically encoder mask)
        :return: Z from the latent layer [Nl,B,H]
        """

        # Pooling switch
        if self.POOLING == "max":
            # Max pooling
            enc_out_fixed = self.max_pooling(encoder_output, src_key_padding_mask)  # [Nl=1,B,H]

        else:
            # Mean pooling (default)
            enc_out_fixed = self.mean_pooling(encoder_output, src_key_padding_mask)  # [Nl=1,B,H]

        # Project to mean and log variance
        mu = self.mu_proj(enc_out_fixed)  # [Nl=1,B,H]
        logvar = self.logvar_proj(enc_out_fixed)  # [Nl=1,B,H]
        # Reparameterise
        z = self.reparameterize_gaussian(mu, logvar)  # [Nl=1,B,H]

        # Mask of correct dimension all Falses
        mask = torch.zeros(
            src_key_padding_mask.size(0), dtype=torch.bool, device=encoder_output.device
        ).unsqueeze(
            -1
        )  # [B,1]

        return {"z": z, "memory_key_padding_mask": mask, "mu": mu, "logvar": logvar}
