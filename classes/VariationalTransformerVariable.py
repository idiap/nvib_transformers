#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch
import torch.nn as nn

from classes.Transformer import Transformer
from utils import mask_from_length


class VariationalTransformerVariable(Transformer):
    """
    A Daughter class of Transformer.
    A VAE using transformer encoder and decoder
    """

    def __init__(self, vocab_size, args):
        super().__init__(vocab_size, args)

        self.prior_mu = float(args.PRIOR_MU)
        self.prior_var = float(args.PRIOR_VAR)

        # Layers for parameters
        self.mu_proj = nn.Linear(args.DIM_H, args.DIM_H)
        self.logvar_proj = nn.Linear(args.DIM_H, args.DIM_H)

    def reparameterize_gaussian(self, mu, logvar):
        """
        Reparameterise for gaussian
        Train = sample
        Test = mean
        Nl = Ns
        :param mu: means [Nl,B,H]
        :param logvar: logged variances [Nl,B,H]
        :return: z: sample from a gaussian distribution or mean
        """
        if self.training:
            std = torch.exp(0.5 * logvar)  # [Nl,B,H]
            eps = torch.randn_like(std)  # [Nl,B,H]
            z = eps.mul(std).add_(mu)  # [Nl,B,H]
        else:
            z = mu  # [Nl,B,H]
        return z  # [Nl,B,H]

    def loss_kl_g(self, mu, logvar, memory_key_padding_mask, **kwargs):
        """
        Returns the KL divergence between our mean and variances and N(0,1)
        Nl =Ns

        :param mu: means - masked [Nl,B,H]
        :param logvar: logged variances - masked [Nl,B,H]
        :param memory_key_padding_mask: mask for latent space [B,Nl]
        :return: KL divergence [B]
        """

        # KL between univariate Gaussians
        var_ratio = logvar.exp() / self.prior_var
        t1 = (mu - self.prior_mu) ** 2 / self.prior_var
        kl = var_ratio + t1 - 1 - var_ratio.log()
        kl = kl.masked_fill(memory_key_padding_mask.T.unsqueeze(-1), 0)

        # Mean over embedding dimension, sum over sentence length
        kl = torch.mean(kl, -1)  # [Nl,B]
        kl = 0.5 * torch.sum(kl, 0)  # [B]

        return kl

    def sample_lengths(
        self,
        number_samples,
        device,
        sentence_length_distribution=None,
        min_length=None,
        max_length=None,
    ):
        """
        Sample the lengths to use.

        :param number_samples: This is like batch size
        :param device: device
        :param sentence_length_distribution: distribution of sentence lengths to sample.
        Dictionary key: length, value: freq
        :param min_length: minimum length to sample
        :param max_length: maximum length to sample
        :return: tensor of lengths [B,max_length]
        """

        # Sample from the training data distribution of lengths
        if sentence_length_distribution is not None:
            print("Sampling from data distribution of lengths")
            sentence_length_distribution = dict(
                sorted(sentence_length_distribution.items())
            )  # Sort by the keys
            length_categories = torch.tensor(
                list(sentence_length_distribution.keys()), device=device
            )
            max_length = max(length_categories)
            frequency_weights = torch.tensor(
                list(sentence_length_distribution.values()), device=device, dtype=float
            )
            # Sample
            sample_idx = torch.multinomial(frequency_weights, number_samples, replacement=True)
            # Plus 2 for the SEP token + Prior
            lengths = torch.tensor(length_categories[sample_idx], device=device) + 2

        # Sample from a uniform distribution between bounds
        else:
            print(
                "Sampling from uniform distribution of lengths U({},{})".format(
                    min_length, max_length
                )
            )
            assert min_length is not None
            assert max_length is not None
            # Between min and max length. Plus 2 for the SEP token + Prior
            lengths = torch.randint(
                min_length + 2, max_length + 2, (number_samples, 1), device=device
            ).squeeze(-1)

        return lengths, max_length

    def sample(self, number_samples, max_length, min_length, device, *args, **kwargs):
        """

        :param number_samples: This is like batch size
        :param min_length: Uniform sampling min
        :param max_length: Uniform samping max
        :param device:
        :param args:
        :param kwargs: kwargs["sentence_length_distribution"]
        :return: latent_output dict
        When not using distribution (given in kwargs["sentence_length_distribution"]) sample uniform with min and max
        """

        # Get length samples (uniform or data distribution)
        lengths, max_length = self.sample_lengths(
            number_samples, device, kwargs["sentence_length_distribution"], min_length, max_length
        )

        # Make mask
        memory_key_padding_mask = mask_from_length(max_length, lengths)  # [B, max_length]

        # Sample from a gaussian
        eps = torch.randn(
            size=(max_length, number_samples, self.args.DIM_H), device=device
        )  # [Ns,B,H]
        z = self.prior_mu + (self.prior_var**0.5) * eps
        z.masked_fill_(memory_key_padding_mask.T.unsqueeze(-1), 0)

        return {"z": z, "memory_key_padding_mask": memory_key_padding_mask}

    def latent_layer(self, encoder_output, src_key_padding_mask):
        """
        Latent layer for variable dimensional VAE transformer

        :param encoder_output: encoder bov output [Ns,B,H]
        :param src_key_padding_mask: Trues where to mask [B,Nl] (typically encoder mask)
        :return: Z from the latent layer [Nl,B,H]
        """

        # Project to mean and log variance
        mu = self.mu_proj(encoder_output)  # [Nl,B,H]
        logvar = self.logvar_proj(encoder_output)  # [Nl,B,H]
        # Reparameterise
        z = self.reparameterize_gaussian(mu, logvar)  # [Nl,B,H]

        # Mask parameters for correct KL calculation
        mask = src_key_padding_mask.T.unsqueeze(-1)  # [Nl,B,1]

        # Mask the parameters
        mu.masked_fill_(mask, 0)
        logvar.masked_fill_(mask, 0)
        z.masked_fill_(mask, 0)

        return {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "memory_key_padding_mask": src_key_padding_mask,
        }

    def loss(self, logits, targets, *args, **kwargs):
        """
        Calculate the loss

        :param logits: output of the decoder [Nt,B,V]
        :param targets: target token ids [B,Nt]
        :return: Dictionary of scalar valued losses. With a value "Loss" to backprop
        """

        # KL loss averaged over batches
        kl_loss = torch.mean(self.loss_kl_g(**kwargs))
        # Cross Entropy where pad = 0
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        # Transform targets
        targets = torch.flatten(targets)  # [B x Nt]
        # Transform vocabulary
        logits = torch.flatten(logits, start_dim=0, end_dim=1)  # [Nt x B, V]
        # Calculates loss over [Nt x B]
        cross_entropy_loss = criterion(logits, targets)  # [Nt x B]
        # Average loss + average KL for backprop and sum loss for logging
        return {
            "Loss": torch.mean(cross_entropy_loss)
            + self.args.KL_GAUSSIAN_LAMBDA * self.args.KL_ANNEALING_FACTOR_GAUSSIAN * kl_loss,
            "CrossEntropy": torch.sum(cross_entropy_loss),
            "KLGaussian": kl_loss,
        }
