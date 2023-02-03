#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch
import torch.nn as nn
from nvib.denoising_attention import DenoisingMultiheadAttention
from nvib.kl import kl_dirichlet, kl_gaussian
from nvib.nvib_layer import Nvib

from classes.Transformer import Transformer


def mask_from_length(max_len, lengths):
    """
    Make a mask from a tensor of lengths where True is to mask
    :param max_len: scalar of max length
    :param lengths: tensor of lengths [Ns]
    :return: boolean mask [Ns,max_length]
    """

    # Make the mask
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return ~mask


class TransformerNVIB(Transformer):
    """
    A Daughter class of Transformer.
    A VAE using transformer encoder and decoder with a non parametric variation information bottleneck
    """

    def __init__(self, vocab_size, args):
        super().__init__(vocab_size, args)

        self.nvib_layer = Nvib(
            size_in=args.DIM_H,
            size_out=args.DIM_H,
            prior_mu=args.PRIOR_MU,
            prior_var=args.PRIOR_VAR,
            prior_alpha=args.PRIOR_ALPHA,
            kappa=args.KAPPA,
            delta=args.DELTA,
        )

        # Overide MHA with DA functio
        for layer_num, layer in enumerate(self.decoder.layers):
            layer.multihead_attn = DenoisingMultiheadAttention(
                embed_dim=args.DIM_H, num_heads=args.NUM_HEADS, dropout=args.DROPOUT, bias=False
            )

    def latent_layer(self, encoder_output, src_key_padding_mask):
        """
        Latent layer for variable dimensional VAE transformer with NVIB

        :param encoder_output: encoder bov output [Ns,B,H]
        :param src_key_padding_mask: Trues where to mask [B,Nl] (typically encoder mask)
        :return: Z from the latent layer [Nl,B,H]
        """
        latent_dict = self.nvib_layer(encoder_output, src_key_padding_mask)

        return {**latent_dict}

    def sample(self, number_samples, max_length, min_length, device, *args, **kwargs):
        """
         Take a sample from the prior distribution and decode it.

         Sampling is done when the model is in evaluation mode (no dropout).
         There is an equivalence between the training and evaluation time attention functions if:
         mu = Z and variance = 0 we get the same function.

         Sample a uniform distribution of the min_length max_length and
        :param number_samples: This is like batch size
        :param min_length: Uniform sampling min
        :param max_length: Uniform samping max
        :param device:
        :param args:
        :param kwargs: kwargs["sentence_length_distribution"]
        :return: latent_output dict
        When not using distribution (given in kwargs["sentence_length_distribution"]) sample uniform with min and max
        """

        if max_length == min_length:
            # Sample from the constant length
            lengths = torch.ones(number_samples, device=device) * max_length
        else:
            # Sample uniform between min and max length. Plus 2 for the initial token + Prior
            lengths = torch.randint(
                min_length + 2, max_length + 2, (number_samples, 1), device=device
            ).squeeze(-1)

        # Make mask
        memory_key_padding_mask = mask_from_length(max_length, lengths)  # [B, max_length]

        z, memory_key_padding_mask = self.nvib_layer.sample(
            number_samples, memory_key_padding_mask, device
        )

        return {
            "z": z,
            "memory_key_padding_mask": memory_key_padding_mask,
        }

    def loss(self, logits, targets, **kwargs):
        """
        Calculate the loss

        :param logits: output of the decoder [Nt,B,V]
        :param targets: target token ids [B,Nt]
        :return: Dictionary of scalar valued losses. With a value "Loss" to backprop
        """

        # KL loss averaged over batches
        kl_loss_g = torch.mean(
            kl_gaussian(
                prior_mu=self.args.PRIOR_MU,
                prior_var=self.args.PRIOR_VAR,
                kappa=self.args.KAPPA,
                **kwargs
            )
        )
        kl_loss_d = torch.mean(
            kl_dirichlet(
                prior_alpha=self.args.PRIOR_ALPHA,
                delta=self.args.DELTA,
                kappa=self.args.KAPPA,
                **kwargs
            )
        )

        # Cross Entropy where pad = 0
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        # Transform targets
        targets = torch.flatten(targets)  # [Nt x B]
        # Transform vocabulary
        logits = torch.flatten(logits, start_dim=0, end_dim=1)  # [Nt x B, V]
        # Calculates loss over [Nt x B]
        cross_entropy_loss = criterion(logits, targets)  # [Nt x B]
        # Average loss + average KL for backprop and sum loss for logging
        return {
            "Loss": torch.mean(cross_entropy_loss)
            + self.args.KL_GAUSSIAN_LAMBDA * self.args.KL_ANNEALING_FACTOR_GAUSSIAN * kl_loss_g
            + self.args.KL_DIRICHLET_LAMBDA * self.args.KL_ANNEALING_FACTOR_DIRICHLET * kl_loss_d,
            "CrossEntropy": torch.sum(cross_entropy_loss),
            "KLGaussian": kl_loss_g,
            "KLDirichlet": kl_loss_d,
        }
