#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch

from classes.VariationalTransformerVariable import VariationalTransformerVariable
from utils import mask_from_length


def make_stride_mask(mask, perc):
    index = torch.arange(0, mask.size(1), device=mask.device)
    if perc == 0:
        stride_mask = mask
    elif perc <= 0.5:
        mod = int(1 / perc)
        index = index[(index + 1) % mod == 0]
        stride_mask = mask.index_fill(1, index, True)
    elif perc < 1:
        stride = int(1 / (1 - perc))
        index = index[0::stride]
        stride_mask = (~mask).index_fill(1, index, False)
        stride_mask.masked_fill_(mask, True)
    else:
        index = index[1:]
        stride_mask = mask.index_fill(1, index, True)

    return stride_mask


class VariationalTransformerStride(VariationalTransformerVariable):
    """
    A Daughter class of VariationalTransformerVariable. It masks out a percentage of the latent variables
    """

    def __init__(self, vocab_size, args):
        super().__init__(vocab_size, args)
        self.args.STRIDE_PERC = float(self.args.STRIDE_PERC)

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
        src_key_padding_mask = mask_from_length(max_length, lengths)  # [B, max_length]
        # Stride mask
        stride_percentage = self.args.STRIDE_PERC
        stride_mask = make_stride_mask(src_key_padding_mask, stride_percentage)
        memory_key_padding_mask = src_key_padding_mask + stride_mask

        # Sample from a gaussian
        eps = torch.randn(
            size=(max_length, number_samples, self.args.DIM_H), device=device
        )  # [Ns,B,H]
        z = self.prior_mu + (self.prior_var**0.5) * eps
        z.masked_fill_(memory_key_padding_mask.T.unsqueeze(-1), 0)

        return {"z": z, "memory_key_padding_mask": memory_key_padding_mask}

    def latent_layer(self, encoder_output, src_key_padding_mask):
        """
        Latent layer for variable dimensional VAE transformer with a stride mask

        :param encoder_output: encoder bov output [Ns,B,H]
        :param src_key_padding_mask: Trues where to mask [B,Nl] (typically encoder mask)
        :return: Z from the latent layer [Nl,B,H]
        """

        # Stride mask
        stride_percentage = self.args.STRIDE_PERC
        stride_mask = make_stride_mask(src_key_padding_mask, stride_percentage)
        memory_key_padding_mask = src_key_padding_mask + stride_mask

        # Project to mean and log variance
        mu = self.mu_proj(encoder_output)  # [Nl,B,H]
        logvar = self.logvar_proj(encoder_output)  # [Nl,B,H]
        # Reparameterise
        z = self.reparameterize_gaussian(mu, logvar)  # [Nl,B,H]

        # Mask parameters for correct KL calculation
        mask = memory_key_padding_mask.T.unsqueeze(-1)  # [Nl,B,1]

        # Mask the parameters
        mu.masked_fill_(mask, 0)
        logvar.masked_fill_(mask, 0)
        z.masked_fill_(mask, 0)

        return {
            "z": z,
            "memory_key_padding_mask": memory_key_padding_mask,
            "mu": mu,
            "logvar": logvar,
        }
