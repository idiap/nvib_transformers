#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import math

import torch
import torch.nn as nn

# Note:
# B: Batch size
# Ns: Source length
# Nt: Target length
# Nl: Latent length (typically = Ns)
# E: Embedding dimension
# H: Model dimension
# V: Vocab dimension


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=1000, mul_by_sqrt=True):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.mul_by_sqrt = mul_by_sqrt

    def forward(self, x):
        x = x.permute(1, 0, 2)
        if self.mul_by_sqrt:
            x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pe = self.pe[:, 1 : seq_len + 1]
        pe = pe.expand_as(x)
        x = x + pe
        x = x.permute(1, 0, 2)
        return x


class Transformer(nn.Transformer):
    """
    A vanilla Transformer Encoder-Decoder in Pytorch

    Data format:
    SRC: ... [EOS]
    TGT: ... [EOS]
    Encoder_input(SRC): ... [EOS]
    Decoder_input(TGT): [SOS] ...

    For an autoencoder x -> x (SRC = TGT)
        The loss function requires SRC and logits.
    For different models x -> y (Eg: translation SRC != TGT)
        The loss function requires TGT and logits.

    If we keep this format the attention masks for padding are identical for autoencoder's encoder + decoder .
    """

    def __init__(self, vocab_size, args):
        super().__init__(
            d_model=args.DIM_H,
            nhead=args.NUM_HEADS,
            num_encoder_layers=args.NUM_LAYERS,
            num_decoder_layers=args.NUM_LAYERS,
            dim_feedforward=4 * args.DIM_H,
            dropout=args.DROPOUT,
            batch_first=False,
            norm_first=False,
        )
        self.args = args
        self.embedding = nn.Embedding(vocab_size, args.DIM_H, padding_idx=0)
        self.positional_encoding = PositionalEncoding(args.DIM_H)
        self.output_proj = nn.Linear(args.DIM_H, vocab_size)
        self.drop = nn.Dropout(args.DROPOUT)

    def encode(self, src, src_key_padding_mask):
        """
        Encode the input ids to embeddings and then pass to the transformer encoder
        :param src: source token ids [Ns, B]
        :param src_key_padding_mask: Trues where to mask [B,Ns]
        :return: memory: [Ns,B,H]
        """
        # Add position encodings + Embeddings
        src = self.positional_encoding(self.drop(self.embedding(src)))  # [Ns,B,H]

        # Transformer encoder
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)  # [Ns,B,H]
        return memory

    def latent_layer(self, encoder_output, src_key_padding_mask):
        """
        Latent layer for child classes like VAE

        :param encoder_output: encoder bov output [Ns,B,H]
        :param src_key_padding_mask: Trues where to mask [B,Nl] (typically encoder mask)
        :return: Z from the latent layer [Nl,B,H]
        """
        z = encoder_output  # [Ns,B,H]
        return {"z": z, "memory_key_padding_mask": src_key_padding_mask}  # [B,Nl]

    def decode(self, tgt, z, memory_key_padding_mask, tgt_key_padding_mask, *args, **kwargs):
        """

        :param tgt: target token ids [Nt,B]
        :param z: output from the latent layer [Nl,B,H]
        :param memory_key_padding_mask: mask for latent layer [B, Nl] (typically Ns = Nl)
        :param tgt_key_padding_mask: target mask [B,Nt]
        :param args:
        :param kwargs:
        :return: logits over the vocabulary [Nt,B,V]
        """

        # Add position encodings + Embeddings
        tgt = self.positional_encoding(self.drop(self.embedding(tgt)))  # [Nt,B,H]
        # Generate target teacher forcing mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(
            tgt.device
        )  # [Nt, Nt]
        output = self.decoder(
            tgt=tgt,  # [Nt,B,H]
            memory=z,  # [Nt,B,H]
            tgt_mask=tgt_mask,  # [Nt,Nt]
            tgt_key_padding_mask=tgt_key_padding_mask,  # [B,Nt]
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [B,Nl]
        logits = self.output_proj(output)  # [Nt,B,V]
        return logits

    def generate(self, z, memory_key_padding_mask, max_len, tokenizer, *args, **kwargs):
        """
        Generate autoregressively without teacher forcing
        :param z: output from the latent layer [Nl,B,H]
        :param memory_key_padding_mask: mask from the latent layer [B,Nl]
        :param max_len: maximum generation length
        :param tokenizer: tokenizer
        :param args:
        :param kwargs:
        :return: logits [Nt,B,V] and list of predictions
        """

        # Initialise target ids with BOS token
        target_ids = (
            torch.tensor([[tokenizer.cls_token_id]])
            .expand(memory_key_padding_mask.shape[0], -1)
            .T.to(memory_key_padding_mask.device)
        )  # [1, B]
        # For each token in length
        for token_idx in range(max_len):
            # Decode the target ids regressively
            logits = self.decode(target_ids, z, memory_key_padding_mask, None)  # [token_idx, B, V]
            # Select only the final set of logits
            prediction = logits[-1, :, :].unsqueeze(0)  # [target_ids1,B,V]
            # Get prediction over vocabulary and return index
            prediction = prediction.argmax(-1)  # [1,B]
            # Concatenate the predictions to form next token_ids
            target_ids = torch.cat((target_ids, prediction), dim=0)  # [token_index, B]

        # Decode into a sentence
        predictions = [tokenizer.decode(encoded) for encoded in target_ids[1:, :].T]  # list [B]
        return logits, predictions

    def loss(self, logits, targets, *args, **kwargs):
        """
        Calculate the loss

        :param logits: output of the decoder [Nt,B,V]
        :param targets: target token ids [Nt, B]
        :return: Dictionary of scalar valued losses. With a value "Loss" to backprop averaged over batches.
        This is important as then the gradients are not dependent on B. However, want to log the loss over all data
        so we shouldn't average over batches as the average of averages is not the same thing when batches can be different sizes!
        https://lemire.me/blog/2005/10/28/average-of-averages-is-not-the-average/#:~:text=The%20average%20of%20averages%20is%20not%20the%20average,-A%20fact%20that&text=In%20fancy%20terms%2C%20the%20average,3

        """

        # Cross Entropy where pad = 0
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        # Transform targets
        targets = torch.flatten(targets)  # [Nt x B]
        # Transform vocabulary
        logits = torch.flatten(logits, start_dim=0, end_dim=1)  # [Nt x B, V]
        # Calculates loss and returns [Nt x B]
        cross_entropy_loss = criterion(logits, targets)  # [Nt x B]
        # Average loss for backprop and sum loss for logging
        return {
            "Loss": torch.mean(cross_entropy_loss),
            "CrossEntropy": torch.sum(cross_entropy_loss),
        }

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        memory_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Forward pass for all transformer models

        :param src: the sequence to the encoder (required). [Ns,B]
        :param tgt: the sequence  nce to the decoder (required). [Nt,B]
        :param src_mask: the additive mask for the src sequence (optional). [Ns, Ns]
        :param tgt_mask: the additive mask for the tgt sequence (optional). [Nt, Nt]
        :param memory_mask: the additive mask for the encoder output (optional). [Nt,Ns]
        :param src_key_padding_mask: the ByteTensor mask for src keys per batch (optional). [B,Ns]
        :param tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional). [B,Nt]
        :param memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).[B,Nl]
        :return: logits and latent dimension dictionary

        Check out here for more info masks on https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask
        The memory ones are interesting. I use memory_key_padding_mask to mask the tokens in the latent space.

        """

        # Encode
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)  # [Ns,B,H]
        # latent layer
        latent_output_dict = self.latent_layer(memory, src_key_padding_mask)
        # Decode
        output = self.decode(
            tgt=tgt,  # [Nt,B,H]
            z=latent_output_dict["z"],  # [Nl,B,H]
            tgt_key_padding_mask=tgt_key_padding_mask,  # [B,Nt]
            memory_key_padding_mask=latent_output_dict["memory_key_padding_mask"],
        )  # [B,Nl]

        return {
            "logits": output,  # [Nt, B, V]
            **latent_output_dict,
        }
