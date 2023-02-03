#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch.utils.data as data


class Dataset(data.Dataset):
    """
    Dataset class to read in for the dataloader.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.sents = []

        with open(self.data_path, "r") as fdata:
            for row in fdata:
                # Remove new line chars
                row = row.strip("\n")
                self.sents.append(row)
            self.sents = list(self.sents)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        sent = self.sents[idx]
        return sent
