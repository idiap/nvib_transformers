#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# Class for storing the average of a variable


class MetricsAverage(object):
    """
    Keeping track of metrics while training
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.cnt = 0
        self.sum = 0
        self.avg = 0

    def update(self, val, n=1):
        self.cnt += n
        self.sum += val * n
        self.avg = self.sum / self.cnt
