#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import csv
import math
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch

import wandb


def seed_worker(worker_id):
    """
    Set seeds for the dataloaders
    :param worker_id:
    :return:
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def reproducibility(SEED):
    """
    Set all seeds
    :param SEED: int
    :return:
    """
    # Reproducability seeds + device
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        DEVICE = torch.device("cpu")

    # Generator for dataloader
    g = torch.Generator()
    g.manual_seed(SEED)

    return DEVICE, g


def save_checkpoint(path, checkpoint_dict, prefix=""):
    """
    Saving Checkpoint
    """
    print(
        "                             Saving {} model                             ".format(prefix)
    )
    torch.save(checkpoint_dict, os.path.join(path, "{}_model.pt".format(prefix)))


def load_checkpoint(path, DEVICE, prefix=""):
    """
    Loading the checkpoint
    """

    print(
        "---------------------------- Loading {} model ----------------------------".format(prefix)
    )
    checkpoint_dict = torch.load(
        os.path.join(path, "{}_model.pt".format(prefix)), map_location=DEVICE
    )
    return checkpoint_dict


def tokenize(tokenizer, batch, DEVICE):
    """
    Take a list batch of sentences and return tokenized, padded and the correct start of sequence and end of sequence tokens
    :param tokenizer: tokenizer
    :param batch: a list of lists of sentences
    :param DEVICE: device (cpu or gput)
    :return: encoder inputs, decoder inputs and masks
    """
    tokenised_batch = tokenizer(batch, padding=True, return_tensors="pt").to(DEVICE)

    # Strip the BOS
    encoder_input_ids = tokenised_batch["input_ids"][:, 1:]
    src_key_padding_mask = tokenised_batch["attention_mask"][:, 1:] == 0  # [bs, sq_len]

    # Contain BOS and EOS
    decoder_input_ids = tokenised_batch["input_ids"][:, :-1]
    tgt_key_padding_mask = tokenised_batch["attention_mask"][:, :-1] == 0  #

    return encoder_input_ids.T, src_key_padding_mask, decoder_input_ids.T, tgt_key_padding_mask


def strip_sep(sents, tokenizer):
    """
    Strip the tokenizer SEP token from list of lists
    :param sents: list of lists of sentences
    :param tokenizer: tokenizer
    :return: stripped sentences
    """
    return [
        sent[: max((sent.index(tokenizer.sep_token) - 1), 0)]
        if tokenizer.sep_token in sent
        else sent
        for sent in sents
    ]


def write_sent(sents, path):
    """
    write sentences to file
    :param sents: Sentences list of lists
    :param path: path to save
    :return:
    """
    with open(path, "w") as f:
        for s in sents:
            f.write(s + "\n")


def save_csv(csv_file, dict_data):
    """
    save a csv
    :param csv_file: file name
    :param dict_data: dictionary to save
    :return:
    """
    # check if file exists
    csv_columns = dict_data.keys()
    with open(csv_file, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in [dict_data]:
            writer.writerow(data)


def mask_from_length(max_len, lengths):
    """
    create a boolean mask from the lengths and max length
    :param max_len: int
    :param lengths: list of lengths
    :return: boolean torch tensor of masks
    """
    # Make the mask
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return ~mask


def FetchRuns(summary_variables, config_variables, FILENAME, OUTPUT_DIR, PROJECT_NAME, ENTITY):
    """
    Fetch information from runs recorded on wandb

    :param summary_variables: variables from the wandb summary
    :param config_variables: variables from the wandb config
    :param FILENAME: Name of the file you would like save
    :param OUTPUT_DIR: output directy usually "outputs"
    :param PROJECT_NAME: name of the project
    :param ENTITY: wandb entity
    :return:
    """
    FILE = os.path.join(OUTPUT_DIR, PROJECT_NAME, FILENAME)

    if not os.path.exists(FILE):
        runs_path = os.path.join(ENTITY, PROJECT_NAME)

        api = wandb.Api()

        # Project is specified by <entity/project-name>
        runs = api.runs(runs_path)

        summary_list, config_list, name_list = [], [], []
        for run in runs:
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files
            summary_list.append(run.summary._json_dict)

            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

            # .name is the human-readable name of the run.
            name_list.append(run.name)

        runs_df = pd.DataFrame({"summary": summary_list, "config": config_list, "name": name_list})

        df = pd.DataFrame({})
        for index in range(0, len(runs_df["config"])):

            config = runs_df["config"][index]
            summary = runs_df["summary"][index]

            # Retrieve variables of interest
            current_df = {}
            for config_name in config_variables:
                if config_name in config:
                    current_df[config_name] = config.pop(config_name)
                else:
                    current_df[config_name] = ""
            for summary_name in summary_variables:
                if summary_name in summary:
                    current_df[summary_name] = summary.pop(summary_name)
                else:
                    current_df[summary_name] = ""

            current_df = pd.DataFrame(current_df, index=[0])
            df = df.append(current_df)

        df.to_pickle(FILE)
    else:
        df = pd.read_pickle(FILE)

    return df


def get_lengths(path, subset="all", samples=False):
    """
    Get number of tokens in data for perplexity calculations and sampling

    :param path: Path to data
    :param subset: train, test, validation or all
    :param samples: use the samples as it doesnt have the subset
    :return: text dict
    """
    if samples:
        file = open(os.path.join(path, "samples_text_length.pkl"), "rb")
        text_dict = pickle.load(file)
        file.close()
        return text_dict["total_tokens"]
    else:
        file = open(os.path.join(path, "text_length.pkl"), "rb")
        text_dict = pickle.load(file)
        file.close()
        if subset == "train":
            return text_dict["test"]["total_tokens"]
        elif subset == "validation":
            return text_dict["validation"]["total_tokens"]
        elif subset == "test":
            return text_dict["test"]["total_tokens"]
        else:
            return (
                text_dict["train"]["total_tokens"],
                text_dict["validation"]["total_tokens"],
                text_dict["test"]["total_tokens"],
            )


def logging(metrics, total_length, type, wandb_dict):
    """
    Logging parameters and metrics with wandb

    :param metrics: collections.defaultdict of metrics objects
    :param total_length: Total length of data
    :param type: train, validation, test
    :param wandb_dict: wandb updating dictionary
    :return: return the updated dict for logging
    """

    print("                             {}                             ".format(type))
    # Note: latent size is not same size as input thus average over input doesnt make sense for KLs
    for name, metric in metrics.items():
        if name == "Loss":
            pass

        elif name == "CrossEntropy":
            average_metric = metric.sum / total_length
            # Wandb logging
            wandb_dict[name + "_" + type] = average_metric
            wandb_dict["Perplexity" + "_" + type] = math.exp(average_metric)
            # Print loggign
            print("{} {}".format(name, round(average_metric, 2)))
            print("Perplexity ", round(math.exp(average_metric), 2))

        elif name == "KLGaussian":
            # Averaged across batches averaged across all batches
            average_metric = metric.avg
            # Wandb logging
            wandb_dict[name + "_" + type] = average_metric
            # Print logging
            print("{} {}".format(name, round(average_metric, 2)))

        elif name == "KLDirichlet":
            # Averaged across batches averaged across all batches
            average_metric = metric.avg
            # Wandb logging
            wandb_dict[name + "_" + type] = average_metric
            # Print logging
            print("{} {}".format(name, round(average_metric, 2)))

        elif name == "avg_num_vec":
            average_metric = metric.avg
            # Wandb logging
            wandb_dict[name + "_" + type] = average_metric
            # Print logging
            print("{} {}".format(name, round(average_metric, 2)))

        elif name == "avg_prop_vec":
            average_metric = metric.avg
            # Wandb logging
            wandb_dict[name + "_" + type] = average_metric
            # Print logging
            print("{} {}".format(name, round(average_metric, 2)))

        elif name == "avg_alpha0":
            average_metric = metric.avg
            # Wandb logging
            wandb_dict[name + "_" + type] = average_metric
            # Print logging
            print("{} {}".format(name, round(average_metric, 2)))

    return wandb_dict
