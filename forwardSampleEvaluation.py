#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# Forward perpexity
# First it will take in a simple trained model (Vanilla Transformer) that has been trained on training data
# Thereafter it will evaluate on the samples of the generative model

import argparse
import json

from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

# Local modules
from classes.Dataset import Dataset
from classes.Transformer import Transformer
from train import evaluation
from utils import *


def main(args):
    # The vanilla model used to test samples
    OUTPUT_PATH = os.path.join(args.OUTPUT_DIR, args.PROJECT_NAME, args.LANGUAGE_MODEL)

    # The model's samples we are currently testing
    SAMPLES_PATH = os.path.join(args.OUTPUT_DIR, args.PROJECT_NAME, args.EXPERIMENT_NAME)

    # Get config for the vanilla language model
    with open(os.path.join(OUTPUT_PATH, "config.json"), "r") as f:
        parser.set_defaults(**json.load(f))
    args = parser.parse_args()
    print(args)

    # Reproducibility function
    DEVICE, g = reproducibility(args.SEED)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Get vocabulary size
    VOCAB_SIZE = tokenizer.vocab_size

    # Define the dataset
    samples_dataset = Dataset(os.path.join(SAMPLES_PATH, "samples.txt"))
    samples_length = get_lengths(SAMPLES_PATH, samples=True)

    # Batch the sample data
    sampleloader = DataLoader(
        samples_dataset,
        batch_size=args.BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=4,
    )

    # Load the model
    model = Transformer(VOCAB_SIZE, args).to(DEVICE)
    LM_checkpoint_dict = load_checkpoint(OUTPUT_PATH, DEVICE, prefix="best")
    model.load_state_dict(LM_checkpoint_dict["model_state_dict"])

    # Get original models wandb ID
    OG_checkpoint_dict = load_checkpoint(SAMPLES_PATH, DEVICE, prefix="checkpoint")
    wandb_id = OG_checkpoint_dict["wandb_id"]

    # WandB
    wandb.init(
        project=args.PROJECT_NAME,
        entity=args.WANDB_ENTITY,
        # config=args,
        id=wandb_id,
        resume="allow",
    )

    model.eval()
    sample_metrics = evaluation(model, sampleloader, tokenizer, DEVICE)

    print("                             Sample Forward                             ")
    wandb_dict = logging(sample_metrics, samples_length, "Forward", wandb_dict={})
    for name, metric in wandb_dict.items():
        wandb.run.summary[name] = metric

    # Save locally
    if not os.path.exists(SAMPLES_PATH + "/forwardSampleResults.csv"):
        save_csv(SAMPLES_PATH + "/forwardSampleResults.csv", wandb_dict)


if __name__ == "__main__":
    # ARGUMENTS
    parser = argparse.ArgumentParser()

    # WandB arguments
    parser.add_argument("--WANDB_ENTITY", default="", help="wandb entity name")

    # Paths
    parser.add_argument(
        "--EXPERIMENT_NAME",
        type=str,
        default="simpleExperiment",
        metavar="N",
        help="Name of current experiment",
    )
    parser.add_argument(
        "--OUTPUT_DIR", type=str, default="outputs", metavar="N", help="Directory for the outputs"
    )

    # WandB arguments
    parser.add_argument("--PROJECT_NAME", default="localExperiments", help="wandb project name")

    # Others
    parser.add_argument("--SEED", type=int, default=42, metavar="N", help="random seed")
    parser.add_argument(
        "--LANGUAGE_MODEL",
        type=str,
        default="vanillaTransformer",
        help="Language model to evaluate samples",
    )

    args = parser.parse_args()

    main(args)
