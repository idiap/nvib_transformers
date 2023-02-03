#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# Backward perplexity
# We take a simple language model (transformer) and train it from scratch on the samples.
# We then evaluate on validation data.

import argparse
import json
import sys
from datetime import datetime

import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# Local modules
from classes.Dataset import Dataset
from classes.Transformer import Transformer
from train import evaluation, initialisation, training
from utils import *

# Disables the warnings for parallelised tokenisers
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main(args):
    # The vanilla model used to test samples
    OUTPUT_PATH = os.path.join(args.OUTPUT_DIR, args.PROJECT_NAME, args.EXPERIMENT_NAME)
    LM_PATH = os.path.join(args.OUTPUT_DIR, args.PROJECT_NAME, args.LANGUAGE_MODEL)

    # Get config for the vanilla language model
    with open(os.path.join(LM_PATH, "config.json"), "r") as f:
        parser.set_defaults(**json.load(f))
    args = parser.parse_args()
    print(args)

    # GLOBAL VARIABLES
    START_TIME = datetime.now().replace(microsecond=0)
    DATA_PATH = os.path.join(args.DATA_DIR, args.DATA_NAME)
    CONFIG_PATH = os.path.join(OUTPUT_PATH, "reverse_config.json")
    CHECKPOINT_MODEL_PATH = os.path.join(OUTPUT_PATH, "checkpoint_reverse_model.pt")

    # Reproducibility function
    DEVICE, g = reproducibility(args.SEED)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Get vocabulary size
    VOCAB_SIZE = tokenizer.vocab_size

    # Define the datasets
    samples_dataset = Dataset(os.path.join(OUTPUT_PATH, "samples.txt"))
    evaluation_dataset = Dataset(os.path.join(DATA_PATH, args.DATA_SUBSET + ".csv"))
    evaluation_length = get_lengths(DATA_PATH, subset=args.DATA_SUBSET)

    # Batch the training data
    sampleloader = DataLoader(
        samples_dataset,
        batch_size=args.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=4,
    )

    # Batch the validation data
    validationloader = DataLoader(
        evaluation_dataset,
        batch_size=args.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=4,
    )

    # Load a vanilla model
    model = Transformer(VOCAB_SIZE, args).to(DEVICE)
    # Testing
    if args.DATA_SUBSET == "test":
        checkpoint_dict = load_checkpoint(OUTPUT_PATH, DEVICE, prefix="best_reverse")

        model.load_state_dict(checkpoint_dict["model_state_dict"])
        wandb_id = checkpoint_dict["wandb_id"]
        wandb.init(
            project=args.PROJECT_NAME, entity=args.WANDB_ENTITY, id=wandb_id, resume="allow"
        )

        model.eval()
        # Test loss
        test_metrics = evaluation(model, validationloader, tokenizer, DEVICE)
        wandb.run.summary["RPPL_test"] = math.exp(
            test_metrics["CrossEntropy"].sum / evaluation_length
        )
        sys.exit("Test completed")

    # Initialisation
    initialisation(model)

    # Optimiser
    OPTIMIZER = optim.Adam(model.parameters(), lr=args.LEARNING_RATE)

    # Load checkpoint
    if os.path.exists(CHECKPOINT_MODEL_PATH):
        checkpoint_dict = load_checkpoint(OUTPUT_PATH, DEVICE, prefix="checkpoint_reverse")

        model.load_state_dict(checkpoint_dict["model_state_dict"])
        OPTIMIZER.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        current_epoch = checkpoint_dict["current_epoch"]
        best_validation_loss = checkpoint_dict["best_validation_loss"]
        wandb_id = checkpoint_dict["wandb_id"]

    # Create checkpoint
    else:
        current_epoch = 1
        best_validation_loss = float("inf")
        # Get original models wandb ID
        OG_checkpoint_dict = load_checkpoint(OUTPUT_PATH, DEVICE, prefix="checkpoint")
        wandb_id = OG_checkpoint_dict["wandb_id"]

        # Create config
        with open(CONFIG_PATH, "w") as f:
            json.dump(args.__dict__, f, indent=2)

    # WandB
    wandb.init(
        project=args.PROJECT_NAME,
        entity=args.WANDB_ENTITY,
        # config=args,
        id=wandb_id,
        resume="allow",
    )

    print("---------------------------- The model ----------------------------")
    print(model)
    print("Model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Different step to not mess with the other steps
    wandb.define_metric("Epoch_reverse")
    wandb.define_metric("Epoch_reverse", step_metric="Epoch_reverse")
    wandb.define_metric("CrossEntropy_Validation_reverse", step_metric="Epoch_reverse")
    wandb.define_metric("Perplexity_Validation_reverse", step_metric="Epoch_reverse")

    for epoch in tqdm(range(current_epoch, args.EPOCHS + 1)):

        print(
            "---------------------------- Reverse Epoch {} {} ----------------------------".format(
                epoch, (datetime.now().replace(microsecond=0) - START_TIME)
            )
        )
        # Training
        model.train()
        training(model, OPTIMIZER, sampleloader, tokenizer, epoch, DEVICE, args.ACCUMULATION_STEPS)

        # Training loss
        model.eval()
        # Validation loss
        validation_metrics = evaluation(model, validationloader, tokenizer, DEVICE)

        # Logging + Checkpoint
        _ = logging(validation_metrics, evaluation_length, "Reverse", {})

        # Save model
        current_validation_loss = validation_metrics["Loss"].avg
        if current_validation_loss < best_validation_loss:
            print(
                "                             Saving best reverse model                           "
            )
            best_validation_loss = current_validation_loss
            wandb.run.summary["Best_RCE"] = (
                validation_metrics["CrossEntropy"].sum / evaluation_length
            )
            wandb.run.summary["Best_RPPL"] = math.exp(
                validation_metrics["CrossEntropy"].sum / evaluation_length
            )

            # Update the local save
            save_checkpoint(
                OUTPUT_PATH,
                checkpoint_dict={
                    "model_state_dict": model.state_dict(),
                    "Best_Epoch_reverse": epoch,
                    "wandb_id": wandb_id,
                },
                prefix="best_reverse",
            )
        save_checkpoint(
            OUTPUT_PATH,
            checkpoint_dict={
                "current_epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": OPTIMIZER.state_dict(),
                "best_validation_loss": best_validation_loss,
                "wandb_id": wandb_id,
            },
            prefix="checkpoint_reverse",
        )


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
    parser.add_argument(
        "--DATA_SUBSET",
        type=str,
        default="validation",
        metavar="N",
        help="Subset for the data - validation, test",
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
