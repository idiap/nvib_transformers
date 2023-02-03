#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import argparse
import collections
import json
from datetime import datetime, timedelta

import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from classes.Dataset import Dataset
from classes.MetricsAverage import MetricsAverage
from classes.Transformer import Transformer
from classes.VariationalTransformerNVIB import TransformerNVIB
from classes.VariationalTransformerPooled import VariationalTransformerPooled
from classes.VariationalTransformerStride import VariationalTransformerStride
from classes.VariationalTransformerVariable import VariationalTransformerVariable

# Local modules
from utils import *

# Disables the warnings for parallelised tokenisers
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def initialisation(model):
    for name, parm in model.named_parameters():
        if parm.dim() > 1:
            torch.nn.init.xavier_uniform_(parm)
            # Small initial embeddings - https://github.com/BlinkDL/SmallInitEmb
            if name == "embedding.weight":
                torch.nn.init.uniform_(parm, a=-1e-4, b=1e-4)


def evaluation(model, dataloader, tokenizer, device, epoch=None):
    metrics = collections.defaultdict(lambda: MetricsAverage())
    for idx, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            # Tokenise the batches
            (
                encoder_input_ids,
                src_key_padding_mask,
                decoder_input_ids,
                tgt_key_padding_mask,
            ) = tokenize(tokenizer, batch, device)

            # Forward pass
            outputs_dict = model(
                encoder_input_ids,
                decoder_input_ids,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )  # [sq_len, bs, Vocab]

            # Get loss
            loss = model.loss(**outputs_dict, targets=encoder_input_ids)

            # Store summed up losses
            for name, metric in loss.items():
                metrics[name].update(float(metric))
            # Logging
            if "alpha" in outputs_dict.keys():
                metrics["avg_num_vec"].update(float(outputs_dict["avg_num_vec"]))
                metrics["avg_prop_vec"].update(float(outputs_dict["avg_prop_vec"]))
                metrics["avg_alpha0"].update(float(outputs_dict["avg_alpha0"]))
    # Log final batch parms
    if type(model).__name__ != "Transformer":
        temp_mask_1 = outputs_dict["memory_key_padding_mask"].T.unsqueeze(-1)
        temp_mask_full = temp_mask_1.expand(
            outputs_dict["mu"].size(0), outputs_dict["mu"].size(1), outputs_dict["mu"].size(2)
        ).clone()
        if "alpha" in outputs_dict.keys():
            temp_mask_full += ~outputs_dict["alpha"].gt(0)
            wandb.log({"alpha_validation": outputs_dict["alpha"][~temp_mask_1]}, step=epoch)
        if "mu" in outputs_dict.keys():
            wandb.log({"mu_validation": outputs_dict["mu"][~temp_mask_full]}, step=epoch)
        if "logvar" in outputs_dict.keys():
            wandb.log(
                {"var_validation": torch.exp(outputs_dict["logvar"][~temp_mask_full])}, step=epoch
            )

    return metrics


def training(model, OPTIMIZER, trainloader, tokenizer, epoch, device, accumulation_steps):
    # Batches
    for idx, batch in enumerate(tqdm(trainloader)):
        # Tokenise the batches
        (
            encoder_input_ids,
            src_key_padding_mask,
            decoder_input_ids,
            tgt_key_padding_mask,
        ) = tokenize(tokenizer, batch, device)

        # Forward pass
        train_outputs_dict = model(
            encoder_input_ids,
            decoder_input_ids,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # [sq_len, bs, Vocab]

        # Get loss
        train_losses = model.loss(**train_outputs_dict, targets=encoder_input_ids)

        # Accumulate gradients
        (train_losses["Loss"] / accumulation_steps).backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        # Gradient accumulation
        if ((idx + 1) % accumulation_steps == 0) or (idx + 1 == len(trainloader)):
            OPTIMIZER.step()
            OPTIMIZER.zero_grad()

    # Log final batch parms
    if type(model).__name__ != "Transformer":
        temp_mask_1 = train_outputs_dict["memory_key_padding_mask"].T.unsqueeze(-1)
        temp_mask_full = temp_mask_1.expand(
            train_outputs_dict["mu"].size(0),
            train_outputs_dict["mu"].size(1),
            train_outputs_dict["mu"].size(2),
        ).clone()
        if "alpha" in train_outputs_dict.keys():
            temp_mask_full += ~train_outputs_dict["alpha"].gt(0)
            wandb.log({"alpha_train": train_outputs_dict["alpha"][~temp_mask_1]}, step=epoch)
        if "mu" in train_outputs_dict.keys():
            wandb.log({"mu_train": train_outputs_dict["mu"][~temp_mask_full]}, step=epoch)
        if "logvar" in train_outputs_dict.keys():
            wandb.log(
                {"var_train": torch.exp(train_outputs_dict["logvar"][~temp_mask_full])}, step=epoch
            )


def kl_annealing(start=0, stop=1, n_epoch=30, type="constant", n_cycle=4, ratio=0.5):
    """
    Cyclic and monotonic cosine KL annealing from:
    https://github.com/haofuml/cyclical_annealing/blob/6ef4ebabb631df696cf4bfc333a965283eba1958/plot/plot_schedules.ipynb

    :param start:0
    :param stop:1
    :param n_epoch:Total epochs
    :param type: Type of annealing "constant", "monotonic" or "cyclic"
    :param n_cycle:
    :param ratio:
    :return: a list of all factors
    """
    L = np.ones(n_epoch)
    if type != "constant":
        if type == "monotonic":
            n_cycle = 1
            ratio = 0.25

        period = n_epoch / n_cycle
        step = (stop - start) / (period * ratio)

        for c in range(n_cycle):

            v, i = start, 0
            while v <= stop:
                L[int(i + c * period)] = 0.5 - 0.5 * math.cos(v * math.pi)
                v += step
                i += 1
    return L


def main(args):
    # GLOBAL VARIABLES
    START_TIME = datetime.now().replace(microsecond=0)
    DELTA_TIME = timedelta(hours=2, minutes=40)  # Short GPUs cap is 3hrs
    # DELTA_TIME = timedelta(hours=200)  # Long
    END_TIME = START_TIME + DELTA_TIME
    OUTPUT_PATH = os.path.join(args.OUTPUT_DIR, args.PROJECT_NAME, args.EXPERIMENT_NAME)
    DATA_PATH = os.path.join(args.DATA_DIR, args.DATA_NAME)
    CHECKPOINT_MODEL_PATH = os.path.join(OUTPUT_PATH, "checkpoint_model.pt")
    CONFIG_PATH = os.path.join(OUTPUT_PATH, "config.json")
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Reproducibility function
    DEVICE, g = reproducibility(args.SEED)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Get vocabulary size
    VOCAB_SIZE = tokenizer.vocab_size

    # Define the datasets
    train_dataset = Dataset(os.path.join(DATA_PATH, "train.csv"))
    validation_dataset = Dataset(os.path.join(DATA_PATH, "validation.csv"))
    train_length, validation_length, _ = get_lengths(DATA_PATH)

    # Batch the training data
    trainloader = DataLoader(
        train_dataset,
        batch_size=args.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=4,
    )

    # Batch the validation data
    validationloader = DataLoader(
        validation_dataset,
        batch_size=args.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=4,
    )

    # KL ANNEALING
    KL_ANNEALING_FACTOR_GAUSSIAN_LIST = kl_annealing(
        n_epoch=args.EPOCHS, type=args.KL_ANNEALING_GAUSSIAN
    )
    KL_ANNEALING_FACTOR_DIRICHLET_LIST = kl_annealing(
        n_epoch=args.EPOCHS, type=args.KL_ANNEALING_DIRICHLET
    )

    # Load the model
    model = {
        "T": Transformer,
        "VTV": VariationalTransformerVariable,
        "VTP": VariationalTransformerPooled,
        "VTS": VariationalTransformerStride,
        "NVIB": TransformerNVIB,
    }[args.MODEL](VOCAB_SIZE, args).to(DEVICE)

    # Initialisation of parameters
    initialisation(model)

    # Optimiser
    OPTIMIZER = optim.Adam(model.parameters(), lr=args.LEARNING_RATE)

    # Load checkpoint
    if os.path.exists(CHECKPOINT_MODEL_PATH):
        checkpoint_dict = load_checkpoint(OUTPUT_PATH, DEVICE, prefix="checkpoint")

        model.load_state_dict(checkpoint_dict["model_state_dict"])
        OPTIMIZER.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        current_epoch = checkpoint_dict["current_epoch"]
        best_validation_loss = checkpoint_dict["best_validation_loss"]
        wandb_id = checkpoint_dict["wandb_id"]

    # Create checkpoint
    else:
        current_epoch = 1
        best_validation_loss = float("inf")
        wandb_id = None

        # Create config
        with open(CONFIG_PATH, "w") as f:
            json.dump(args.__dict__, f, indent=2)

    # WandB
    wandb.init(
        project=args.PROJECT_NAME,
        entity=args.WANDB_ENTITY,
        config=args,
        id=wandb_id,
        resume="allow",
    )
    wandb_id = wandb.run.id
    wandb.watch(model, log_freq=1, log=None)  # log="all" , None , "gradients"

    print("---------------------------- The model ----------------------------")
    print(model)
    print("Model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Training
    # torch.autograd.set_detect_anomaly(True) # for debugging
    for epoch in tqdm(range(current_epoch, args.EPOCHS + 1)):

        print(
            "---------------------------- Epoch {} {} ----------------------------".format(
                epoch, (datetime.now().replace(microsecond=0) - START_TIME)
            )
        )
        # KL ANNEALING
        model.args.KL_ANNEALING_FACTOR_GAUSSIAN = KL_ANNEALING_FACTOR_GAUSSIAN_LIST[epoch - 1]
        model.args.KL_ANNEALING_FACTOR_DIRICHLET = KL_ANNEALING_FACTOR_DIRICHLET_LIST[epoch - 1]

        # Training
        model.train()
        training(model, OPTIMIZER, trainloader, tokenizer, epoch, DEVICE, args.ACCUMULATION_STEPS)

        # Training loss
        model.eval()
        # training_metrics = evaluation(model,
        #                               trainloader,
        #                               tokenizer,
        #                               DEVICE)
        # Validation loss
        validation_metrics = evaluation(model, validationloader, tokenizer, DEVICE, epoch)

        # Logging + Checkpoint
        if datetime.now() < END_TIME:
            # Logging
            wandb_dict = {
                "Epoch": epoch,
                "Learning Rate": OPTIMIZER.param_groups[0]["lr"],
                "KLg": args.KL_GAUSSIAN_LAMBDA * model.args.KL_ANNEALING_FACTOR_GAUSSIAN,
                "KLd": args.KL_DIRICHLET_LAMBDA * model.args.KL_ANNEALING_FACTOR_DIRICHLET,
            }
            # wandb_dict = logging(training_metrics, train_length, "Training", wandb_dict)
            wandb_dict = logging(validation_metrics, validation_length, "Validation", wandb_dict)

            # Save model - Best average of all batch average losses
            current_validation_loss = validation_metrics["Loss"].avg
            if current_validation_loss < best_validation_loss:
                print(
                    "                             Saving best model                             "
                )
                best_validation_loss = current_validation_loss
                wandb.run.summary["Best_Loss_Validation"] = best_validation_loss

                # Update the local save
                save_checkpoint(
                    OUTPUT_PATH,
                    checkpoint_dict={
                        "Best_Loss_Validation": best_validation_loss,
                        "model_state_dict": model.state_dict(),
                        "Best_Epoch": epoch,
                        "wandb_id": wandb_id,
                    },
                    prefix="best",
                )

            # WandB logging
            wandb.log(wandb_dict, step=epoch)

            # Checkpoint
            save_checkpoint(
                OUTPUT_PATH,
                checkpoint_dict={
                    "current_epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": OPTIMIZER.state_dict(),
                    # 'current_loss': training_metrics["loss"].avg,
                    "best_validation_loss": best_validation_loss,
                    "wandb_id": wandb_id,
                },
                prefix="checkpoint",
            )

    print("---------------------------- Fin ----------------------------")


if __name__ == "__main__":
    # ARGUMENTS
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument(
        "--EXPERIMENT_NAME",
        type=str,
        default="vanillaTransformer",
        metavar="N",
        help="Name of current experiment",
    )
    parser.add_argument(
        "--OUTPUT_DIR", type=str, default="outputs", metavar="N", help="Directory for the outputs"
    )
    parser.add_argument(
        "--DATA_DIR", type=str, default="data", metavar="N", help="Directory for the data"
    )
    parser.add_argument(
        "--DATA_NAME",
        type=str,
        default="wikitext2",
        metavar="N",
        help="Sub directory for the data if its a specific subset or dataset",
    )

    # WandB arguments
    parser.add_argument("--WANDB_ENTITY", default="", help="wandb entity name")
    parser.add_argument("--PROJECT_NAME", default="localExperiments", help="wandb project name")

    # Architecture arguments
    parser.add_argument(
        "--MODEL", default="T", choices=["T", "VTV", "VTP", "VTS", "NVIB"], help="Select a model"
    )
    parser.add_argument(
        "--DIM_H", type=int, default=256, metavar="D", help="dimension of model per layer"
    )
    parser.add_argument(
        "--NUM_HEADS",
        type=int,
        default=1,
        metavar="N",
        help="Number of heads for transformer encoder and decoder",
    )
    parser.add_argument("--NUM_LAYERS", type=int, default=1, metavar="N", help="number of layers")

    # Training arguments
    parser.add_argument("--SEED", type=int, default=42, metavar="N", help="random seed")
    parser.add_argument(
        "--DROPOUT",
        type=float,
        default=0.1,
        metavar="DROP",
        help="dropout probability (0 = no dropout)",
    )
    parser.add_argument(
        "--LEARNING_RATE", type=float, default=1e-4, metavar="LR", help="learning rate"
    )
    parser.add_argument(
        "--EPOCHS", type=int, default=3, metavar="N", help="number of training epochs"
    )
    parser.add_argument("--BATCH_SIZE", type=int, default=128, metavar="N", help="batch size")
    parser.add_argument(
        "--ACCUMULATION_STEPS",
        type=int,
        default=1,
        metavar="N",
        help="Number of steps for gradient",
    )

    # VT models
    parser.add_argument(
        "--KL_GAUSSIAN_LAMBDA",
        type=float,
        default=0,
        metavar="N",
        help="Weight for gaussian kl loss",
    )
    parser.add_argument(
        "--KL_ANNEALING_GAUSSIAN",
        metavar="N",
        default="constant",
        choices=["constant", "monotonic", "cyclic"],
        help="The KL annealing style",
    )
    # VT-pooling models
    parser.add_argument(
        "--POOLING", default="mean", choices=["mean", "max"], help="Select a pooling method"
    )
    # VT-Stride model
    parser.add_argument(
        "--STRIDE_PERC", default=0, metavar="N", help="Percentage of vectors to drop for VT stride"
    )
    # NVIB models
    parser.add_argument(
        "--PRIOR_MU", default=0, type=float, metavar="N", help="Prior for unknown mean"
    )
    parser.add_argument(
        "--PRIOR_VAR", default=1, type=float, metavar="N", help="Prior for unknown variance"
    )
    parser.add_argument(
        "--PRIOR_ALPHA", default=1, type=float, metavar="N", help="Prior for unknown alpha"
    )
    parser.add_argument(
        "--DELTA", default=1, type=float, metavar="N", help="Conditional prior for alpha"
    )
    parser.add_argument("--KAPPA", default=1, type=int, metavar="N", help="number of samples")
    parser.add_argument(
        "--KL_DIRICHLET_LAMBDA",
        type=float,
        default=0,
        metavar="N",
        help="Weight for dirichlet kl loss",
    )
    parser.add_argument(
        "--KL_ANNEALING_DIRICHLET",
        metavar="N",
        default="constant",
        choices=["constant", "monotonic", "cyclic"],
        help="The multiplicative factor for KL annealing",
    )

    args = parser.parse_args()

    main(args)
