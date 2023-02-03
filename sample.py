#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# Sample from the models

import argparse
import json

from tqdm import tqdm
from transformers import AutoTokenizer

# Local modules
from classes.Transformer import Transformer
from classes.VariationalTransformerNVIB import TransformerNVIB
from classes.VariationalTransformerPooled import VariationalTransformerPooled
from classes.VariationalTransformerStride import VariationalTransformerStride
from classes.VariationalTransformerVariable import VariationalTransformerVariable
from utils import *

# Disables the warnings for parallelised tokenisers
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def sample(model, tokenizer, OUTPUT_PATH, run, DATA_PATH, DEVICE, args):
    predictions = []
    # Only sample a certain amount at a time for memory
    sample_size = 100
    remaining_num_samples = args.NUMBER_SAMPLES

    # Get dictionary of sentence lengths
    if args.SAMPLING_DISTRIBUTION == "data":
        with open(os.path.join(DATA_PATH, "sentence_length_distribution.pkl"), "rb") as f:
            sentence_length_distribution = pickle.load(f)
    else:
        sentence_length_distribution = None

    for sample_batch in tqdm(range(0, (args.NUMBER_SAMPLES // sample_size) + 1)):
        if remaining_num_samples <= sample_size:
            current_num_samples = remaining_num_samples
        else:
            current_num_samples = sample_size
            remaining_num_samples -= sample_size
        # sample
        sample_dict = model.sample(
            number_samples=current_num_samples,
            max_length=args.MAX_LENGTH,
            min_length=args.MIN_LENGTH,
            device=DEVICE,
            sentence_length_distribution=sentence_length_distribution,
        )

        # Get predictions (up until the sentence length + 50)
        max_len = 100
        _, current_predictions = model.generate(
            max_len=max_len, tokenizer=tokenizer, **sample_dict
        )
        predictions.extend(current_predictions)

    # Strip EOS
    predictions = strip_sep(predictions, tokenizer)

    def get_length(sentence):
        length = len(tokenizer(sentence)["input_ids"][1:-1])
        return length

    lengths = list(map(get_length, tqdm(predictions)))
    total_tokens = sum(lengths)
    text_dict = {"text": predictions, "length": lengths, "total_tokens": total_tokens}
    print("---------------------------- Samples ----------------------------")
    for i in range(0, min(len(predictions), 5)):
        print(i, "Samples: len", lengths[i], predictions[i])

    # Save and upload
    write_sent(predictions, os.path.join(OUTPUT_PATH, "samples.txt"))
    run.upload_file(os.path.join(OUTPUT_PATH, "samples.txt"))

    file = open(os.path.join(OUTPUT_PATH, "samples_" + "text_length.pkl"), "wb")
    pickle.dump(text_dict, file)
    file.close()


def main(args):
    OUTPUT_PATH = os.path.join(args.OUTPUT_DIR, args.PROJECT_NAME, args.EXPERIMENT_NAME)

    # Fetch the experiments config and update args
    with open(os.path.join(OUTPUT_PATH, "config.json"), "r") as f:
        parser.set_defaults(**json.load(f))
    args = parser.parse_args()
    print(args)

    # Data distributions of sentence lengths
    DATA_PATH = os.path.join(args.DATA_DIR, args.DATA_NAME)

    # Reproducibility function
    DEVICE, g = reproducibility(args.SEED)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Get vocabulary size
    VOCAB_SIZE = tokenizer.vocab_size

    # Load the model
    model = {
        "T": Transformer,
        "VTV": VariationalTransformerVariable,
        "VTP": VariationalTransformerPooled,
        "VTS": VariationalTransformerStride,
        "NVIB": TransformerNVIB,
    }[args.MODEL](VOCAB_SIZE, args).to(DEVICE)

    # KL ANNEALING
    model.args.KL_ANNEALING_FACTOR_GAUSSIAN = 1
    model.args.KL_ANNEALING_FACTOR_DIRICHLET = 1

    model.eval()
    print(model)

    # Load best model checkpoint
    checkpoint_dict = load_checkpoint(OUTPUT_PATH, DEVICE, prefix="best")

    model.load_state_dict(checkpoint_dict["model_state_dict"])
    wandb_id = checkpoint_dict["wandb_id"]

    wandb.init(
        project=args.PROJECT_NAME,
        entity=args.WANDB_ENTITY,
        # config=args, # Needs to be the same namespace obj
        id=wandb_id,
        resume="allow",
    )
    api = wandb.Api()
    run = api.run("{}/{}/{}".format(args.WANDB_ENTITY, args.PROJECT_NAME, wandb_id))

    # Clear cache (frees up memory for GPU)
    torch.cuda.empty_cache()

    # Sampling current distribution
    sample_func = getattr(model, "sample", None)
    sample_path_exists = os.path.exists(os.path.join(OUTPUT_PATH, "samples.txt"))
    if callable(sample_func):
        if not sample_path_exists:
            sample(model, tokenizer, OUTPUT_PATH, run, DATA_PATH, DEVICE, args)


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
    parser.add_argument("--PROJECT_NAME", default="localExperiments", help="wandb project name")
    parser.add_argument("--SEED", type=int, default=42, metavar="N", help="random seed")

    # Sample specific parameters
    parser.add_argument(
        "--MIN_LENGTH",
        type=int,
        default=5,
        metavar="N",
        help="Minimum length of words generated autoregressively",
    )
    parser.add_argument(
        "--MAX_LENGTH",
        type=int,
        default=50,
        metavar="N",
        help="Maximum length of words generated autoregressively",
    )
    parser.add_argument(
        "--NUMBER_SAMPLES", type=int, default=200, metavar="N", help="Number of random samples"
    )
    parser.add_argument(
        "--SAMPLING_DISTRIBUTION",
        default="uniform",
        choices=["uniform", "data"],
        help="The sampling distribution of sentence lengths",
    )

    args = parser.parse_args()

    main(args)
