#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# Testing
import argparse
import collections
import json
from itertools import chain

from datasets import load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# Local modules
from classes.Dataset import Dataset
from classes.MetricsAverage import MetricsAverage
from classes.Transformer import Transformer
from classes.VariationalTransformerNVIB import TransformerNVIB
from classes.VariationalTransformerPooled import VariationalTransformerPooled
from classes.VariationalTransformerStride import VariationalTransformerStride
from classes.VariationalTransformerVariable import VariationalTransformerVariable
from utils import *

# Disables the warnings for parallelised tokenisers
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def reconstruct(
    model,
    tokenizer,
    dataloader,
    reconstruct_dataset,
    reconstruct_length,
    OUTPUT_PATH,
    DATA_SUBSET,
    run,
    DEVICE,
):
    """
    Run the reconstruction

    :param model: Model you wish to evaluate
    :param tokenizer: tokenizer
    :param dataloader: reconstruction dataloader (valdiation test or train)
    :param reconstruct_dataset: for comparing against bleu
    :param reconstruct_length: for perplexity
    :param OUTPUT_PATH: location of outputs for saving
    :param DATA_SUBSET: train validation or test
    :param run: for bertscore
    :param DEVICE: cpu or gpu
    :return:
    """
    # Saving flag for validation and test results
    if DATA_SUBSET == "test":
        flag = "_test"
    else:
        flag = ""

    predictions = []
    evaluation_metrics = collections.defaultdict(lambda: MetricsAverage())
    torch.cuda.empty_cache()
    for idx, batch in enumerate(tqdm(dataloader)):
        # Tokenise the batches
        encoder_input_ids, src_key_padding_mask, _, _ = tokenize(tokenizer, batch, DEVICE)

        encoded_inputs = model.encode(encoder_input_ids, src_key_padding_mask)  # SEQ, BS, EMB
        latent_outputs_dict = model.latent_layer(encoded_inputs, src_key_padding_mask)

        # Predictions autoregressively
        logits, batch_predictions = model.generate(
            max_len=encoded_inputs.size(0), tokenizer=tokenizer, **latent_outputs_dict
        )

        predictions.append(batch_predictions)
        loss_dict = model.loss(logits=logits, targets=encoder_input_ids, **latent_outputs_dict)
        # Store losses
        for name, metric in loss_dict.items():
            evaluation_metrics[name].update(float(metric))
        # Logging alphas
        if "alpha" in latent_outputs_dict.keys():
            evaluation_metrics["avg_num_vec"].update(float(latent_outputs_dict["avg_num_vec"]))
            evaluation_metrics["avg_prop_vec"].update(float(latent_outputs_dict["avg_prop_vec"]))
            evaluation_metrics["avg_alpha0"].update(float(latent_outputs_dict["avg_alpha0"]))

    # Unnest the lists
    predictions = list(chain.from_iterable(predictions))
    # Strip EOS
    predictions = strip_sep(predictions, tokenizer)

    print("---------------------------- Losses ----------------------------")
    wandb_dict = logging(evaluation_metrics, reconstruct_length, "EVAL" + flag, wandb_dict={})
    for name, metric in wandb_dict.items():
        wandb.run.summary[name] = metric

    print("---------------------------- Reconstuctions ----------------------------")
    for i in range(0, int(min(dataloader.batch_size, 5))):
        batch = next(iter(dataloader))
        print(i, "Reference:  ", batch[i])
        print(i, "Prediction: ", predictions[i])

    print("---------------------------- SACRE BLEU ----------------------------")
    sacrebleu = load_metric("sacrebleu", device=DEVICE)
    results = sacrebleu.compute(
        predictions=[pred.split() for pred in predictions],
        references=[[ref.split()] for ref in reconstruct_dataset[:]],
    )
    for key in results:
        print(key, results[key])
        wandb.run.summary["Sacre_Bleu_" + key + flag] = results[key]

    save_csv(os.path.join(OUTPUT_PATH, "sacre_bleu" + flag + ".csv"), results)

    # print("---------------------------- BERT SCORE ----------------------------")
    # bertscore = load_metric("bertscore", device=DEVICE)
    # results = bertscore.compute(predictions=[pred.split() for pred in predictions],
    #                            references=[[ref.split()] for ref in reconstruct_dataset[:]], lang="en")
    # bertscore_dict = {}
    # for key in results:
    #    if key != "hashcode":
    #        bertscore_dict[key] = np.mean(results[key])
    #        print("Mean_" + key, np.mean(results[key]))
    #        wandb.run.summary["BertScore_"  + os.path.basename(DATA_PATH) + key] = np.mean(results[key])
    #
    # save_csv(os.path.join(OUTPUT_PATH, 'bertscore_' + os.path.basename(DATA_PATH) + '.csv'), bertscore_dict)

    # Write reconstructions + upload to Wandb
    # write_sent(predictions, os.path.join(
    #    OUTPUT_PATH, "reconstruction" + os.path.basename(DATA_PATH) + '.txt'))
    # run.upload_file(os.path.join(
    #    OUTPUT_PATH, "reconstruction" + os.path.basename(DATA_PATH) + '.txt'))


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

    # Define the dataset
    reconstruct_dataset = Dataset(os.path.join(DATA_PATH, args.DATA_SUBSET + ".csv"))
    reconstruct_length = get_lengths(DATA_PATH, args.DATA_SUBSET)

    # Batch the data
    dataloader = DataLoader(
        reconstruct_dataset,
        batch_size=args.GENERATION_BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=4,
    )

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

    # Reconstruction
    reconstruct(
        model,
        tokenizer,
        dataloader,
        reconstruct_dataset,
        reconstruct_length,
        OUTPUT_PATH,
        args.DATA_SUBSET,
        run,
        DEVICE,
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
        default="vanillaTransformer",
        metavar="N",
        help="Name of current experiment",
    )
    parser.add_argument(
        "--GENERATION_BATCH_SIZE", type=int, default=128, metavar="N", help="batch size"
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
    parser.add_argument("--PROJECT_NAME", default="localExperiments", help="wandb project name")
    parser.add_argument("--SEED", type=int, default=42, metavar="N", help="random seed")
    args = parser.parse_args()
    main(args)
