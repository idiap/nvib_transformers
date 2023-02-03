#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# Evaluation whether the interpolations:
# Are interpolations different across tau?
# Are these interpolations fluent? F-PPL on validation interpolations
# Are these interpolations semantically smooth? - Qualitative?

import argparse
import json
from itertools import chain

# Local modules
import torch.utils.data as data
from evaluate import load
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from classes.Dataset import Dataset
from classes.Transformer import Transformer
from classes.VariationalTransformerNVIB import TransformerNVIB
from classes.VariationalTransformerPooled import VariationalTransformerPooled
from classes.VariationalTransformerStride import VariationalTransformerStride
from classes.VariationalTransformerVariable import VariationalTransformerVariable
from train import evaluation
from utils import *


class interpolationDataset(data.Dataset):
    """
    Dataset class to read in for the dataloader. Src and target (reversed)
    """

    def __init__(self, src_path):
        self.src_path = src_path
        self.src_sents = []
        self.tgt_sents = []

        with open(self.src_path, "r") as fdata:
            for row in fdata:
                # Remove new line chars
                row = row.strip("\n")
                self.src_sents.append(row)

        with open(self.src_path, "r") as fdata:
            for row in fdata:
                # Remove new line chars
                row = row.strip("\n")
                self.tgt_sents.append(row)

        # Reverse
        self.tgt_sents = self.tgt_sents[::-1]

        # Sanity check
        assert len(self.src_sents) == len(self.tgt_sents)

    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, idx):
        src = self.src_sents[idx]
        tgt = self.tgt_sents[idx]
        return [src, tgt]


class interpolationDataset(data.Dataset):
    """
    Dataset class to read in for the dataloader. Src and target (reversed)
    """

    def __init__(self, src_path):
        self.src_path = src_path
        self.src_sents = []
        self.tgt_sents = []

        with open(self.src_path, "r") as fdata:
            for row in fdata:
                # Remove new line chars
                row = row.strip("\n")
                self.src_sents.append(row)

        with open(self.src_path, "r") as fdata:
            for row in fdata:
                # Remove new line chars
                row = row.strip("\n")
                self.tgt_sents.append(row)

        # Reverse
        self.tgt_sents = self.tgt_sents[::-1]

        # Sanity check
        assert len(self.src_sents) == len(self.tgt_sents)

    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, idx):
        src = self.src_sents[idx]
        tgt = self.tgt_sents[idx]
        return [src, tgt]


def combine_latent_output(model, latent_outputs_dict_s1, latent_outputs_dict_s2, tau):
    # NVIB
    if type(model).__name__ == "TransformerNVIB":
        # Ns B E
        z = torch.cat((latent_outputs_dict_s1["z"][0], latent_outputs_dict_s2["z"][0]), 0)
        mu = torch.cat((latent_outputs_dict_s1["mu"], latent_outputs_dict_s2["mu"]), 0)
        logvar = torch.cat((latent_outputs_dict_s1["logvar"], latent_outputs_dict_s2["logvar"]), 0)
        memory_key_padding_mask = torch.cat(
            (
                latent_outputs_dict_s1["memory_key_padding_mask"],
                latent_outputs_dict_s2["memory_key_padding_mask"],
            ),
            1,
        )
        # This is where the magic happens!
        pi = torch.cat(
            (latent_outputs_dict_s1["pi"] * (1 - tau), latent_outputs_dict_s2["pi"] * (tau)), 0
        )

        latent_outputs_dict = {
            "z": (z, pi, mu, logvar),
            "pi": pi,
            "memory_key_padding_mask": memory_key_padding_mask,
            "mu": mu,
            "logvar": logvar,
        }
    else:
        # CONCATENATION and tau
        # Pretty clear it doesn't work

        # z = torch.cat((latent_outputs_dict_s1['z'] * (1-tau), latent_outputs_dict_s2['z'] * (tau)), 0)
        # memory_key_padding_mask = torch.cat(
        #     (latent_outputs_dict_s1['memory_key_padding_mask'], latent_outputs_dict_s2['memory_key_padding_mask']),
        #     1)
        # latent_outputs_dict = {"z": z,
        #                        'memory_key_padding_mask': memory_key_padding_mask,
        #                        }

        # ALIGNMENT by location, pad and tau
        if latent_outputs_dict_s1["z"].size(0) != latent_outputs_dict_s2["z"].size(0):
            if latent_outputs_dict_s1["z"].size(0) > latent_outputs_dict_s2["z"].size(0):
                # s2 is shorter

                diff = abs(
                    latent_outputs_dict_s1["z"].size(0) - latent_outputs_dict_s2["z"].size(0)
                )
                prior_pad = torch.zeros_like(
                    latent_outputs_dict_s1["z"], device=latent_outputs_dict_s1["z"].device
                )[0:diff, :, :]
                pad_mask = torch.ones(
                    (latent_outputs_dict_s1["memory_key_padding_mask"].size(0), diff),
                    device=latent_outputs_dict_s1["z"].device,
                    dtype=bool,
                )
                padded_z_s2 = torch.cat((latent_outputs_dict_s2["z"], prior_pad), 0)
                padded_mask_s2 = torch.cat(
                    (latent_outputs_dict_s2["memory_key_padding_mask"], pad_mask), 1
                )

                z = latent_outputs_dict_s1["z"] * (1 - tau) + padded_z_s2 * (tau)
                memory_key_padding_mask = ~(
                    ~latent_outputs_dict_s1["memory_key_padding_mask"] + ~padded_mask_s2
                )
            else:
                # s1 is shorter
                diff = abs(
                    latent_outputs_dict_s1["z"].size(0) - latent_outputs_dict_s2["z"].size(0)
                )
                prior_pad = torch.zeros_like(
                    latent_outputs_dict_s2["z"], device=latent_outputs_dict_s2["z"].device
                )[0:diff, :, :]
                pad_mask = torch.ones(
                    (latent_outputs_dict_s2["memory_key_padding_mask"].size(0), diff),
                    device=latent_outputs_dict_s1["z"].device,
                    dtype=bool,
                )
                padded_z_s1 = torch.cat((latent_outputs_dict_s1["z"], prior_pad), 0)
                padded_mask_s1 = torch.cat(
                    (latent_outputs_dict_s1["memory_key_padding_mask"], pad_mask), 1
                )

                z = padded_z_s1 * (1 - tau) + latent_outputs_dict_s2["z"] * (tau)
                memory_key_padding_mask = ~(
                    ~latent_outputs_dict_s2["memory_key_padding_mask"] + ~padded_mask_s1
                )
        else:
            z = latent_outputs_dict_s1["z"] * (1 - tau) + latent_outputs_dict_s2["z"] * (tau)
            memory_key_padding_mask = ~(
                ~latent_outputs_dict_s1["memory_key_padding_mask"]
                + ~latent_outputs_dict_s2["memory_key_padding_mask"]
            )

        latent_outputs_dict = {
            "z": z,
            "memory_key_padding_mask": memory_key_padding_mask,
        }
    return latent_outputs_dict


def get_predictions(model, tokenizer, latent_outputs_dict, max_len):
    predictions = []
    # Predictions autoregressively
    logits, batch_predictions = model.generate(
        max_len=max_len, tokenizer=tokenizer, **latent_outputs_dict
    )

    predictions.append(batch_predictions)

    # Unnest the lists
    predictions = list(chain.from_iterable(predictions))
    # Strip EOS
    predictions = strip_sep(predictions, tokenizer)
    return predictions


def main(args):
    # The vanilla model used to test samples
    OUTPUT_PATH = os.path.join(args.OUTPUT_DIR, args.PROJECT_NAME, args.EXPERIMENT_NAME)
    LM_PATH = os.path.join(args.OUTPUT_DIR, args.PROJECT_NAME, args.LANGUAGE_MODEL)

    with open(os.path.join(LM_PATH, "config.json"), "r") as f:
        parser.set_defaults(**json.load(f))
    lm_args = parser.parse_args()
    print("Language model arguments:")
    print(lm_args)

    with open(os.path.join(OUTPUT_PATH, "config.json"), "r") as f:
        parser.set_defaults(**json.load(f))
    args = parser.parse_args()
    print("Our model arguments:")
    print(args)

    # Reproducibility function
    DEVICE, g = reproducibility(args.SEED)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Get vocabulary size
    VOCAB_SIZE = tokenizer.vocab_size

    # Define the dataset
    validation_dataset = interpolationDataset(
        os.path.join(args.DATA_DIR, args.DATA_NAME, "validation.csv")
    )
    validation_length = get_lengths(os.path.join(args.DATA_DIR, args.DATA_NAME), samples=False)

    # Batch the validation data
    val_loader = DataLoader(
        validation_dataset,
        batch_size=256,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=4,
    )

    # Load the evaluation model for FPPL
    language_model = Transformer(VOCAB_SIZE, lm_args).to(DEVICE)
    LM_checkpoint_dict = load_checkpoint(LM_PATH, DEVICE, prefix="best")
    language_model.load_state_dict(LM_checkpoint_dict["model_state_dict"])
    language_model.eval()

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
    checkpoint_dict = load_checkpoint(OUTPUT_PATH, DEVICE, prefix="best")
    model.eval()

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

    if args.RUN_INTERPOLATIONS:

        # DO INTERPOLATIONS AND SEE IF THEY ARE DIFFERENT
        print("Getting interpolations")
        predictions = []
        sacrebleu = load("sacrebleu", device=DEVICE)
        # bertscore = load("bertscore", device=DEVICE)
        for tau in [0, 0.25, 0.5, 0.75, 1]:
            print("Tau: ", tau)
            results = {}
            bleu_s1 = []
            bleu_s2 = []
            # bert_score_s1 = []
            # bert_score_s2 = []
            count = 0

            for s1, s2 in tqdm(val_loader):
                s1 = list(s1)
                s2 = list(s2)
                # Tokenise the batches
                encoder_input_ids_s1, encoder_attention_mask_s1, _, _ = tokenize(
                    tokenizer, s1, DEVICE
                )
                encoder_input_ids_s2, encoder_attention_mask_s2, _, _ = tokenize(
                    tokenizer, s2, DEVICE
                )

                # Here is our BoV
                encoded_inputs_s1 = model.encode(encoder_input_ids_s1, encoder_attention_mask_s1)
                latent_outputs_dict_s1 = model.latent_layer(
                    encoded_inputs_s1, encoder_attention_mask_s1
                )

                encoded_inputs_s2 = model.encode(encoder_input_ids_s2, encoder_attention_mask_s2)
                latent_outputs_dict_s2 = model.latent_layer(
                    encoded_inputs_s2, encoder_attention_mask_s2
                )

                # Combine with tau
                if tau == 0:
                    latent_outputs_dict = latent_outputs_dict_s1
                elif tau == 1:
                    latent_outputs_dict = latent_outputs_dict_s2
                else:
                    latent_outputs_dict = combine_latent_output(
                        model, latent_outputs_dict_s1, latent_outputs_dict_s2, tau
                    )

                # get predictions
                batch_predictions = get_predictions(
                    model,
                    tokenizer,
                    latent_outputs_dict,
                    max_len=encoded_inputs_s1.size(0) + encoded_inputs_s2.size(0),
                )

                # Count if there is not an interpolation
                for i in range(0, len(batch_predictions)):
                    if (batch_predictions[i] == s1[i]) | (batch_predictions[i] == s2[i]):
                        count += 1
                    else:
                        # Only include the ones that are different to not bias FPPL
                        if tau == 0.5:
                            predictions.append(batch_predictions[i])

                # calculate bleu and bertscore for interpolations
                def calculate_bleu(prediction, ref):
                    return sacrebleu.compute(predictions=[prediction], references=[[ref]])["score"]

                # def calculate_bertscore(prediction, ref):
                #    return bertscore.compute(predictions=[prediction], references=[[ref]], lang="en")["f1"][0]

                bleu_s1.extend(list(map(calculate_bleu, batch_predictions, s1)))
                bleu_s2.extend(list(map(calculate_bleu, batch_predictions, s2)))
                # bert_score_s1.extend(list(map(calculate_bertscore, batch_predictions, s1)))
                # bert_score_s2.extend(list(map(calculate_bertscore, batch_predictions, s2)))

            prop_interpolations = (
                validation_dataset.__len__() - count
            ) / validation_dataset.__len__()
            print("Proportion of interpolations for tau ", tau, "is", prop_interpolations)
            wandb.run.summary["Prop_interpolations_tau" + str(tau)] = prop_interpolations

            print("S1 Bleu for tau ", tau, "is", np.mean(bleu_s1))
            results[str(tau) + "_bleu_s1"] = np.mean(bleu_s1)
            wandb.run.summary[str(tau) + "_bleu_s1"] = np.mean(bleu_s1)

            print("S2 Bleu for tau ", tau, "is", np.mean(bleu_s2))
            results[str(tau) + "_bleu_s2"] = np.mean(bleu_s2)
            wandb.run.summary[str(tau) + "_bleu_s2"] = np.mean(bleu_s2)

            # print("S1 Bertscore for tau ", tau, "is", np.mean(bert_score_s1))
            # results[str(tau) + "_bertscore_s1"] = np.mean(bert_score_s1)
            # wandb.run.summary[str(tau) + "_bertscore_s1"] = np.mean(bert_score_s1)

            # print("S2 Bertscore for tau ", tau, "is", np.mean(bert_score_s2))
            # results[str(tau) + "_bertscore_s2"] = np.mean(bert_score_s2)
            # wandb.run.summary[str(tau) + "_bertscore_s2"] = np.mean(bert_score_s2)

        # predictions = list(chain.from_iterable(predictions))
        write_sent(predictions, os.path.join(OUTPUT_PATH, "interpolations_tau0.5.txt"))

        def get_length(sentence):
            length = len(tokenizer(sentence)["input_ids"][1:-1])
            return length

        lengths = list(map(get_length, tqdm(predictions)))
        total_tokens = sum(lengths)
        text_dict = {"text": predictions, "length": lengths, "total_tokens": total_tokens}

        file = open(os.path.join(OUTPUT_PATH, "interpolations_" + "text_length.pkl"), "wb")
        pickle.dump(text_dict, file)
        file.close()

    # GET FPPL of interpolations
    interpolation_dataset = Dataset(os.path.join(OUTPUT_PATH, "interpolations_tau0.5.txt"))
    file = open(os.path.join(OUTPUT_PATH, "interpolations_text_length.pkl"), "rb")
    text_dict = pickle.load(file)
    file.close()
    interpolation_length = text_dict["total_tokens"]

    # Batch the sample data
    interpolation_loader = DataLoader(
        interpolation_dataset,
        batch_size=args.BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=4,
    )

    interpolation_metrics = evaluation(language_model, interpolation_loader, tokenizer, DEVICE)

    print(
        "                             Sample Forward Interpolations                             "
    )
    wandb_dict = logging(
        interpolation_metrics, interpolation_length, "Interpolation_Forward", wandb_dict={}
    )
    for name, metric in wandb_dict.items():
        wandb.run.summary[name] = metric

    # Save locally
    if not os.path.exists(OUTPUT_PATH + "/forwardinterpolationResults.csv"):
        save_csv(OUTPUT_PATH + "/forwardinterpolationResults.csv", wandb_dict)


if __name__ == "__main__":
    # ARGUMENTS
    parser = argparse.ArgumentParser()
    # Run booling
    parser.add_argument(
        "--RUN_INTERPOLATIONS", type=bool, default=False, help="wandb project name"
    )
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

# python interpolation_evaluation_v2.py --RUN_INTERPOLATIONS True --EXPERIMENT_NAME NVIBklg0.001kld1delta0.3kappa1seed1/ --PROJECT_NAME iclr
