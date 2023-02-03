#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Fabio Fehr <fabio.fehr@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# Take a dataset (local or download),
# clean it (different for each),
# put it by sentence,
# store it in its own folder as train validation test and sentence length distribution

import argparse
import itertools
import os
import pickle
import re

import matplotlib
import nltk.data
import seaborn as sns
from datasets import DatasetDict, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import reproducibility

matplotlib.use("pdf")
import matplotlib.pyplot as plt

nltk.download("punkt")

# Sentence splitting tokeniser (GLOBAL)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
nltk_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")


def get_bert_length(sentence):
    length = len(bert_tokenizer(sentence)["input_ids"][1:-1])
    return length


def split_sentences(text):
    sentences = nltk_tokenizer.tokenize(text)
    return sentences


def CountFrequency(lst):
    count = {}
    for i in lst:
        count[i] = count.get(i, 0) + 1
    return count


def save_file(data, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, name + ".csv"), "w") as f:
        for i in tqdm(range(0, len(data))):
            sentence = data[i]
            f.write(sentence + "\n")


def load_my_dataset(data_name, local_path):
    """
    Load local or from huggingface
    :param data_name: name of the data
    :param local_path: local path to data
    :return: corpus dictionary with train, validation and test data
    """
    if data_name == "wikitext103":
        # Train 1801350
        # Test 4358
        # Validation 3760

        if os.path.exists(local_path):
            print("Using local path")
            # LOCAL
            data_files = {
                "train": local_path + "wiki.train.tokens",
                "test": local_path + "wiki.test.tokens",
                "validation": local_path + "wiki.valid.tokens",
            }
            corpus = load_dataset("text", data_files=data_files)

        else:
            print("Using HuggingFace")
            # DOWNLOAD
            corpus = load_dataset("wikitext", "wikitext-103-v1")

    elif data_name == "wikitext2":
        # Train 36718
        # Test 4358
        # Validation 3760

        if os.path.exists(local_path):
            print("Using local path")
            # LOCAL
            data_files = {
                "train": local_path + "wiki.train.tokens",
                "test": local_path + "wiki.test.tokens",
                "validation": local_path + "wiki.valid.tokens",
            }
            corpus = load_dataset("text", data_files=data_files)

        else:
            print("Using HuggingFace")
            # DOWNLOAD
            corpus = load_dataset("wikitext", "wikitext-2-v1")

    elif data_name == "yelp":
        # Train 650K
        # Test 50K
        # Validation (first 50K of train)
        # DOWNLOAD
        validation = load_dataset("yelp_review_full", split="train[0:50000]")
        train = load_dataset("yelp_review_full", split="train[50000:]")
        test = load_dataset("yelp_review_full", split="test")
        corpus = DatasetDict({"train": train, "validation": validation, "test": test})

    elif data_name == "ptb":
        # Train 42068
        # Test 3761
        # Validation 3370
        # DOWNLOAD
        corpus = load_dataset("ptb_text_only")

    elif data_name == "yahoo":
        # Train 1 400 000
        # Test 60K
        # Validation (first 60K of train)
        # DOWNLOAD
        validation = load_dataset("yahoo_answers_topics", split="train[0:60000]")
        train = load_dataset("yahoo_answers_topics", split="train[60000:]")
        test = load_dataset("yahoo_answers_topics", split="test")
        corpus = DatasetDict({"train": train, "validation": validation, "test": test})

    else:
        print("Dataset not found")

    return corpus


def process_dataset(corpus, data_name, save_path):
    """
    Preprocess the data from a corpus to a cleaned dictionary with the text, lengths and total tokens
    :param corpus: the origonal corpus
    :param data_name: the name of the data
    :param save_path: path for saving
    :return: dictionary with the text, lengths and total tokens
    """
    if not os.path.exists(os.path.join(save_path, "text_length.pkl")):

        if data_name == "wikitext2" or data_name == "wikitext103":

            def preprocess(sentence):
                # Clean the sentences
                # Remove everything but a-zA-Z0-9 or <> for <unk> or . ,
                sentence = re.sub("[^a-zA-Z0-9 \,'\<\>\.]", "", sentence)
                # Remove spaces before . , '
                sentence = re.sub(r"\s+([.,\'])", r"\1", sentence)
                # Make lowercase
                sentence = sentence.lower()
                # Strip trailing white space
                sentence = sentence.strip()
                # Strip multiple white space
                sentence = re.sub(" +", " ", sentence)
                # Make unks compatible with BERT tokeniser
                sentence = re.sub("<unk>", "[UNK]", sentence)
                return sentence

            text_dict = {}
            for subset in tqdm(corpus):
                print("Processing the subset: ", subset)
                print("Splitting data:")
                text = list(
                    filter(
                        lambda x: (len(x) != 0 and len(x) != 1 and x[1] != "="),
                        tqdm(corpus[subset]["text"]),
                    )
                )
                sentences = list(map(split_sentences, tqdm(text)))
                sentences = list(itertools.chain(*sentences))
                print("Cleaning data:")
                sentences = list(map(preprocess, tqdm(sentences)))
                print("Collect BERT lengths:")
                lengths = list(map(get_bert_length, tqdm(sentences)))
                print("Filter by length:")
                lengths, sentences = zip(
                    *(
                        (length, sentences)
                        for length, sentences in zip(lengths, sentences)
                        if (length > 5 and length < 50)
                    )
                )
                total_tokens = sum(lengths)
                text_dict[subset] = {
                    "text": sentences,
                    "length": lengths,
                    "total_tokens": total_tokens,
                }
                print("Saving:")
                with open(os.path.join(save_path, subset + ".csv"), "w") as f:
                    for sentence in sentences:
                        f.write(sentence + "\n")

        elif data_name == "ptb":

            def preprocess(sentence):
                # Clean the sentences (Not necessary for this data)
                # Remove everything but a-zA-Z0-9 or <> for <unk> or . ,
                sentence = re.sub("[^a-zA-Z0-9 \,'\<\>\.]", "", sentence)
                # Remove spaces before . , '
                sentence = re.sub(r"\s+([.,\'])", r"\1", sentence)
                # Special case
                sentence = sentence.replace(" n't", "n't")
                # Make lowercase
                sentence = sentence.lower()
                # Strip trailing white space
                sentence = sentence.strip()
                # Strip multiple white space
                sentence = re.sub(" +", " ", sentence)
                # Make unks compatible with BERT tokeniser
                sentence = re.sub("<unk>", "[UNK]", sentence)
                return sentence

            text_dict = {}
            for subset in tqdm(corpus):
                print("Processing the subset: ", subset)
                print("Cleaning data:")
                sentences = list(map(preprocess, tqdm(corpus[subset]["sentence"])))
                print("Collect BERT lengths:")
                lengths = list(map(get_bert_length, tqdm(sentences)))
                print("Filter by length:")
                lengths, sentences = zip(
                    *(
                        (length, sentences)
                        for length, sentences in zip(lengths, sentences)
                        if (length > 5 and length < 50)
                    )
                )
                total_tokens = sum(lengths)
                text_dict[subset] = {
                    "text": sentences,
                    "length": lengths,
                    "total_tokens": total_tokens,
                }
                print("Saving:")
                with open(os.path.join(save_path, subset + ".csv"), "w") as f:
                    for sentence in sentences:
                        f.write(sentence + "\n")

        elif data_name == "yahoo":

            def preprocess(sentence):
                sentence = sentence.replace("\\n", " ")
                sentence = sentence.replace("<br />", " ")
                sentence = re.sub("[^a-zA-Z0-9 \,'\.\$]", " ", sentence)
                sentence = sentence.strip()
                sentence = re.sub(" +", " ", sentence)
                sentence = sentence.lower()
                return sentence

            text_dict = {}
            for subset in corpus:
                print("Processing the subset: ", subset)
                print("Cleaning data:")
                sentences = list(map(preprocess, tqdm(corpus[subset]["best_answer"])))
                print("Collect BERT lengths:")
                lengths = list(map(get_bert_length, tqdm(sentences)))
                topics = corpus[subset]["topic"]
                # Filter out only sentences with lengths < 100
                print("Filter by length:")
                lengths, sentences, topics = zip(
                    *(
                        (length, sentences, topics)
                        for length, sentences, topics in zip(lengths, sentences, topics)
                        if length < 50
                    )
                )
                total_tokens = sum(lengths)
                text_dict[subset] = {
                    "text": sentences,
                    "length": lengths,
                    "label": topics,
                    "total_tokens": total_tokens,
                }
                print("Saving:")
                with open(os.path.join(save_path, subset + ".csv"), "w") as f:
                    for sentence in sentences:
                        f.write(sentence + "\n")

        elif data_name == "yelp":
            text_dict = {}

            def preprocess(sentence):
                sentence = sentence.replace("\\n", " ")
                sentence = re.sub("[^a-zA-Z0-9 \,'\.\$\?\!]", " ", sentence)
                sentence = re.sub(" +", " ", sentence)
                sentence = sentence.lower()
                return sentence

            for subset in corpus:
                print("Processing the subset: ", subset)
                print("Cleaning data:")
                sentences = list(map(preprocess, tqdm(corpus[subset]["text"])))
                print("Collect BERT lengths:")
                lengths = list(map(get_bert_length, tqdm(sentences)))
                labels = corpus[subset]["label"]
                # Filter out only sentences with lengths < 100
                print("Filter by length:")
                lengths, sentences, labels = zip(
                    *(
                        (length, sentences, labels)
                        for length, sentences, labels in zip(lengths, sentences, labels)
                        if length < 50
                    )
                )
                total_tokens = sum(lengths)
                text_dict[subset] = {
                    "text": sentences,
                    "length": lengths,
                    "label": labels,
                    "total_tokens": total_tokens,
                }
                print("Saving:")
                with open(os.path.join(save_path, subset + ".csv"), "w") as f:
                    for sentence in sentences:
                        f.write(sentence + "\n")

        else:
            print("Dataset not found")

        file = open(os.path.join(save_path, "text_length.pkl"), "wb")
        pickle.dump(text_dict, file)
        file.close()
    else:
        file = open(os.path.join(save_path, "text_length.pkl"), "rb")
        text_dict = pickle.load(file)
        file.close()

    return text_dict


def plotData(text_dict, save_path, cummulative=False):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    fig = sns.histplot(
        text_dict["train"]["length"],
        # bins=range(0, 100, 2),
        cumulative=cummulative,
        fill=False,
        # color="",  # blue
        ax=ax1,
    )
    ax1.set_xlabel("Sentence length", fontsize=16)
    ax1.set_ylabel("Cummulative number of sentences", fontsize=16)
    ax1.tick_params(labelsize=14)
    sns.kdeplot(text_dict["train"]["length"], cumulative=cummulative, color="#009e73", ax=ax2)
    ax2.set_xlabel("Sentence length", fontsize=16)
    ax2.set_ylabel("Density", fontsize=16)
    ax2.tick_params(labelsize=14)
    if cummulative:
        fig.set(xlabel="Sentence length", ylabel="Cummulative number of sentences")
    else:
        fig.set(xlabel="Sentence length", ylabel="Number of sentences")

    plt.tight_layout()
    plt.ticklabel_format(style="plain", axis="y")
    fig = fig.get_figure()

    if cummulative:
        fig.savefig(os.path.join(save_path, "sentLengthCumm.pdf"), dpi=80, bbox_inches="tight")
    else:
        fig.savefig(os.path.join(save_path, "sentLengthDist.pdf"), dpi=80, bbox_inches="tight")

    plt.clf()


def main(args):
    # Set seed
    reproducibility(args.SEED)

    # Get corpus
    corpus = load_my_dataset(args.DATA, args.LOCAL_PATH)

    # Directory
    SAVEPATH = os.path.join("data", args.DATA)
    if not os.path.exists(SAVEPATH):
        os.makedirs(SAVEPATH)

    # Process the dataset
    text_length_dictionary = process_dataset(corpus, args.DATA, SAVEPATH)

    # Plot distributions
    plotData(text_length_dictionary, SAVEPATH, cummulative=False)
    plotData(text_length_dictionary, SAVEPATH, cummulative=True)

    # # Save sentence_length_distribution
    sentence_length_distribution = CountFrequency(text_length_dictionary["train"]["length"])
    with open(os.path.join(SAVEPATH, "sentence_length_distribution.pkl"), "wb") as f:
        pickle.dump(sentence_length_distribution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--DATA", choices=["wikitext103", "wikitext2", "yelp", "ptb", "yahoo"], help="Select data"
    )
    parser.add_argument("--LOCAL_PATH", default=None, type=str, help="local path to data")
    parser.add_argument("--SEED", type=int, default=42, metavar="N", help="Seed for sample")
    args = parser.parse_args()
    main(args)
