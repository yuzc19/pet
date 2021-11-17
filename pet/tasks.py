# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the logic for loading training and test data for all tasks.
"""

import csv
import json
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

import log
from pet import task_helpers
from pet.utils import InputExample

logger = log.get_logger("root")


def _shuffle_and_restrict(
    examples: List[InputExample], num_examples: int, seed: int = 42
) -> List[InputExample]:
    """
    Shuffle a list of examples and restrict it to a given maximum size.

    :param examples: the examples to shuffle and restrict
    :param num_examples: the maximum number of examples
    :param seed: the random seed for shuffling
    :return: the first ``num_examples`` elements of the shuffled list
    """
    if 0 < num_examples < len(examples):
        random.Random(seed).shuffle(examples)
        examples = examples[:num_examples]
    return examples


class LimitedExampleList:
    def __init__(self, labels: List[str], max_examples=-1):
        """
        Implementation of a list that stores only a limited amount of examples per label.

        :param labels: the set of all possible labels
        :param max_examples: the maximum number of examples per label. This can either be a fixed number,
               in which case `max_examples` examples are loaded for every label, or a list with the same size as
               `labels`, in which case at most `max_examples[i]` examples are loaded for label `labels[i]`.
        """
        self._labels = labels
        self._examples = []
        self._examples_per_label = defaultdict(int)

        if isinstance(max_examples, list):
            self._max_examples = dict(zip(self._labels, max_examples))
        else:
            self._max_examples = {label: max_examples for label in self._labels}

    def is_full(self):
        """Return `true` iff no more examples can be added to this list"""
        for label in self._labels:
            if (
                self._examples_per_label[label] < self._max_examples[label]
                or self._max_examples[label] < 0
            ):
                return False
        return True

    def add(self, example: InputExample) -> bool:
        """
        Add a new input example to this list.

        :param example: the example to add
        :returns: `true` iff the example was actually added to the list
        """
        label = example.label
        if (
            self._examples_per_label[label] < self._max_examples[label]
            or self._max_examples[label] < 0
        ):
            self._examples_per_label[label] += 1
            self._examples.append(example)
            return True
        return False

    def to_list(self):
        return self._examples


class DataProcessor(ABC):
    """
    Abstract class that provides methods for loading training, testing, development and unlabeled examples for a given
    task
    """

    @abstractmethod
    def get_train_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the test set."""
        pass

    @abstractmethod
    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the unlabeled set."""
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Get the list of labels for this data set."""
        pass

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            MnliProcessor._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            MnliProcessor._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched",
        )

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)

        return examples

    @staticmethod
    def _read_tsv(input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_mismatched",
        )

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()


class AgnewsProcessor(DataProcessor):
    """Processor for the AG news data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.csv"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["1", "2", "3", "4"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path) as f:
            reader = csv.reader(f, delimiter=",")
            for idx, row in enumerate(reader):
                label, headline, body = row
                guid = "%s-%s" % (set_type, idx)
                text_a = headline.replace("\\", " ")
                text_b = body.replace("\\", " ")

                example = InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label=label
                )
                examples.append(example)

        return examples


class YahooAnswersProcessor(DataProcessor):
    """Processor for the Yahoo Answers data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.csv"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding="utf8") as f:
            reader = csv.reader(f, delimiter=",")
            for idx, row in enumerate(reader):
                label, question_title, question_body, answer = row
                guid = "%s-%s" % (set_type, idx)
                text_a = " ".join(
                    [
                        question_title.replace("\\n", " ").replace("\\", " "),
                        question_body.replace("\\n", " ").replace("\\", " "),
                    ]
                )
                text_b = answer.replace("\\n", " ").replace("\\", " ")

                example = InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label=label
                )
                examples.append(example)

        return examples


class YelpPolarityProcessor(DataProcessor):
    """Processor for the YELP binary classification set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.csv"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["1", "2"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path) as f:
            reader = csv.reader(f, delimiter=",")
            for idx, row in enumerate(reader):
                label, body = row
                guid = "%s-%s" % (set_type, idx)
                text_a = body.replace("\\n", " ").replace("\\", " ")

                example = InputExample(guid=guid, text_a=text_a, label=label)
                examples.append(example)

        return examples


class YelpFullProcessor(YelpPolarityProcessor):
    """Processor for the YELP full classification set."""

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_labels(self):
        return ["1", "2", "3", "4", "5"]


class XStanceProcessor(DataProcessor):
    """Processor for the X-Stance data set."""

    def __init__(self, language: str = None):
        if language is not None:
            assert language in ["de", "fr"]
        self.language = language

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"))

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"))

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["FAVOR", "AGAINST"]

    def _create_examples(self, path: str) -> List[InputExample]:
        examples = []

        with open(path, encoding="utf8") as f:
            for line in f:
                example_json = json.loads(line)
                label = example_json["label"]
                id_ = example_json["id"]
                text_a = example_json["question"]
                text_b = example_json["comment"]
                language = example_json["language"]

                if self.language is not None and language != self.language:
                    continue

                example = InputExample(
                    guid=id_, text_a=text_a, text_b=text_b, label=label
                )
                examples.append(example)

        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_unlabeled_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "unlabeled.tsv")), "unlabeled"
        )

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class AxGProcessor(RteProcessor):
    """Processor for the AX-G diagnostic data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "AX-g.jsonl"), "train")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "AX-g.jsonl"), "test")


class AxBProcessor(RteProcessor):
    """Processor for the AX-B diagnostic data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "AX-b.jsonl"), "train")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "AX-b.jsonl"), "test")

    def _create_examples(
        self, path, set_type, hypothesis_name="sentence2", premise_name="sentence1"
    ):
        return super()._create_examples(path, set_type, hypothesis_name, premise_name)


class CbProcessor(RteProcessor):
    """Processor for the CB data set."""

    def get_labels(self):
        return ["entailment", "contradiction", "neutral"]


class WicProcessor(DataProcessor):
    """Processor for the WiC data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled"
        )

    def get_labels(self):
        return ["F", "T"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []
        with open(path, encoding="utf8") as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json["idx"]
                if isinstance(idx, str):
                    idx = int(idx)
                label = "T" if example_json.get("label") else "F"
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json["sentence1"]
                text_b = example_json["sentence2"]
                meta = {"word": example_json["word"]}
                example = InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label,
                    idx=idx,
                    meta=meta,
                )
                examples.append(example)
        return examples


class WscProcessor(DataProcessor):
    """Processor for the WSC data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled"
        )

    def get_labels(self):
        return ["False", "True"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []
        with open(path, encoding="utf8") as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json["idx"]
                label = str(example_json["label"]) if "label" in example_json else None
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json["text"]
                meta = {
                    "span1_text": example_json["target"]["span1_text"],
                    "span2_text": example_json["target"]["span2_text"],
                    "span1_index": example_json["target"]["span1_index"],
                    "span2_index": example_json["target"]["span2_index"],
                }
                example = InputExample(
                    guid=guid, text_a=text_a, label=label, meta=meta, idx=idx
                )
                if set_type == "train" and label != "True":
                    continue
                examples.append(example)
        return examples


class BoolQProcessor(DataProcessor):
    """Processor for the BoolQ data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled"
        )

    def get_labels(self):
        return ["False", "True"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding="utf8") as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json["idx"]
                label = str(example_json["label"]) if "label" in example_json else None
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json["passage"]
                text_b = example_json["question"]
                example = InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx
                )
                examples.append(example)

        return examples


class CopaProcessor(DataProcessor):
    """Processor for the COPA data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled"
        )

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding="utf8") as f:
            for line in f:
                example_json = json.loads(line)
                label = str(example_json["label"]) if "label" in example_json else None
                idx = example_json["idx"]
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json["premise"]
                meta = {
                    "choice1": example_json["choice1"],
                    "choice2": example_json["choice2"],
                    "question": example_json["question"],
                }
                example = InputExample(
                    guid=guid, text_a=text_a, label=label, meta=meta, idx=idx
                )
                examples.append(example)

        if set_type == "train" or set_type == "unlabeled":
            mirror_examples = []
            for ex in examples:
                label = "1" if ex.label == "0" else "0"
                meta = {
                    "choice1": ex.meta["choice2"],
                    "choice2": ex.meta["choice1"],
                    "question": ex.meta["question"],
                }
                mirror_example = InputExample(
                    guid=ex.guid + "m", text_a=ex.text_a, label=label, meta=meta
                )
                mirror_examples.append(mirror_example)
            examples += mirror_examples
            logger.info(
                f"Added {len(mirror_examples)} mirror examples, total size is {len(examples)}..."
            )
        return examples


class MultiRcProcessor(DataProcessor):
    """Processor for the MultiRC data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled"
        )

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding="utf8") as f:
            for line in f:
                example_json = json.loads(line)

                passage_idx = example_json["tid"]
                text = example_json["text"]
                question_idx = example_json["qid"]
                question = example_json["question"]
                label = str(example_json["label"]) if "label" in example_json else None
                answer_idx = example_json["aid"]
                guid = f"{set_type}-p{passage_idx}-q{question_idx}-a{answer_idx}"
                meta = {
                    "passage_idx": passage_idx,
                    "question_idx": question_idx,
                    "answer_idx": answer_idx,
                    "answer": example_json["answer"],
                }
                idx = [passage_idx, question_idx, answer_idx]
                example = InputExample(
                    guid=guid,
                    text_a=text,
                    text_b=question,
                    label=label,
                    meta=meta,
                    idx=idx,
                )
                examples.append(example)

        question_indices = list(
            set(example.meta["question_idx"] for example in examples)
        )
        label_distribution = Counter(example.label for example in examples)
        logger.info(
            f"Returning {len(examples)} examples corresponding to {len(question_indices)} questions with label "
            f"distribution {list(label_distribution.items())}"
        )
        return examples


class RecordProcessor(DataProcessor):
    """Processor for the ReCoRD data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled"
        )

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(
        path, set_type, seed=42, max_train_candidates_per_question: int = 10
    ) -> List[InputExample]:
        examples = []

        entity_shuffler = random.Random(seed)

        with open(path, encoding="utf8") as f:
            for idx, line in enumerate(f):
                example_json = json.loads(line)

                idx = example_json["tid"]
                text = example_json["text"]
                entities = set()

                entities = list(set(example_json["entities"].split(", ")))

                text = text.replace(
                    "@highlight\n", "- "
                )  # we follow the GPT-3 paper wrt @highlight annotations

                question = example_json["query"]
                question_idx = example_json["qid"]
                answers = set()

                for answer_json in example_json["answers"]:
                    answer = answer_json["text"]
                    answers.add(answer)

                answers = list(answers)

                if set_type == "train":
                    # create a single example per *correct* answer
                    for answer_idx, answer in enumerate(answers):
                        candidates = [ent for ent in entities if ent not in answers]
                        if len(candidates) > max_train_candidates_per_question - 1:
                            entity_shuffler.shuffle(candidates)
                            candidates = candidates[
                                : max_train_candidates_per_question - 1
                            ]

                        guid = f"{set_type}-p{idx}-q{question_idx}-a{answer_idx}"
                        meta = {
                            "passage_idx": idx,
                            "question_idx": question_idx,
                            "candidates": [answer] + candidates,
                            "answers": [answer],
                        }
                        ex_idx = [idx, question_idx, answer_idx]
                        example = InputExample(
                            guid=guid,
                            text_a=text,
                            text_b=question,
                            label="1",
                            meta=meta,
                            idx=ex_idx,
                        )
                        examples.append(example)

                else:
                    # create just one example with *all* correct answers and *all* answer candidates
                    guid = f"{set_type}-p{idx}-q{question_idx}"
                    meta = {
                        "passage_idx": idx,
                        "question_idx": question_idx,
                        "candidates": entities,
                        "answers": answers,
                    }
                    example = InputExample(
                        guid=guid, text_a=text, text_b=question, label="1", meta=meta
                    )
                    examples.append(example)

        question_indices = list(
            set(example.meta["question_idx"] for example in examples)
        )
        label_distribution = Counter(example.label for example in examples)
        logger.info(
            f"Returning {len(examples)} examples corresponding to {len(question_indices)} questions with label "
            f"distribution {list(label_distribution.items())}"
        )
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_unlabeled_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "unlabeled.tsv")), "unlabeled"
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type) -> List[InputExample]:
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
            )
        return examples


PROCESSORS = {
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "agnews": AgnewsProcessor,
    "yahoo": YahooAnswersProcessor,
    "yelp-polarity": YelpPolarityProcessor,
    "yelp-full": YelpFullProcessor,
    "xstance-de": lambda: XStanceProcessor("de"),
    "xstance-fr": lambda: XStanceProcessor("fr"),
    "xstance": XStanceProcessor,
    "wic": WicProcessor,
    "rte": RteProcessor,
    "cb": CbProcessor,
    "wsc": WscProcessor,
    "boolq": BoolQProcessor,
    "copa": CopaProcessor,
    "multirc": MultiRcProcessor,
    "record": RecordProcessor,
    "ax-g": AxGProcessor,
    "ax-b": AxBProcessor,
    "sst-2": Sst2Processor,
}  # type: Dict[str,Callable[[],DataProcessor]]

TASK_HELPERS = {
    "wsc": task_helpers.WscTaskHelper,
    "multirc": task_helpers.MultiRcTaskHelper,
    "copa": task_helpers.CopaTaskHelper,
    "record": task_helpers.RecordTaskHelper,
}

METRICS = {
    "cb": ["acc", "f1-macro"],
    "multirc": ["acc", "f1", "em"],
    "record": ["acc", "f1-macro"],
}

DEFAULT_METRICS = ["acc"]

TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
UNLABELED_SET = "unlabeled"

SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, UNLABELED_SET]


def load_examples(
    task,
    data_dir: str,
    set_type: str,
    *_,
    num_examples: int = None,
    num_examples_per_label: int = None,
    seed: int = 42,
) -> List[InputExample]:
    """Load examples for a given task."""
    assert (num_examples is not None) ^ (
        num_examples_per_label is not None
    ), "Exactly one of 'num_examples' and 'num_examples_per_label' must be set."
    assert (not set_type == UNLABELED_SET) or (
        num_examples is not None
    ), "For unlabeled data, 'num_examples_per_label' is not allowed"

    processor = PROCESSORS[task]()

    ex_str = (
        f"num_examples={num_examples}"
        if num_examples is not None
        else f"num_examples_per_label={num_examples_per_label}"
    )
    logger.info(
        f"Creating features from dataset file at {data_dir} ({ex_str}, set_type={set_type})"
    )

    if set_type == DEV_SET:
        examples = processor.get_dev_examples(data_dir)
    elif set_type == TEST_SET:
        examples = processor.get_test_examples(data_dir)
    elif set_type == TRAIN_SET:
        examples = processor.get_train_examples(data_dir)
    elif set_type == UNLABELED_SET:
        examples = processor.get_unlabeled_examples(data_dir)
        for example in examples:
            example.label = processor.get_labels()[0]
    else:
        raise ValueError(
            f"'set_type' must be one of {SET_TYPES}, got '{set_type}' instead"
        )

    if num_examples is not None:
        examples = _shuffle_and_restrict(examples, num_examples, seed)

    elif num_examples_per_label is not None:
        limited_examples = LimitedExampleList(
            processor.get_labels(), num_examples_per_label
        )
        for example in examples:
            limited_examples.add(example)
        examples = limited_examples.to_list()

    label_distribution = Counter(example.label for example in examples)
    logger.info(
        f"Returning {len(examples)} {set_type} examples with label dist.: {list(label_distribution.items())}"
    )

    return examples
