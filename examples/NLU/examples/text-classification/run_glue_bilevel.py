#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
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
""" Finetuning the library models for sequence classification on GLUE."""
import random
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import warnings
from copy import deepcopy
from typing import Union, List

from betty.configs import Config, EngineConfig
from betty.engine import Engine
from betty.problems import ImplicitProblem
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split

from utils.architecture import *
from utils.arguments import ModelTrainingArguments, ModelArguments, DataTrainingArguments

assert torch.cuda.is_available()
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    default_data_collator,
)
from optimizer import (
    create_optimizer_scheduler,
    create_lora_adam_optimizer_from_args,
    create_arch_adam_optimizer_from_args
)
from transformers.utils import check_min_version
from utils.util import *

warnings.filterwarnings('ignore')

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.4.0")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "amazon_reviews_multi": ("review_body", None),
}

task_to_metrics = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "accuracy",
    "qnli": "accuracy",
    "qqp": "accuracy",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",
    "wnli": "accuracy",
    "amazon_reviews_multi": "accuracy",
}

logger = logging.getLogger(__name__)

from transformers import logging as transformers_logging

logging.basicConfig(level=logging.WARNING)
transformers_logging.set_verbosity_warning()


def get_pretrained_objects(model_args, data_training_args, model_training_args, num_labels):
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_training_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        cls_dropout=model_training_args.cls_dropout,
        lora_type=model_args.lora_type,
        lora_alpha=model_args.lora_alpha,
        lora_r=model_args.lora_r,
        lora_dropout=model_args.lora_dropout,
        apply_adapter=model_args.lora_type == 'adapter',
        adapter_type='houlsby',
        adapter_size=model_args.lora_r,
        reg_loss_wgt=model_args.reg_loss_wgt,
        masking_prob=model_args.masking_prob
    )
    config.problem_type = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        apply_adapter=model_args.lora_type == 'adapter',
        adapter_type='houlsby',
        adapter_size=model_args.lora_r,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    return config, tokenizer, model


def get_dataset_and_model(model_training_args, data_training_args, model_args):
    if data_training_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_training_args.task_name)
    else:
        datasets = load_dataset(data_training_args.dataset_name, data_training_args.dataset_config_name,
                                trust_remote_code=True)
    print(f'End loading dataset {data_training_args.task_name}')

    # Labels
    if data_training_args.task_name is not None:
        is_regression = data_training_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    print(f'Begin fetching pretrained model {model_args.model_name_or_path}')
    config, tokenizer, model = get_pretrained_objects(model_args, data_training_args, model_training_args, num_labels)
    print(f'End fetching pretrained model {model_args.model_name_or_path}')

    # Preprocessing the datasets
    print(f'Begin preprocessing dataset')
    if data_training_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_training_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_training_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_training_args.task_name is not None
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_training_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_training_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_training_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_training_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_training_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True,
                            load_from_cache_file=not data_training_args.overwrite_cache)
    if model_training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_training_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_training_args.max_train_samples))

    if model_training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_training_args.task_name == "mnli" else "validation"]
        eval_dataset_mm = datasets["validation_mismatched" if data_training_args.task_name == "mnli" else "validation"]
        if data_training_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_training_args.max_val_samples))

    if data_training_args.task_name is not None:
        metric = load_metric("glue", data_training_args.task_name)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_training_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    if data_training_args.pad_to_max_length:
        data_collator = default_data_collator
    elif model_training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    seed = model_training_args.seed
    g.manual_seed(seed)

    inner_portion = int(len(train_dataset) * data_training_args.inner_training_portion)
    outer_portion = len(train_dataset) - inner_portion
    train_dataset_inner, train_dataset_outer = random_split(train_dataset, [inner_portion, outer_portion], g)

    if model_training_args.train_train:
        print('[Warning]: using "train-train" mode')
        train_dataset_inner = train_dataset
        train_dataset_outer = train_dataset

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=model_training_args.train_batch_size,
        collate_fn=data_collator,
        drop_last=model_training_args.dataloader_drop_last,
        num_workers=model_training_args.dataloader_num_workers,
        pin_memory=model_training_args.dataloader_pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )

    train_dataloader_inner = DataLoader(
        train_dataset_inner,
        batch_size=model_training_args.train_batch_size,
        collate_fn=data_collator,
        drop_last=model_training_args.dataloader_drop_last,
        num_workers=model_training_args.dataloader_num_workers,
        pin_memory=model_training_args.dataloader_pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )

    train_dataloader_outer = DataLoader(
        train_dataset_outer,
        batch_size=model_training_args.train_batch_size,
        collate_fn=data_collator,
        drop_last=model_training_args.dataloader_drop_last,
        num_workers=model_training_args.dataloader_num_workers,
        pin_memory=model_training_args.dataloader_pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=model_training_args.eval_batch_size,
        collate_fn=data_collator,
        drop_last=model_training_args.dataloader_drop_last,
        num_workers=model_training_args.dataloader_num_workers,
        pin_memory=model_training_args.dataloader_pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )

    eval_dataloader_mm = DataLoader(
        eval_dataset_mm,
        batch_size=model_training_args.eval_batch_size,
        collate_fn=data_collator,
        drop_last=model_training_args.dataloader_drop_last,
        num_workers=model_training_args.dataloader_num_workers,
        pin_memory=model_training_args.dataloader_pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )
    print(f"train_inner_num: {len(train_dataset_inner)}")
    print(f"train_outer_num: {len(train_dataset_outer)}")
    print(f"eval_num: {len(eval_dataset)}")
    print(f"eval_mm_num: {len(eval_dataset_mm)}")
    return (train_dataloader, train_dataloader_inner, train_dataloader_outer, eval_dataloader, eval_dataloader_mm,
            model, compute_metrics)


def train(model_training_args, model_args, data_training_args):
    seed = model_training_args.seed

    set_configuration(model_training_args)

    model_task_work_dir = os.path.join(
        model_training_args.work_dir, model_args.lora_type, model_args.model_name_or_path, data_training_args.task_name,
        str(model_args.lora_r), str(model_training_args.lr), str(model_training_args.arch_lr),
        str(model_training_args.weight_decay), str(model_training_args.arch_weight_decay), str(seed))
    check_dir(model_task_work_dir)

    (train_dataloader, train_dataloader_inner, train_dataloader_outer, eval_dataloader, eval_dataloader_mm, model,
     compute_metrics) = get_dataset_and_model(model_training_args, data_training_args, model_args)
    model_copy = deepcopy(model)

    arch = get_arch(model_args.model_name_or_path, model)

    class Inner(ImplicitProblem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def training_step(self, batch):
            data = batch
            data = {key: value for key, value in data.items()}
            _input = data['input_ids'].to(model_training_args.device)
            _target = data['labels'].to(model_training_args.device)
            _msk = data['attention_mask'].to(model_training_args.device)
            inner_loss_batch, logits = self.module(alphas=self.outer(), input_ids=_input, attention_mask=_msk,
                                                   labels=_target, return_dict=False)

            reg_loss = compute_direction_regularization(self.module, model_training_args.reg_loss_d)

            return inner_loss_batch + reg_loss

    class Outer(ImplicitProblem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def forward(self):
            return self.module()

        def training_step(self, batch):
            data = batch
            data = {key: value for key, value in data.items()}
            _input = data['input_ids'].to(model_training_args.device)
            _target = data['labels'].to(model_training_args.device)
            _msk = data['attention_mask'].to(model_training_args.device)
            outer_loss_batch, logits = self.inner.module(alphas=self.module(), input_ids=_input,
                                                         attention_mask=_msk, labels=_target, return_dict=False)
            return outer_loss_batch

    class BilevelEngine(Engine):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.metric_list = []
            self.query_variation_points = []
            self.value_variation_points = []

        @torch.no_grad()
        def validation(self):

            avg_lm_loss = AverageMeter()
            avg_lm_acc = AverageMeter()
            samples_num = len(eval_dataloader.dataset)
            batch_num = len(eval_dataloader.dataset) // model_training_args.eval_batch_size
            left_over = samples_num - len(
                eval_dataloader.dataset) // model_training_args.eval_batch_size * model_training_args.eval_batch_size
            with torch.no_grad():
                if data_training_args.task_name == "mnli":
                    losses_host: torch.Tensor = None
                    preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
                    labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
                    for eval_data in [eval_dataloader, eval_dataloader_mm]:
                        for idx, data in enumerate(eval_data):
                            data = {key: value for key, value in data.items()}

                            _input = data['input_ids'].to(model_training_args.device)
                            _target = data['labels'].to(model_training_args.device)
                            _msk = data['attention_mask'].to(model_training_args.device)
                            alphas = self.outer()
                            _loss = self.inner.module(alphas, input_ids=_input, attention_mask=_msk,
                                                      labels=_target, return_dict=False)
                            _loss, output = _loss
                            logits = output
                            logits = nested_detach(logits)
                            if _loss is not None:
                                losses = _loss.repeat(len(_target))
                                losses_host = losses if losses_host is None else torch.cat(
                                    (losses_host, losses), dim=0)
                            if logits is not None:
                                preds_host = logits if preds_host is None else nested_concat(
                                    preds_host, logits, padding_index=-100)
                            if _target is not None:
                                labels_host = _target if labels_host is None else nested_concat(
                                    labels_host, _target, padding_index=-100)

                    losses_host = nested_numpify(losses_host)
                    preds_host = nested_numpify(preds_host)
                    labels_host = nested_numpify(labels_host)
                    samples = len(eval_dataloader.dataset) + len(eval_dataloader_mm.dataset)
                    eval_loss = nested_truncate(losses_host, samples)
                    preds = nested_truncate(preds_host, samples)
                    label_ids = nested_truncate(labels_host, samples)
                    metrics = compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
                else:
                    losses_host: torch.Tensor = None
                    preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
                    labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
                    for idx, data in enumerate(eval_dataloader):
                        data = {key: value for key, value in data.items()}
                        _input = data['input_ids'].to(model_training_args.device)
                        _target = data['labels'].to(model_training_args.device)
                        _msk = data['attention_mask'].to(model_training_args.device)
                        _loss, logits = self.inner.module(
                            alphas=self.outer(), input_ids=_input, attention_mask=_msk, labels=_target,
                            return_dict=False)
                        ax = 1
                        if data_training_args.task_name == 'stsb':
                            ax = 0
                        preds = torch.argmax(logits, dim=ax)
                        if _loss is not None:
                            losses = _loss.repeat(len(_target))
                            losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
                        if logits is not None:
                            preds_host = logits if preds_host is None else nested_concat(preds_host, logits,
                                                                                         padding_index=-100)
                        if _target is not None:
                            labels_host = _target if labels_host is None else nested_concat(labels_host, _target,
                                                                                            padding_index=-100)
                        _acc = torch.mean((preds == _target).to(torch.float)).item()
                        if idx == batch_num:
                            avg_lm_loss.update(_loss, left_over / model_training_args.eval_batch_size)
                            avg_lm_acc.update(_acc, left_over / model_training_args.eval_batch_size)
                        else:
                            avg_lm_loss.update(_loss)
                            avg_lm_acc.update(_acc)
                    losses_host = nested_numpify(losses_host)
                    preds_host = nested_numpify(preds_host)
                    labels_host = nested_numpify(labels_host)
                    samples = len(eval_dataloader.dataset)
                    eval_loss = nested_truncate(losses_host, samples)
                    preds = nested_truncate(preds_host, samples)
                    label_ids = nested_truncate(labels_host, samples)
                    metrics = compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
                print(metrics)
            self.metric_list.append(metrics)
            return

    inner_optimizer = create_lora_adam_optimizer_from_args(model, model_training_args)
    inner_scheduler = create_optimizer_scheduler(inner_optimizer, model_training_args)
    outer_optimizer = create_arch_adam_optimizer_from_args(arch, model_training_args)
    outer_scheduler = create_optimizer_scheduler(outer_optimizer, model_training_args, is_arch=True)
    outer_config = Config(
        type="darts", retain_graph=True, gradient_accumulation=model_training_args.gradient_accumulation)
    inner_config = Config(
        type="darts", unroll_steps=model_training_args.unroll_step, gradient_accumulation=1)
    engine_config = EngineConfig(
        train_iters=model_training_args.train_iters, valid_step=model_training_args.valid_step)
    outer = Outer(name="outer", module=arch, optimizer=outer_optimizer, scheduler=outer_scheduler,
                  config=outer_config, train_data_loader=train_dataloader_outer)
    inner = Inner(name="inner", module=model, optimizer=inner_optimizer, scheduler=inner_scheduler,
                  config=inner_config, train_data_loader=train_dataloader_inner)
    problems = [outer, inner]
    l2u = {inner: [outer]}
    u2l = {outer: [inner]}
    dependencies = {"l2u": l2u, "u2l": u2l}
    engine = BilevelEngine(config=engine_config, problems=problems, dependencies=dependencies)
    engine.run()
    if model_training_args.retrain:
        retrain(model_training_args, model_args, data_training_args, model_copy, arch, compute_metrics,
                train_dataloader, eval_dataloader, eval_dataloader_mm)


def retrain(model_training_args, model_args, data_training_args, model, arch, compute_metrics, train_dataloader,
            eval_dataloader, eval_dataloader_mm):
    class RetrainProblem(ImplicitProblem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def training_step(self, batch):
            data = batch
            data = {key: value for key, value in data.items()}
            _input = data['input_ids'].to(model_training_args.device)
            _target = data['labels'].to(model_training_args.device)
            _msk = data['attention_mask'].to(model_training_args.device)
            inner_loss_batch, logits = self.module(alphas=arch(), input_ids=_input, attention_mask=_msk,
                                                   labels=_target, return_dict=False)

            reg_loss = compute_direction_regularization(self.module, model_training_args.reg_loss_d)

            return inner_loss_batch + reg_loss

    class RetainEngine(Engine):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.metric_list = []
            self.query_variation_points = []
            self.value_variation_points = []

        @torch.no_grad()
        def validation(self):
            avg_lm_loss = AverageMeter()
            avg_lm_acc = AverageMeter()
            samples_num = len(eval_dataloader.dataset)
            batch_num = len(eval_dataloader.dataset) // model_training_args.eval_batch_size
            left_over = samples_num - len(
                eval_dataloader.dataset) // model_training_args.eval_batch_size * model_training_args.eval_batch_size
            with torch.no_grad():
                if data_training_args.task_name == "mnli":
                    losses_host: torch.Tensor = None
                    preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
                    labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
                    for eval_data in [eval_dataloader, eval_dataloader_mm]:
                        for idx, data in enumerate(eval_data):
                            data = {key: value for key, value in data.items()}

                            _input = data['input_ids'].to(model_training_args.device)
                            _target = data['labels'].to(model_training_args.device)
                            _msk = data['attention_mask'].to(model_training_args.device)
                            _loss = self.problem.module(arch(), input_ids=_input, attention_mask=_msk,
                                                        labels=_target, return_dict=False)
                            _loss, output = _loss
                            logits = output
                            logits = nested_detach(logits)
                            if _loss is not None:
                                losses = _loss.repeat(len(_target))
                                losses_host = losses if losses_host is None else torch.cat(
                                    (losses_host, losses), dim=0)
                            if logits is not None:
                                preds_host = logits if preds_host is None else nested_concat(
                                    preds_host, logits, padding_index=-100)
                            if _target is not None:
                                labels_host = _target if labels_host is None else nested_concat(
                                    labels_host, _target, padding_index=-100)

                    losses_host = nested_numpify(losses_host)
                    preds_host = nested_numpify(preds_host)
                    labels_host = nested_numpify(labels_host)
                    samples = len(eval_dataloader.dataset) + len(eval_dataloader_mm.dataset)
                    eval_loss = nested_truncate(losses_host, samples)
                    preds = nested_truncate(preds_host, samples)
                    label_ids = nested_truncate(labels_host, samples)
                    metrics = compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
                else:
                    losses_host: torch.Tensor = None
                    preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
                    labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
                    for idx, data in enumerate(eval_dataloader):
                        data = {key: value for key, value in data.items()}
                        _input = data['input_ids'].to(model_training_args.device)
                        _target = data['labels'].to(model_training_args.device)
                        _msk = data['attention_mask'].to(model_training_args.device)
                        _loss, logits = self.problem.module(
                            alphas=arch(), input_ids=_input, attention_mask=_msk, labels=_target,
                            return_dict=False)
                        ax = 1
                        if data_training_args.task_name == 'stsb':
                            ax = 0
                        preds = torch.argmax(logits, dim=ax)
                        if _loss is not None:
                            losses = _loss.repeat(len(_target))
                            losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
                        if logits is not None:
                            preds_host = logits if preds_host is None else nested_concat(preds_host, logits,
                                                                                         padding_index=-100)
                        if _target is not None:
                            labels_host = _target if labels_host is None else nested_concat(labels_host, _target,
                                                                                            padding_index=-100)
                        _acc = torch.mean((preds == _target).to(torch.float)).item()
                        if idx == batch_num:
                            avg_lm_loss.update(_loss, left_over / model_training_args.eval_batch_size)
                            avg_lm_acc.update(_acc, left_over / model_training_args.eval_batch_size)
                        else:
                            avg_lm_loss.update(_loss)
                            avg_lm_acc.update(_acc)
                    losses_host = nested_numpify(losses_host)
                    preds_host = nested_numpify(preds_host)
                    labels_host = nested_numpify(labels_host)
                    samples = len(eval_dataloader.dataset)
                    eval_loss = nested_truncate(losses_host, samples)
                    preds = nested_truncate(preds_host, samples)
                    label_ids = nested_truncate(labels_host, samples)
                    metrics = compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
                print(metrics)
            self.metric_list.append(metrics)

            return

    model_training_args.lr = model_training_args.retrain_lr
    model_training_args.train_iters = model_training_args.retrain_iters
    model_training_args.max_step = model_training_args.retrain_iters
    model_training_args.warmup_step = int(model_training_args.train_iters * 0.1)
    model_training_args.scheduler = model_training_args.retrain_scheduler
    if model_training_args.retrain_weight_decay is not None:
        model_training_args.weight_decay = model_training_args.retrain_weight_decay

    optimizer = create_lora_adam_optimizer_from_args(
        model, model_training_args, grouped_parameters=model.parameters() if model_args.lora_type == 'ft' else None)
    scheduler = create_optimizer_scheduler(optimizer, model_training_args)
    config = Config(type="darts", unroll_steps=model_training_args.unroll_step, gradient_accumulation=1)
    engine_config = EngineConfig(
        train_iters=model_training_args.train_iters, valid_step=model_training_args.valid_step)
    problem = RetrainProblem(name="problem", module=model, optimizer=optimizer, scheduler=scheduler,
                             config=config, train_data_loader=train_dataloader)
    problems = [problem]
    l2u = {}
    u2l = {}
    dependencies = {"l2u": l2u, "u2l": u2l}
    engine = RetainEngine(config=engine_config, problems=problems, dependencies=dependencies)
    engine.run()


def main():
    print('Begin parsing arguments...')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ModelTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments = args[0]
    data_training_args: DataTrainingArguments = args[1]
    model_training_args: ModelTrainingArguments = args[2]
    model_training_args.max_step = model_training_args.train_iters
    model_training_args.warmup_step = int(model_training_args.train_iters * 0.1)
    print(f'model_args: {model_args}')
    print(f'data_training_args: {data_training_args}')
    print(f'model_training_args: {model_training_args}')
    print('End parsing arguments...')

    train(model_training_args, model_args, data_training_args)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
