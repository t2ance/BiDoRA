from dataclasses import dataclass, field
from typing import Optional

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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "It is used if the dataset is not a task of GLUE dataset"},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Additional information such as subset name"},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    inner_training_portion: Optional[float] = field(
        default=0.8,
        metadata={"help": "the training set portion of inner"},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    ########
    lora_type: Optional[str] = field(
        default='bidora',
        metadata={"help": "The type of LoRA to use."},
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA alpha"},
    )
    lora_r: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA r"},
    )
    lora_dropout: Optional[float] = field(
        default=0.
    )
    ########
    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of LoRA parameters."},
    )
    # hyperparameters for adapter
    apply_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply adapter or not."},
    )
    adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of adapter parameters."},
    )
    adapter_type: Optional[str] = field(
        default='houlsby',
        metadata={"help": "houlsby or pfeiffer"},
    )
    adapter_size: Optional[int] = field(
        default=64,
        metadata={"help": "8, 16, 32, 64"},
    )
    # hyperparameters for bitfit
    apply_bitfit: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply bitfit or not."},
    )
    # hyperparameters for bidora
    reg_loss_wgt: Optional[float] = field(
        default=0.0,
        metadata={"help": "Regularization Loss Weight"},
    )
    masking_prob: Optional[float] = field(
        default=0.0,
        metadata={"help": "Token Masking Probability"},
    )
    arch_type: Optional[str] = field(
        default='softmax',
        metadata={"help": "type of architecture"},
    )
    arch_std: Optional[float] = field(
        default=0.0,
        metadata={"help": "init std for architecture"},
    )
    lamb: Optional[float] = field(
        default=1e-4,
        metadata={"help": "lambda for regularizer"},
    )
    cl_bottom: Optional[float] = field(
        default=1e-6,
        metadata={"help": "clamp bottom for architecture"},
    )


@dataclass
class ModelTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    train_batch_size: Optional[int] = field(
        default=16,
        metadata={"help": "train batch size"},
    )
    eval_batch_size: Optional[int] = field(
        default=16,
        metadata={"help": "train batch size"},
    )
    test_batch_size: Optional[int] = field(
        default=16,
        metadata={"help": "train batch size"},
    )
    seed: Optional[int] = field(
        default=0,
        metadata={"help": "random seed"},
    )
    # hyperparameters for optimizer
    lr: Optional[float] = field(
        default=4e-5,
        metadata={"help": "model learning rate"},
    )
    arch_lr: Optional[float] = field(
        default=4e-5,
        metadata={"help": "arch learning rate"},
    )
    weight_decay: Optional[float] = field(
        default=0.,
        metadata={"help": "model weight decay"},
    )
    arch_weight_decay: Optional[float] = field(
        default=0.,
        metadata={"help": "arch weight decay"},
    )
    correct_bias: Optional[bool] = field(
        default=True,
        metadata={"help": "correct bias"},
    )
    adam_epsilon: Optional[float] = field(
        default=1e-6,
        metadata={"help": "adam epislon"},
    )
    no_decay_bias: Optional[bool] = field(
        default=False,
        metadata={"help": "no decay bias"},
    )
    adam_beta1: Optional[float] = field(
        default=0.9,
        metadata={"help": "adam beta 1"},
    )
    adam_beta2: Optional[float] = field(
        default=0.98,
        metadata={"help": "adam beta 2"},
    )
    # hyperparameters for learning rate scheduler
    scheduler: Optional[str] = field(
        default='linear',
        metadata={"help": "scheduler type"},
    )
    max_step: Optional[int] = field(
        default=5000,
        metadata={"help": "max learning steps"},
    )
    warmup_step: Optional[int] = field(
        default=300,
        metadata={"help": "warmup steps"},
    )
    i_steps: Optional[str] = field(
        default='0',
        metadata={"help": "interval_steps for cycle scheduler"},
    )
    i_lrs: Optional[str] = field(
        default='0.00025',
        metadata={"help": "interval_lrs for cycle scheduler"},
    )
    do_train: Optional[bool] = field(
        default=True,
        metadata={"help": "do training"},
    )
    do_test: Optional[bool] = field(
        default=True,
        metadata={"help": "do test"},
    )
    do_eval: Optional[bool] = field(
        default=True,
        metadata={"help": "do eval"},
    )
    do_predict: Optional[bool] = field(
        default=True,
        metadata={"help": "do predict"}
    )
    use_deterministic_algorithms: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use deterministic algorithms."}
    )
    output_dir: Optional[str] = field(
        default='./mnli/model',
        metadata={"help": "output path"}
    )
    overwrite_output_dir: Optional[bool] = field(
        default=True,
        metadata={"help": "overwrite output path"}
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "use fp16"}
    )
    n_gpu: Optional[int] = field(
        default=1,
        metadata={"help": "num of gpus"}
    )
    local_rank: Optional[int] = field(
        default=-1,
        metadata={"help": "local rank"}
    )
    cls_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "cls dropout"}
    )
    dataloader_num_workers: Optional[int] = field(
        default=0,
        metadata={"help": "dataloader num workers"}
    )
    dataloader_drop_last: Optional[bool] = field(
        default=False,
        metadata={"help": "drop last"}
    )
    dataloader_pin_memory: Optional[bool] = field(
        default=False,
        metadata={"help": "pin memory"}
    )
    unroll_step: Optional[int] = field(
        default=5,
        metadata={"help": "inner unroll step"}
    )
    train_iters: Optional[int] = field(
        default=50000,
        metadata={"help": "train iters"}
    )
    valid_step: Optional[int] = field(
        default=5000,
        metadata={"help": "valid interval"}
    )
    work_dir: Optional[str] = field(
        default="/checkpoint",
        metadata={"help": "work dir"}
    )
    arch_init_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "arch init checkpoint path"}
    )
    save_step: Optional[int] = field(
        default=1000,
        metadata={"help": "checkpoint save step"}
    )
    device: Optional[str] = field(
        default='cuda:0',
        metadata={"help": "the device to run the model"}
    )
    gradient_accumulation: Optional[int] = field(
        default=1,
        metadata={"help": "gradient accumulation to improve the upper layer"}
    )
    train_train: Optional[bool] = field(
        default=False,
        metadata={"help": "using 'train-train' mode, or not"}
    )
    retrain: Optional[bool] = field(
        default=False,
        metadata={"help": "fine-tune the lower level using all the data, or not"}
    )
    retrain_lr: Optional[float] = field(
        default=0,
        metadata={"help": "model re-learning rate"},
    )
    retrain_iters: Optional[int] = field(
        default=0,
        metadata={"help": "retrain iters"}
    )
    retrain_train_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "retrain batch size"},
    )
    retrain_scheduler: Optional[str] = field(
        default='linear',
        metadata={"help": "retrain scheduler type"},
    )
    retrain_weight_decay: Optional[float] = field(
        default=None,
        metadata={"help": "retrain weight decay"},
    )
    reg_loss_d: Optional[float] = field(
        default=0.0,
        metadata={"help": "regularization for direction"},
    )


if __name__ == '__main__':
    ...
