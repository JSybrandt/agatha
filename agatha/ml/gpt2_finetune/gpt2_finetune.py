import torch
from agatha.ml.module import AgathaModule
from argparse import ArgumentParser, Namespace
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict, Any, Tuple
from agatha.ml.util.sqlite3_dataset import Sqlite3ValueDataset
from agatha.util.misc_util import Record
from agatha.ml.util.lamb_optimizer import Lamb


class Gpt2Finetune(AgathaModule):
  def __init__(self, hparams:Namespace):
    super(Gpt2Finetune, self).__init__(hparams)

    self.abstracts = None
    if hasattr(hparams, "abstract_db"):
      self.configure_paths(abstract_db=hparams.abstract_db)

    self.language_model = GPT2LMHeadModel.from_pretrained(
        self.hparams.baseline_model
    )
    self.tokenizer = GPT2Tokenizer.from_pretrained(
        self.hparams.baseline_model
    )
    # Defaults
    self.training_abstracts = None
    self.validation_abstracts = None
    self._abstract_tokenizer_dataset = None

  def configure_paths(
      self,
      abstract_db:Path,
  )->None:
    abstract_db = Path(abstract_db)
    assert abstract_db.is_file(), f"Failed to find {abstract_db}"
    self.abstracts = Sqlite3ValueDataset(abstract_db)

  def prepare_for_training(self)->None:
    assert self.abstracts is not None, \
        "Must call configure_paths before prepare_for_training"
    self._abstract_tokenizer_dataset = AbstractTokenizerDataset(
        abstracts=self.abstracts,
        tokenizer=self.tokenizer,
        max_length=self.hparams.max_length,
    )
    self.training_abstracts, self.validation_abstracts = (
        self.training_validation_split(
          self._abstract_tokenizer_dataset
        )
    )

  def train_dataloader(self)->torch.utils.data.DataLoader:
    self._vprint("Getting Training Dataloader")
    return self._configure_dataloader(
        self.training_abstracts,
        shuffle=True,
        batch_size=self.hparams.batch_size,
        collate_fn=self._abstract_tokenizer_dataset.collate,
    )

  def val_dataloader(self)->torch.utils.data.DataLoader:
    self._vprint("Getting Validation Dataloader")
    return self._configure_dataloader(
        self.validation_abstracts,
        shuffle=False,
        batch_size=self.hparams.batch_size,
        collate_fn=self._abstract_tokenizer_dataset.collate,
    )

  def forward(
      self,
      *args, **kwargs
  )->torch.FloatTensor:
    """
    Sends input to language model

    Args:
      input_ids: (batch_size) X (seq_length)

    Returns:
      (batch_size) X (seq_lenth) X (vocab_size)
    """
    return self.language_model.forward(tokens, *args, **kwargs)

  def _step(
      self,
      model_in:Dict[str, Any],
  )->Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Args:
      tokens: (batch_size) X (seq_length)
    """
    loss = self.language_model.forward(**model_in)[0]
    return (
        loss,
        dict( # pbar metrics
        )
    )

  def training_step(
      self,
      model_in:Dict[str, Any],
      batch_idx:int
  )->Dict[str, Any]:
    loss, metrics = self._step(model_in)
    return dict(
        loss=loss,
        progress_bar=metrics,
        log=metrics
    )

  def validation_step(
      self,
      model_in:Dict[str, Any],
      batch_idx:int
  )->Dict[str, Any]:
    loss, metrics = self._step(model_in)
    val_metrics = {f"val_{k}": v for k, v in metrics.items()}
    val_metrics["val_loss"] = loss
    return val_metrics

  def configure_optimizers(self):
    self._vprint("Configuring optimizers")
    return Lamb(
        self.parameters(),
        lr=self.hparams.lr,
        weight_decay=self.hparams.weight_decay,
    )

  @staticmethod
  def add_argparse_args(parser:ArgumentParser)->ArgumentParser:
    parser = AgathaModule.add_argparse_args(parser)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--abstract-db", type=Path)
    parser.add_argument("--baseline-model", type=str, default="gpt2")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--weight-decay", type=float)
    return parser



## Helper Functions


def abstract_record_to_string(abstract:Record)->str:
  assert "text_data" in abstract, \
      f"Abstract record missing text_data field"
  return " ".join([
    td["text"] for td in abstract["text_data"]
  ])


class AbstractTokenizerDataset(torch.utils.data.Dataset):
  def __init__(
      self,
      abstracts:torch.utils.data.Dataset,
      tokenizer:GPT2Tokenizer,
      max_length:int
  ):
    self.abstracts = abstracts
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __getitem__(self, idx:int)->List[int]:
    return self.tokenizer.encode(
        text=abstract_record_to_string(self.abstracts[idx]),
        add_special_tokens=True,
        max_length=self.max_length,
    )

  def __len__(self):
    return len(self.abstracts)

  def collate(self, tokens:List[List[int]])->Dict[str, Any]:
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor(t) for t in tokens],
        batch_first=True,
        padding_value=-100
    )
    attention_mask = (input_ids != -1).float()
    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids, # labels are shifted inside language model
    )

