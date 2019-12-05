import torch
import pickle
import sentencepiece as spm
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any, Dict
import math
from random import shuffle
import pytorch_lightning as pl
from pymoliere.ml.abstract_generator.pickle_dataset import KVStoreDictDataset, LoadWholeKVStore
from pymoliere.ml.abstract_generator.lamb_optimizer import Lamb
import os


class AbstractGenerator(pl.LightningModule):
  def __init__(self,
      total_embed_size:int,
      vocab_size:int,
      padding_idx:int,
      vocab_start_idx:int,
      embedding_dim:int,
      max_text_length:int,
      num_attention_heads:int,
      num_encoder_layers:int,
      num_decoder_layers:int,
      intermediate_dropout:float,
      intermediate_feedforward_dim:int,
      training_data_dir:Path,
      batch_size:int,
      warmup_steps:int,
      learning_rate:float,
      dataset_workers:int=4,
  ):
    """
    Learns to generate following text given sliding windows across abstracts.
    """
    super(AbstractGenerator, self).__init__()

    self.dataset_workers= dataset_workers
    self.batch_size = batch_size
    self.training_data_dir = training_data_dir
    self.warmup_steps = warmup_steps
    self.learning_rate = learning_rate
    self.max_text_length = max_text_length+1
    self.vocab_size = vocab_size
    self.padding_idx = padding_idx
    self.vocab_start_idx = vocab_start_idx

    self.embeddings = torch.nn.Embedding(
        total_embed_size,
        embedding_dim,
        padding_idx=padding_idx,
    )

    # Positional encoding is (Max Sequence Length, 1, Embedding Dim)
    self.register_buffer(
        "positional_encoding",
        self.generate_positional_encoding(
          max_text_length=self.max_text_length,
          embedding_dim=embedding_dim,
        )
    )

    self.transformer = torch.nn.Transformer(
        d_model=embedding_dim,
        nhead=num_attention_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=intermediate_feedforward_dim,
        dropout=intermediate_dropout,
    )

    # This mask has -inf for all values that follow the target input
    # Of size (text, text)
    self.register_buffer(
        "text_attention_mask",
        self.transformer.generate_square_subsequent_mask(self.max_text_length),
    )

    self.predicted_text = torch.nn.Linear(
        embedding_dim,
        vocab_size,
    )

    self.softmax = torch.nn.LogSoftmax(dim=2)
    self.loss_fn = torch.nn.NLLLoss()

  def generate_positional_encoding(
      self,
      max_text_length:int,
      embedding_dim:int,
  )->torch.FloatTensor:
    # Dim must be even
    assert embedding_dim % 2 == 0

    # Returns a (seq_len, emb) tensor
    positional_encodings = []
    for pos in range(max_text_length):
      positional_encodings.append([])
      for i in range(0, int(embedding_dim / 2)):
        # Even index
        positional_encodings[-1].append(
          math.sin(pos / (10000 ** (2 * i / embedding_dim)))
        )
        # Odd index
        positional_encodings[-1].append(
          math.cos(pos / (10000 ** (2 * i / embedding_dim)))
        )
    result = torch.FloatTensor(
        positional_encodings,
    ).unsqueeze(1)
    assert result.shape == (max_text_length, 1, embedding_dim)
    return result

  def forward(
      self,
      context:torch.LongTensor,
      text:torch.LongTensor,
  ):
    # C is the sequence length of context
    # T is the sequence length of text
    # B is the batch size
    assert len(context.shape) == 2
    assert len(text.shape) == 2
    # Batch size is consistent
    assert context.shape[1] == text.shape[1]
    # T is consistent, and less than the expected max
    assert text.shape[0] <= self.max_text_length

    # Adds an additional E-sized embedding vector for each long
    context_emb = self.embeddings(context)
    text_emb = self.embeddings(text)
    # Need to merge text and position
    text_length = text.shape[0]
    txt_typ_pos = text_emb + self.positional_encoding[text_length, :, :]

    encoded = self.transformer(
        src=context_emb,
        src_key_padding_mask=(context==self.padding_idx).t_(),
        tgt=txt_typ_pos,
        tgt_key_padding_mask=(text==self.padding_idx).t_(),
        tgt_mask=self.text_attention_mask[:text_length,:text_length],
    )

    predicted_text = self.predicted_text(encoded)

    # produce softmax results across "vocab"
    return {
        "text": self.softmax(predicted_text),
    }

  # Added for pytorch-lightning
  def training_step(self, batch, batch_nb):
    model_in, expected = batch
    predicted = self.forward(**model_in)

    mask = (expected['text'] != self.padding_idx)
    # matrix of probability vectors over vocab
    masked_predicted_text = (
        predicted['text'][mask]
        .view(-1, predicted['text'].shape[2])
    )
    # vector of indices from 0 to vocab size
    masked_expected_text = (
        expected['text'][mask]
        .view(-1)
        -self.vocab_start_idx
    )
    loss = self.loss_fn(masked_predicted_text, masked_expected_text)
    accuracy = (
        (masked_predicted_text.argmax(dim=1) == masked_expected_text)
        .float()
        .mean()
    )
    # Return dict of metrics
    return {
        'loss': loss,
        'progress_bar': {
          'loss': loss,
          'text_accuracy': accuracy,
        },
        'log': {
          'loss': loss,
          'text_accuracy': accuracy,
        }
    }


  @pl.data_loader
  def train_dataloader(self):
    dataset=KVStoreDictDataset(self.training_data_dir)
    #dataset=LoadWholeKVStore(self.training_data_dir)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    return torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=self.batch_size,
        collate_fn=AbstractGenerator.collate,
        #num_workers=self.dataset_workers,
    )

  def configure_optimizers(self):
    optimizer = Lamb(
        self.parameters(),
        # facebook paper says linear growth with batch size
        lr=self.learning_rate,
        weight_decay=0.01,
    )
    return optimizer

  def optimizer_step(
      self,
      epoch_idx,
      batch_idx,
      optimizer,
      optimizer_idx,
      second_order_closure=None
  ):
    # warm up lr
    if  self.trainer.global_step < self.warmup_steps:
      lr_scale = min(
          1.,
          float(self.trainer.global_step + 1)/float(self.warmup_steps)
      )
      for pg in optimizer.param_groups:
        pg['lr'] = lr_scale * self.learning_rate
    optimizer.step()
    optimizer.zero_grad()


  def collate(batch:List[Dict[str,List[int]]]):
    # Group the elements together into tensors
    padded_tensors = {
        field_name: torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(b[field_name]) for b in batch]
        )
        for field_name in batch[0].keys()
    }
    return (
        # Model Input
        {
          "text": padded_tensors["text"],
          "context": padded_tensors["context"]
        },
        # Target
        {"text": padded_tensors["shifted_text"]},
    )

  def init_ddp_connection(self, proc_rank, world_size):
    print(f"Initing {proc_rank}/{world_size}")
    torch.distributed.init_process_group(
        'gloo',
        rank=proc_rank,
        world_size=world_size
    )
