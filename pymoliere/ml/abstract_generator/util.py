from typing import Dict, Any
from transformers import BertModel, BertTokenizer
import torch

class AbstractGenerator(BertModel):
  def __init__(self, config:Dict[str, Any]):
    super(AbstractGenerator, self).__init__(config)
    for param in self.parameters():
      param.requires_grad = False
    # This last layer converts the hidden layer to a predicted word
    self.last_hidden2voccab = torch.nn.Linear(
        config.hidden_size,
        config.vocab_size,
    )
    # Then we pick the word
    self.last_softmax = torch.nn.LogSoftmax(dim=1)

  def forward(self, x):
    # x is batch_first
    batch_size, seq_len = x.shape
    # The 0th element is the hidden layer per-word
    x = super(AbstractGenerator, self).forward(x)[0]

    # Convert to (batch_size*seq_len), hidden_size to apply linear layer to
    # each row
    x = x.view((batch_size*seq_len), self.config.hidden_size)
    # apply prediction
    x = self.last_hidden2voccab(x)
    x = self.last_softmax(x)
    # reconstitute correct shape
    x = x.view(batch_size, seq_len, self.config.vocab_size)
    return x
