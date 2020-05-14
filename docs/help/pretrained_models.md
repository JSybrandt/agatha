Pretrained Models
=================

## 2015

[DOWNLOAD AGATHA 2015][agatha_2015]

The 2015 version of the Agatha model was trained on 2020-05-13. This model uses
all Medline abstracts dating before 2015-01-01. This model is used to validate
the performance of Agatha. Use this model to replicate our experiments from the
Agatha publication. Note, this model is a retrained version of the same model
used to report numbers.

### Contents

```
model_release/
  model.pt
  predicate_embeddings/
    embeddings_*.v5.h5
  predicate_entities.sqlite3
  predicate_graph.sqlite3
```

### Data Construction Parameters




## 2020

[DOWNLOAD AGATHA 2020][agatha_2020]

The 2020 version of the Agatha model was trained on 2020-05-04. This model uses
all available Medline abstracts as well as all available predicates in the most
up-to-date release of SemMedDB[semmeddb]. *This model does NOT contain any
COVID-19 related terms or customizations.*

### Contents

```
model_release/
  model.pt
  predicate_embeddings/
    embeddings_*.v5.h5
  predicate_entities.sqlite3
  predicate_graph.sqlite3
```

### Model Training Parameters
```python3
{
  'logger':                         True,
  'checkpoint_callback':            True,
  'early_stop_callback':            False,
  'gradient_clip_val':              1.0,
  'process_position':               0,
  'num_nodes':                      5,
  'num_processes':                  1,
  'gpus':                           '0,1',
  'auto_select_gpus':               False,
  'num_tpu_cores':                  None,
  'log_gpu_memory':                 None,
  'progress_bar_refresh_rate':      1,
  'overfit_pct':                    0.0,
  'track_grad_norm':                -1,
  'check_val_every_n_epoch':        1,
  'fast_dev_run':                   False,
  'accumulate_grad_batches':        1,
  'max_epochs':                     10,
  'min_epochs':                     1,
  'max_steps':                      None,
  'min_steps':                      None,
  'train_percent_check':            0.1,
  'val_percent_check':              0.1,
  'test_percent_check':             1.0,
  'val_check_interval':             1.0,
  'log_save_interval':              100,
  'row_log_interval':               10,
  'distributed_backend':            'ddp',
  'precision':                      16,
  'print_nan_grads':                False,
  'weights_summary':                'full',
  'num_sanity_val_steps':           3,
  'truncated_bptt_steps':           None,
  'resume_from_checkpoint':         None,
  'benchmark':                      False,
  'reload_dataloaders_every_epoch': False,
  'auto_lr_find':                   False,
  'replace_sampler_ddp':            True,
  'progress_bar_callback':          True,
  'amp_level':                      'O1',
  'terminate_on_nan':               False,
  'dataloader_workers':             3,
  'dim':                            512,
  'lr':                             0.02,
  'margin':                         0.1,
  'negative_scramble_rate':         10,
  'negative_swap_rate':             30,
  'neighbor_sample_rate':           15,
  'positives_per_batch':            80,
  'transformer_dropout':            0.1,
  'transformer_ff_dim':             1024,
  'transformer_heads':              16,
  'transformer_layers':             4,
  'validation_fraction':            0.2,
  'verbose':                        True,
  'warmup_steps':                   100,
  'weight_decay':                   0.01,
  'disable_cache':                  False
}
```

### Data Construction Parameters

```proto
cluster {
  address: "10.128.3.160"
  shared_scratch: "/scratch4/jsybran/agatha_2020"
  local_scratch: "/tmp/agatha_local_scratch"
}
parser {
  # This is the code for scibert in huggingface
  bert_model: "monologg/scibert_scivocab_uncased"
  scispacy_version: "en_core_sci_lg"
  stopword_list: "/zfs/safrolab/users/jsybran/agatha/data/stopwords/stopword_list.txt"
}
sentence_knn {
  num_neighbors: 25
  training_probability: 0.005
}
sys {
  disable_gpu: true
}
phrases {
  min_ngram_support_per_partition: 10
  min_ngram_support: 50
  ngram_sample_rate: 0.2
}
```

### Model Training Parameters

```python3
{
  'accumulate_grad_batches':         1
  'amp_level':                       'O1'
  'auto_lr_find':                    False
  'auto_select_gpus':                False
  'benchmark':                       False
  'check_val_every_n_epoch':         1
  'checkpoint_callback':             True
  'dataloader_workers':              3
  'dim':                             512
  'distributed_backend':             'ddp'
  'early_stop_callback':             False
  'fast_dev_run':                    False
  'gpus':                            '0,1'
  'gradient_clip_val':               1.0
  'log_gpu_memory':                  None
  'log_save_interval':               100
  'logger':                          True
  'lr':                              0.02
  'margin':                          0.1
  'max_epochs':                      10
  'max_steps':                       None
  'min_epochs':                      1
  'min_steps':                       None
  'negative_scramble_rate':          10
  'negative_swap_rate':              30
  'neighbor_sample_rate':            15
  'num_nodes':                       10
  'num_processes':                   1
  'num_sanity_val_steps':            3
  'num_tpu_cores':                   None
  'overfit_pct':                     0.0
  'positives_per_batch':             80
  'precision':                       16
  'print_nan_grads':                 False
  'process_position':                0
  'progress_bar_callback':           True
  'progress_bar_refresh_rate':       1
  'reload_dataloaders_every_epoch':  False
  'replace_sampler_ddp':             True
  'resume_from_checkpoint':          None
  'row_log_interval':                10
  'terminate_on_nan':                False
  'test_percent_check':              1.0
  'track_grad_norm':                 -1
  'train_percent_check':             0.1
  'transformer_dropout':             0.1
  'transformer_ff_dim':              1024
  'transformer_heads':               16
  'transformer_layers':              4
  'truncated_bptt_steps':            None
  'val_check_interval':              1.0
  'val_percent_check':               0.1
  'validation_fraction':             0.2
  'verbose':                         True
  'warmup_steps':                    100
  'weight_decay':                    0.01
  'weights_summary':                 'full'
}
```

[agatha_2015]:https://drive.google.com/open?id=1pDSaj2Ox2BRua5PmbJE5kcgCOfp40S5V
[agatha_2020]:https://drive.google.com/open?id=1GLKh9OJI0QVfeDZga2XlnMTa8bQGhp1F
[semmeddb]:https://skr3.nlm.nih.gov/SemMed/
