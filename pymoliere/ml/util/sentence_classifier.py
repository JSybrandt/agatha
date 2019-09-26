IDX2LABEL = [
  'abstract:background',
  'abstract:conclusions',
  'abstract:methods',
  'abstract:objective',
  'abstract:results',
]
LABEL2IDX = {l: i for i, l in enumerate(IDX2LABEL)}
NUM_LABELS = len(IDX2LABEL)
SCIBERT_OUTPUT_DIM = 768

