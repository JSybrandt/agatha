# CBAG: Conditional Biomedical Abstract Generation 

This submodule contains the code necessary to train, evaluate, and generate text
with the CBAG model.

First, This guide assumes you have already installed Agatha, which you can do by
following the guide on the [root of this repo][base_agatha_link]. You will have
to install the extra dependencies in `requirements.txt` in order to run the CBAG
model.

```
cd <AGATHA_INSTALL_DIR>
pip install -r requirements.txt
```

Next, you will need to download the CBAG pretrained model. You have to options
to do so, either directly through [this Google Drive link][model_link], or with
the `gdown` utility that comes along with the Agatha install:

```
# Remeber where you place your file
cd <AGATHA_DATA_DIR>
# This will place cbag.tar.gz in AGATHA_DATA_DIR
gdown --id 19645oQA6MSnmV8tV2J_meF3iI2Puj41A
# Unzip the download, creates the "model" directory
tar -zxvf cbag.tar.gz
```

The model is going to expect the files within the newly downloaded directory to
be at the current working directory. If you `cd` to the model dir, you should
expect to find the following:

```
./model
 |- model.pt            # The actual CBAG model
 |- tokenizer.model     # The subword tokenizer.
 |- tokenizer.vocab
 |- condition_index.pkl # A mapping to/from mesh terms to integers.
```

Now we are ready to generate some text!

```python
import torch
from agatha.ml.abstract_generator import generation_util
# Load model from the current working directory
model = torch.load("./model.pt")
# Setup the tokenizer using the included helper data.
model.init_tokenizer()

# Make a new abstract!

new_abstract = generation_util.generate_new_text(
    model,
    # Each parameter is optional
    model.tokenizer.encode_for_generation(
        # This text seeds the generation
        initial_text="In this study",
        # When the result would have been written
        year=2019,
        # What MeSH terms to include. Note, many mesh terms are omitted due to
        # infrequent use within the training set.
        mesh_terms=[
            "D005947", # Glucose
        ]
    ),
    # By default this function outputs a single sentence. This flag keeps the
    # generation going until the end-of-abstract token.
    gen_whole_abstract=True,
)

>>>"""
   [In this study], we have investigated the effect of oxidative changes in
   metabolism of 6-ketoformate (bohd)--increases, glucose, glucose and glucose
   on glucose, glucose content, and glucose metabolism. the metabolism of
   2-deoxy-d-d-glucose was studied using a quantitative assay of glucose
   metabolism. this reaction had been observed when the presence of these
   glucose groups was increased or not observed. the effect of temperature,
   glucose-water, glucose-, potassium-glucose, and glucose was observed, as
   evidenced by the rate of glucose uptake by the in vitro and in vivo glucose
   production. on the other hand, the rate constant was increased, and a slower
   apparent life. the physiological decline in the number of glucose-free
   glucose groups was observed only after glucose-water treatment, while
   lactate-free metabolism was preserved. the metabolism of in vivo glucose
   with the same heat level is affected by glucose-water utilization but not by
   the in vitro glucose transport process.
   """
```

[model_link]:https://drive.google.com/open?id=19645oQA6MSnmV8tV2J_meF3iI2Puj41A
[base_agatha_link]:https://github.com/JSybrandt/agatha
