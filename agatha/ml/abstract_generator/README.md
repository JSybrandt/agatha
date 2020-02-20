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
gdown --id 1ISr-YfLew8Sy8O3aR5z8z-nFNEK57iz4
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
>>>'a high-level model for glucose in vitro. the potential concentration for use
>>>in man has to be established to define the contribution of glucose. the
>>>mechanism of action of glucose on glucose metabolism is studied by
>>>determining several biological variables. it could be found that glucose
>>>transport in tissues can be determined by a continuous washout of a single
>>>cycle and a time range of concentration from 20 to 200 ms for glucose. as is
>>>not based on guanidine (atp) in the concentration range of 3-11 mg, glucose
>>>release was studied by examining the relationship between (i) glucose uptake
>>>and (ii) glucose uptake into the medium. for this experiment the observed
>>>concentration dependent release of glucose into organs and is particularly
>>>proportional to ph in a time series to 20 hr. a second set of proteins
>>>(septic and -orylsulfonyl) has been found to be independent of saturation the
>>>blood glucose concentration of brain glucose. the isolol ratio for glucose is
>>>decreased in acid conditions and not temperature. the pernicu temperature
>>>difference of energy was further higher than that of insulin. in the presence
>>>of glucose, however, the maximum transport is still large.'
```

[model_link]:https://drive.google.com/file/d/1ISr-YfLew8Sy8O3aR5z8z-nFNEK57iz4/view?usp=sharing
[base_agatha_link]:https://github.com/JSybrandt/agatha
