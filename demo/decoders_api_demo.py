# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: bloqade-decoders (3.14.3.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Getting Started
# This is a short demo for the interfaces for decoders defined in `bloqade-decoders` and how to use them.
#
# We will cover two concrete implementations of the decoders: a lookup-table decoder (`TableDecoder`) that constructs a table from each detector pattern to the frequency of the observable corrections, and a Most-Likely Error (MLE) decoder (`GurobiDecoder`) that solves for the most likely error that triggered the detector flips.

# %% [markdown]
# ## Constructing Decoders
# To construct a decoder, you pass in a `stim.DetectorErrorModel` object. Depending on the decoder, you can optionally pass in additional arguments to initialize the decoder.

# %%
# Define a stim detector error model used for decoding
import stim

demo_dem = stim.DetectorErrorModel("""
    error(0.1) D0
    error(0.09) D0 D1
    error(0.11) D1 L0
""")

# %%
# Construct a lookup-table decoder.
from bloqade.decoders import TableDecoder

lookup_table_decoder = TableDecoder(demo_dem)

# %%
# Construct a most-likely error decoder.
from bloqade.decoders import GurobiDecoder

mle_decoder = GurobiDecoder(demo_dem)

# %% [markdown]
# You can also optionally specify some initialization arguments that can be passed as keywords to initialize the decoders. For example, for the `TableDecoder`, you can specify the `num_shots` used to train it, and for the `GurobiDecoder`, you can specify the verbosity in logging (whether `verbose` is True).
# > By default, the `TableDecoder` uses 10,000 shots for training, and the `GurobiDecoder` has `verbose` set to False.

# %%
lookup_table_decoder_1_million_shots = TableDecoder(demo_dem, num_shots=1_000_000)

# %%
mle_decoder_verbose = GurobiDecoder(demo_dem, verbose=True)

# %% [markdown]
# ## Performing decoding
# Each decoder defines a `decode` method, which takes in a numpy array of booleans as detector bits. You can supply an array of booleans in for one detector pattern, or you can supply a batch of detector patterns.

# %%
# Use numpy for defining some mock detector patterns
import numpy as np

# %%
lookup_table_correction = lookup_table_decoder_1_million_shots.decode(
    detector_bits=np.array([True, True])
)

# %%
# Returns no flip for L0; the most frequently seen correction.
print(lookup_table_correction)

# %%
mle_correction = mle_decoder.decode(detector_bits=np.array([True, True]))

# %%
# Returns the observable flip associated with the most likely error; in this case, the error that flips D0 and D1 but does not flip L0.
print(mle_correction)

# %%
# We can also get corrections in batches by supplying multiple detector patterns.
lookup_table_correction_batched = lookup_table_decoder_1_million_shots.decode(
    detector_bits=np.array([[True, True], [False, True]])
)

# %%
print(lookup_table_correction_batched)

# %%
mle_correction_batched = mle_decoder.decode(
    detector_bits=np.array(
        [
            [True, True],
            [False, True],
        ]
    )
)

# %%
print(mle_correction_batched)

# %% [markdown]
# ## Obtaining Confidence from Decoding
# Using the `decode_confidence(detectors)` method, you can additionally obtain a confidence value with your decoding result. Understanding this confidence score will vary based on the decoder, but generally the scores will be in the interval `[0.0, 1.0]` where a higher value indicates a higher confidence in decoding.
# > For the TableDecoder, the confidence for a given observable correction is computed by the fraction of shots seen for that correction divided by the total number of shots seen for that detector; for the GurobiDecoder, the confidence is a normalized ratio of the probability of the most likely error and the second most likely error.
#
# The inputs for `decode_confidence` are the same as `decode`. `decode_confidence` additionally returns a float or a numpy array of floats representing the confidence score for each detector.
# > The default implementation of `decode_confidence` returns all corrections as equally confident (1.0).

# %%
lookup_table_correction_confidence = (
    lookup_table_decoder_1_million_shots.decode_confidence(
        detector_bits=np.array([True, True])
    )
)

# %%
# The confidence here is roughly the probability of the second error mechanism divided by the total probability of D0 and D1 triggering.
print(lookup_table_correction_confidence)

# %%
mle_correction_confidence = mle_decoder.decode_confidence(
    detector_bits=np.array([True, True])
)

# %%
# The confidence here is fairly large due to the probability of the most likely error being quite larger than the second most likely error.
print(mle_correction_confidence)

# %%
lookup_table_correction_confidence_batch = (
    lookup_table_decoder_1_million_shots.decode_confidence(
        detector_bits=np.array([[True, True], [False, True]])
    )
)

# %%
print(lookup_table_correction_confidence_batch)

# %%
mle_correction_confidence_batch = mle_decoder.decode_confidence(
    detector_bits=np.array([[True, True], [False, True]])
)

# %%
print(mle_correction_confidence_batch)

# %%
