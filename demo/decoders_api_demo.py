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
# We will cover two concrete implementations of the decoders: a lookup-table decoder (`TableDecoder`) that constructs a table from each detector pattern to the frequency of the observable corrections, and a most-likely error (MLE) decoder (`GurobiDecoder`) that solves for the most likely error that triggered the detector flips.

# %% [markdown]
# ## Constructing decoders
# To construct a decoder, you pass in the `stim.DetectorErrorModel`. Depending on the decoder, you can optionally pass in arguments to initialize the decoder.

# %%
# Define a stim detector error model used for decoding
import stim

demo_dem = stim.DetectorErrorModel("""
    error(0.09) D0 L0 L1
    error(0.1) D0 L1 L2
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
lookup_table_decoder_1millionshots = TableDecoder(demo_dem, num_shots=1_000_000)

# %%
mle_decoder_verbose = GurobiDecoder(demo_dem, verbose=True)

# %% [markdown]
# ## Performing decoding
# Each decoder defines a `decode` method, which takes in a numpy array of detector bits. You can supply an array of bits in for one detector pattern, or you can supply a batch of detectors.

# %%
# Use numpy for defining some mock detector patterns
import numpy as np

# %%
lookup_table_correction = lookup_table_decoder_1millionshots.decode(
    detector_bits=np.array([True])
)

# %%
# Returns the correction for L1 and L2 as those observable flips; the most frequently seen corrections.
print(lookup_table_correction)

# %%
mle_correction = mle_decoder.decode(detector_bits=np.array([True]))

# %%
# Returns the most likely error; in this case, the error that triggers D0, L1, and L2.
print(mle_correction)

# %%
# We can also get corrections in batches by supplying multiple detector patterns.
lookup_table_correction_batched = lookup_table_decoder_1millionshots.decode(
    detector_bits=np.array([[True], [False]])
)

# %%
print(lookup_table_correction_batched)

# %%
mle_correction_batched = mle_decoder.decode(
    detector_bits=np.array(
        [
            [True],
            [False],
        ]
    )
)

# %%
print(mle_correction_batched)

# %% [markdown]
# ## Obtaining Confidence from Decoding
# Using the `decode_confidence(detectors)` method, you can additionally obtain a confidence value regarding how "confident" you are in your decoding. Understanding this confidence score will vary based on the decoder, but generally the scores will be in the interval `[0.0, 1.0]` where a higher value indicates a higher confidence in decoding.
# > For the TableDecoder, the confidence for a given observable correction is computed by the fraction of shots seen for that correction divided by the total number of shots seen for that detector; for the GurobiDecoder, the confidence is the ratio of the probability of the most likely error and the second most likely error.
#
# The inputs for `decode_confidence` is the same as `decode`. `decode_confidence` additionally returns a float or array of floats representing the confidence score for each detector.
# > The default implementation of `decode_confidence` returns all corrections as equally confident (1.0).

# %%
lookup_table_correction_confidence = (
    lookup_table_decoder_1millionshots.decode_confidence(detector_bits=np.array([True]))
)

# %%
# The confidence here is roughly (0.1 / (0.1 + 0.09)).
print(lookup_table_correction_confidence)

# %%
mle_correction_confidence = mle_decoder.decode_confidence(
    detector_bits=np.array([True])
)

# %%
# The confidence here is small due to the most likely and second most likely correction being similarly probable.
print(mle_correction_confidence)

# %%
lookup_table_correction_confidence_batch = (
    lookup_table_decoder_1millionshots.decode_confidence(
        detector_bits=np.array([[True], [False]])
    )
)

# %%
print(lookup_table_correction_confidence_batch)

# %%
mle_correction_confidence_batch = mle_decoder.decode_confidence(
    detector_bits=np.array([[True], [False]])
)

# %%
print(mle_correction_confidence_batch)

# %%
