# bloqade-decoders

The QEC user interface providing integration with popular open-source decoders for the [Bloqade SDK](https://github.com/QuEraComputing/bloqade).

By default, the following decoders from the [`ldpc`](https://github.com/quantumgizmos/ldpc) package and
their corresponding interfaces are immediately available for decoding use upon installation of this package:

- BP+OSD - through `bloqade.decoders.BpOsdDecoder`
- BP+LSD - through `bloqade.decoders.BpLsdDecoder`
- Belief Find - through `bloqade.decoders.BeliefFindDecoder`

Interfaces also exist for the Hypergraph Minimum-Weight Parity Factor [MWPF](https://github.com/yuewuo/mwpf) decoder as well as
the [Tesseract decoder](https://github.com/quantumlib/tesseract-decoder) but the decoders themselves are not included as a dependency and are instead optional.

You can install them separately or specify you would like them included with the `bloqade-decoders` installation through the
additional instructions below.

## Installation

For access to the `ldpc`-package originating decoders and their respective interfaces, just do the following:

```bash
pip install bloqade-decoders
```

To add the tesseract decoder you can do:

```bash
pip install bloqade-decoders[tesseract]
```

Or for MWPF do:

```bash
pip install bloqade-decoders[mwpf]
```

and for both, you can do:

```bash
pip install bloqade-decoders[mwpf, tesseract]
```

## Usage

The decoding interfaces are designed to align as closely as possible with the decoders
themselves in terms of arguments. The only major difference is you're expected to pass in
a Detector Error Model (DEM) to instantiate the interface.

Furthermore, all decoder interfaces are designed to accept the detector results of a single shot
OR a batch of shots as a numpy `ndarray` of booleans, with the result being the observable correction (also as an `ndarray` of booleans).

```python
from bloqade.decoders import BpOsdDecoder
import numpy as np
import stim

dem = stim.DetectorErrorModel("""
    error(0.1) D0
    error(0.1) D0 D1
    error(0.1) D1 L0
""")
# Pretend that circuit was executed twice,
# with two sets of detector results.
syndromes = np.array([[False, False], [False, True]])

# instantiate decoder, passing in desired arguments as you would
# the original decoder interface.
decoder = BpOsdDecoder(dem, bp_method="product_sum")

decoded_observable = decoder.decode(syndromes)
# decoded_observable should give you
# np.array([[False], [True]])
```
