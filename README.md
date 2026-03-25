# bloqade-decoders

The QEC user interface providing integration with popular open-source decoders for the [Bloqade SDK](https://github.com/QuEraComputing/bloqade).

By default, the following decoders from the [`ldpc`](https://github.com/quantumgizmos/ldpc) package and
their corresponding interfaces are immediately available for decoding use upon installation of this package:

- BP+OSD - through `bloqade.decoders.BpOsdDecoder`
- BP+LSD - through `bloqade.decoders.BpLsdDecoder`
- Belief Find - through `bloqade.decoders.BeliefFindDecoder`

Interfaces also exist for the following optional decoders, which are not included as dependencies by default:

- MWPF - through `bloqade.decoders.MWPFDecoder` ([MWPF](https://github.com/yuewuo/mwpf))
- Tesseract - through `bloqade.decoders.TesseractDecoder` ([Tesseract](https://github.com/quantumlib/tesseract-decoder))
- MLE (Gurobi) - through `bloqade.decoders.GurobiDecoder`, finds the most likely error pattern via mixed-integer programming
- MLD (Table Lookup) - through `bloqade.decoders.TableDecoder`, builds a lookup table from sampled data

Sinter-compatible adapters for MLE and MLD are also available through `bloqade.decoders.sinter_interface`.

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

For MLE (a full [Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/) is needed for larger problems, but small models work with the [size-limited license](https://support.gurobi.com/hc/en-us/articles/360051597492-How-do-I-resolve-a-Model-too-large-for-size-limited-Gurobi-license-error) bundled with `gurobipy`):

```bash
pip install bloqade-decoders[mle]
```

For MLD:

```bash
pip install bloqade-decoders[mld]
```

You can combine multiple extras:

```bash
pip install bloqade-decoders[mwpf, tesseract, mle, mld, sinter]
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

### MLE / MLD Decoders

The `GurobiDecoder` takes a DEM directly (note: must use `decompose_errors=False`):

```python
from bloqade.decoders import GurobiDecoder

decoder = GurobiDecoder(dem)
corrections = decoder.decode(syndromes)
```

The `TableDecoder` can be constructed directly with a DEM and pre-computed counts, or
from a stim circuit which handles the sampling for you:

```python
from bloqade.decoders import TableDecoder

# from a circuit (samples shots to build the lookup table)
decoder = TableDecoder.from_stim_circuit(circuit, num_shots=100_000)

# or from pre-sampled detector-observable shots
decoder = TableDecoder.from_det_obs_shots(dem, det_obs_shots)

corrections = decoder.decode(syndromes)
```
