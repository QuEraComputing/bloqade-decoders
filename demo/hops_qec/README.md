# Score a lookup decoder on external syndrome histories

Run `python demo/hops_qec/score.py` from an environment with Bloqade Decoders installed. The example trains two lookup tables on the same 100,000 independent-fault samples from a Stim detector model. It then calls the public batched decoder on 64 held-out HOPS–QEC trajectories. One decoder receives the full history, and the other sees the last raw check outcomes and the perfect terminal boundary.

The included circuit specifies the independent Pauli training prior. The CSV contains simulated records for a three-data-qubit repetition memory with two dedicated ancillas and two extraction rounds. Each physical qubit couples to its own resonant damped mode with coupling 0.3 and number decay rate 2 in inverse circuit-time units. Both ancillas retain bath memory through reset. The HOPS–QEC run used IDs 0–127, seed 20261120, maximum timestep 0.02, hierarchy depth 3 and tensor truncation cutoff 1e-8. Odd IDs form this fixture.

D0 and D1 are the first two checks. D2 and D3 encode the changes at round two. D4 and D5 encode the changes between the last checks and the perfect final checks. The second logical observable in the circuit measures logical Z times reference Z. Each CSV row includes one sampled logical flip and its conditional Born probability before correction. Both enter only after predictions are frozen. Averaging conditional error integrates over the final measurement randomness. The sampled bit gives the ordinary count of incorrect decisions.

On this fixture both tables agree on every held-out prediction, with zero sampled errors and 0.005240 expected error per path. The example demonstrates interchange and scoring. Four distinct histories appear in the fixture, which limits its statistical reach.
