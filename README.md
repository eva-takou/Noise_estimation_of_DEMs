# Estimating noise of detector error models

The code developed here reproduces the results of the paper "Estimating decoding graphs and hypergraphs of memory QEC experiments" by Evangelia Takou and Kenneth R. Brown. It is used to estimate Pauli noise on decoding graphs or hypergraphs such as:

- Repetition code with bare-ancilla syndrome extraction (graph)
- Unrotated surface code with bare-ancilla syndrome extraction (graph)
- Repetition code with Steane-style syndrome extraction (hypergraph)
- Color code with bare-ancilla syndrome extraction (hypergraph)


In the presence of independent Pauli noise, a decoding graph or hypergraph of a memory experiment can be reconstructed based on the statistics of ancilla measurement outcomes. The statistics are collected by performing the same memory experiments for multiple experimental shots. 

For decoding graphs, one can count the number of times a single detector fires, $\langle v_i\rangle$, or two detectors fire together, $\langle v_iv_j \rangle$, across several experimental trials. These quantities suffice to reconstruct the error probabilities of the decoding graph.

For decoding hypergraphs, given that up to $m$ detectors fire simultaneously, we need to collect up to $m$-point correlators according to the structure of the DEM.

## Prerequisites

In addition to basic python libraries, the code uses the following open-source packages:
- Stim: https://github.com/quantumlib/Stim/tree/main
- Pymatching: https://github.com/oscarhiggott/PyMatching
- Color code decoder developed in the paper [Color code decoder with improved scaling for correcting circuit-level noise](https://quantum-journal.org/papers/q-2025-01-27-1609/), whose code is available in https://github.com/seokhyung-lee/color-code-stim. Note: this package requires Python version >= 3.11.
- Some issue might arise with cargo for macos. Steps to troubleshoot this are inside the jupyter notebooks for color code simulations.

## Contact
For any questions contact evangelia.takou@duke.edu.

## License
This code is licensed under 

