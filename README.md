# PaleoCNN – an explainable neural network for reconstructing paleoclimates

This code trains a bagged CNN ensemble on gridded rainfall and paleoclimate records. The trained CNN can then be used to reconstruct gridded rainfall over the full length of the paleoclimate records.

This code is set up to run on the Indian monsoon for 500 years, as described in our recent paper (https://cp.copernicus.org/articles/21/1/2025/).

The directory layout is as follows:
- Paleoclimate data (and parsing code) are stored in hrnh2k/, iso2k/, and pages2k/.
- Observations are stored in cet/ (for Central England Temperature) and monsoon/ (for South Asian monsoon). Gridded rainfall is not included due to file size constraints.
- Code to train, run, and explain the models are in final/.




