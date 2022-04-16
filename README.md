# Sprufluo: Streaming Datasets on Web Datasets

This is a library to combine Hugging Face's Datasets library (and the associated hub) with the webdatasets library, 
using the former's extensive collection of well-documented data sets and the latter's more robust streaming support. It 
takes the form of several loader functions, as well as resources for sharding to individual machines.

## Desiderata

* Load a dataset like The Pile or C4 using information from HF's datasets, pass through to webdataset
* ad hoc mixtures of datasets
* [ ] make sharding work correctly when the number of shards doesn't divide evenly into the number of nodes
* support caching via the dl_manager of huggingface??!

