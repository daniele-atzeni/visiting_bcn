# Embeddings analysis of different pretrained audio models

## Preparatives

Before computing the embeddings, remove_unlabeled_audios.py must be run in order
to consider only Sons al Balco data with labels and removing labels attached to
non existing files

## To compute the embeddings

Take care of the parameters to use pretrained models (i.e. windows length and
windows shift). This choice will be later addressed in order to have embeddings
relative to file pieces of the same length. In the following the details of each
available model:

### Yamnet

run compute_yamnet_embeddings.py

### Birdnet

run:
python3 BirdNET-Analyzer/embeddings.py --i data/sons_al_balco_audios/audios_2020/ --o data/embeddings/birdnet/sons_al_balco_2020/
python3 BirdNET-Analyzer/embeddings.py --i data/sons_al_balco_audios/audios_2021/ --o data/embeddings/birdnet/sons_al_balco_2021/
python3 BirdNET-Analyzer/embeddings.py --i data/sabadell/ --o data/embeddings/birdnet/sabadell/
python3 BirdNET-Analyzer/embeddings.py --i data/granollers/ --o data/embeddings/birdnet/granollers/

### Perch

run perch/embed_audio.ipynb with the correct source_file_patterns and output_dir

## After computing the embeddings

Aggregate the embeddings to have consistent windows lengths across different pretrained models by calling make_consistent_emb.py

## Embeddings Analysis

In the file dimensionality_reduction_study.ipynb, we studied the hyper-parameters of different dimensionality reduction algorithm. We identified (by eye) the ideal configuration to compute 2D embeddings and do a qualitative embedding and distribution shifts analysis.
