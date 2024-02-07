# To compute the embeddings

## Yamnet

run compute_yamnet_embeddings.py

## Birdnet

run:
python3 BirdNET-Analyzer/embeddings.py --i data/sons_al_balco_audios/audios_2020/ --o data/embeddings/birdnet/sons_al_balco_2020/
python3 BirdNET-Analyzer/embeddings.py --i data/sons_al_balco_audios/audios_2021/ --o data/embeddings/birdnet/sons_al_balco_2021/
python3 BirdNET-Analyzer/embeddings.py --i data/sabadell/ --o data/embeddings/birdnet/sabadell/
python3 BirdNET-Analyzer/embeddings.py --i data/granollers/ --o data/embeddings/birdnet/granollers/

## Perch

run perch/embed_audio.ipynb with the correct source_file_patterns and output_dir

## AudioMAE

....
