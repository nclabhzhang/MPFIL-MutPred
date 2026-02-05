# MPFIL-MutPred
This repository provides the implemention for the paper Multi-level Protein Feature Integrated Learning Framework for Predicting Mutation-Induced Protein-Protein Binding Affinity Changes.

Please cite our paper if our datasets or code are helpful to you ~ 😊

## Requirements
* Python 3.9
* Pytorch 1.12.1
* Transformers 4.21.3
* sentencepiece 0.2.0
* numpy 1.23.1
* openpyxl
* joblib


## Dataset
* [SKEMPI] Given in the dataset folder.
* [MPAD]  (http://compbio.clemson.edu/SAAMBE-MEM & https://web.iitm.ac.in/bioinfo2/mpad)

Processing the dataset:
```bash
cd data
unzip mapping.zip
cd ..
cd model
python generate_skempi.py
python generate_mpad.py
```

## Protein LLM Settings
* Download [pytorch_model.bin](https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main) and unzip the file in folder ./model/protein_llm/.


## Training & Evaluation for Ten-Fold-Cross-Validation
```bash
python main.py
```
