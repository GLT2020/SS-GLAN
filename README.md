# SS-GLAN

This repository is used to store the code for the vulnerability detection methodology for the paper Line-level Smart Contract Vulnerability Detection via Semantic-Syntactic Feature Extraction and Global-Local Attention Network.

Installation environment:
```python
conda env create -f environment_linux.yml
# we also provide the environment for window
conda env create -f environment_win.yml
```

Quick Start:
```python
python main.py -M gat_lstm -DT graph_vectortype --mode train --epochs 100 --pca True -C access_control
```

Results Display:
```python
python print.py
```
