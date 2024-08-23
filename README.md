# Repository of the AAAI Submission "_IOHunter_: Graph Foundation Model to Uncover Online Information Operations".

## Reproducibility Steps
### Data preprocessing
- Clone the repository in your local space
- Download the data from this [zenodo anonymous link](https://zenodo.org/records/13357621?preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTcyNDI4NDQ1OSwiZXhwIjoxNzM1NjAzMTk5fQ.eyJpZCI6IjcwNTdkZDE5LWViZWQtNGI2My04MzNhLTVmMGVkY2NiZTRkYiIsImRhdGEiOnt9LCJyYW5kb20iOiI2ZmVlM2IyZGQyMWJjMWM2NGFiMTY1Yjc3OWFiNjBjYiJ9.QTi4WgOHUwLbYw2u4NoL2MO61bR3a8CFRj3TsjGX2otvFKJsiPXU4_vwfhUIx4T_cv9esO6QD6h7TXmU_PPNZg) and unzip it in the main folder.
  - Your project tree should resemble this structure:
    - /src
    - /data/
    - /data/processed/UAE
    - /data/processed/cuba
    - /data/processed/russia
    - /data/processed/venezuela
    - /data/processed/iran
    - /data/processed/china

 ### Running scripts
 - Each running script takes as input several parameters, a typical run is the following:
   - ```python run_MultiModalGNN_CrossAttention.py --dataset russia --lr 1e-2 --early 30 --gnn sage```
 - Argument ```dataset``` accepts values in ```UAE, cuba, russia, venezuela, iran, china``` (same dataset names as in the paper).
 - Argument ```lr``` accepts continuous values and it represents the learning rate of the Adam optimizer.
 - Argument ```early``` is the number of epochs without improvement in Macro-F1 after which the early stopping halts the training.
 - Argument ```gnn``` accepts values in ```gcn, sage``` and represents whether the backbone GNN model is a GCN or a Sage.
 - You can also add the argument ```undersampling``` to specify whether you want to train the model in a data scarcity regimes. It accepts values in ```0.5, 0.75, 0.9, 0.95, 0.99, 0.999``` as used in the paper.

