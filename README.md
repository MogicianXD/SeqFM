# SeqFM

A pytorch implementation of [SeqFM](http://arxiv.org/abs/1911.02752):

Chen Tong, Yin Hongzhi, Hung Nguyen, Peng Wen-Chih, Li Xue, Zhou Xiaofang. (2020). Sequence-Aware Factorization Machines for Temporal Predictive Analytics. In ICDE'20, 2020

## Files in the folder
- `src/`
    - `BaseModel.py`: The base class encapsulates device initialization, train & test tasks for rank, classification and regression.
    - `layer.py`: Provides masked softmax function and `SelfAttention` layer along with `MultiHeadSelfAttention`
    - `dataset.py`: Derived from `torch.util.data.Dataset`
        - `FMDataset`: Dataset for rank and classification with negative sampling.
        - `RatingDataset`: Dataset for regression with numerical value per sample.
    - `util.py`: Provides `BPR` loss functions, `MRR` and `NDCG` functions, and other functions.
    - `SeqFM.py`: 
        - `ResFNN`: The residual forward-feeding network specially modified in this model.
        - `SeqFM`: The core model.
- `main.py`: You can look for descriptions of args with `-h` 
- `preprocess_neg.py`: export data with negative sampling.
- `preprocess_rate.py`: export data without negative sampling, but with ratings.
- `../benchmarks/datasets/`  