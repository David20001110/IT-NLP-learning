# Day 21 - 使用 Datasets 庫 (1)

#### 講完了 transformers 之後，也介紹了一些開源的資料集，接下來我們就是要來學怎麼做資料載入這件事，我在 Day5 的時候也稍微提到過這個，它提供了方便的API，用於下載、載入、處理和準備數據，使我們能夠輕鬆存取和使用訓練數據。

首先，打開 Jupyter Notebook 後，先來安裝套件
```python
pip install datasets
```

我們直接來載入一個 Hugging Face Hub 上用來訓練 NER 模型的數據集  

```python
from datasets import load_dataset

datasets = load_dataset('wikiann', 'zh')
print(datasets)
```
- 我們使用 load_dataset 載入 `wikiann` 數據集的中文版本

```python
DatasetDict({
    validation: Dataset({
        features: ['tokens', 'ner_tags', 'langs', 'spans'],
        num_rows: 10000
    })
    test: Dataset({
        features: ['tokens', 'ner_tags', 'langs', 'spans'],
        num_rows: 10000
    })
    train: Dataset({
        features: ['tokens', 'ner_tags', 'langs', 'spans'],
        num_rows: 20000
    })
})
```
- 可以看到它主要將 dataset 分為三種 `train` (訓練集 20000 個)、`validation` (驗證集 10000 個)、`test` (測試集 10000 個)
- 每個樣本包含以下特徵
  - tokens：文本中的標記（單字或子詞）。
  - ner_tags：與文字中每個標記相關的命名實體識別標籤。
  - langs：文本的語言（中文）。
  - spans：可能包含文字中命名實體的字元等級範圍

> 那這個資料集的用途呢是進行命名實體識別訓練 NER 任務，其中模型的目標是從文本中識別和標記命名實體（主要例如人名、地名、組織名等，這也是最常見的）。我們可以使用訓練集來訓練 NER 模型，使用驗證集來調整模型的超參數，然後使用測試集來評估模型的效能。