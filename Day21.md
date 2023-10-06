# Day 21 - 使用 Datasets 庫 (1) - Hugging Face Hub 的 dataset

#### 講完了 transformers 之後，也介紹了一些開源的資料集，接下來我們就是要來學怎麼做資料載入這件事，我在 Day5 的時候也稍微提到過這個，它提供了方便的API，用於下載、載入、處理和準備數據，使我們能夠輕鬆存取和使用訓練數據。

首先，打開 Jupyter Notebook 後，先來安裝套件
```python
pip install datasets
```

#### 我們直接來載入一個我第一次練習訓練時使用的 Hugging Face Hub 上用來訓練 NER 模型的數據集  

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
- 每個樣本都包含四個特徵 tokens、ner_tags、langs、spans
```python
datasets["train"].features
```
```python
{'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
 'ner_tags': Sequence(feature=ClassLabel(names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'], id=None), length=-1, id=None),
 'langs': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
 'spans': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None)}
```
- tokens：文本中的標記（單字或子詞）。
- ner_tags：與文字中每個標記相關的命名實體識別標籤，這裡包含了 LOC (地點)、PER (人名)、ORG (組織) 三種實體。 
- langs：文本的語言（中文）。
- spans：可能包含文字中命名實體的字元等級範圍

來看看完整的一筆資料裡面的內容
```python
print(datasets["train"][0])
```
```python
{
  'tokens': ['2', '0', '0', '9', '年', '：', '李', '民', '基', '《', 'E', 't', 'e', 'r', 'n', 'a', 'l', '#', 'S', 'u', 'm', 'm', 'e', 'r', '》'], 
  'ner_tags': [0, 0, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
  'langs': ['zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'zh'], 
  'spans': ['PER: 李 民 基']
}
```
- 這裡 tokens 的切法是直接將每一個字元都切成一個 token
-  tags 的部分是經過一層轉換，例如  `B-PER` -> `1`、`I-PER` -> `2`，這種方式可以節省儲存空間，還可以提高電腦在處理和計算這些標籤時的速度


```python
datasets = load_dataset('wikiann', 'zh', split='train')
print(datasets)
```
- 我們也可以透過使用參數`split`指定要載入的特定子集，這裡我選擇 `train`。

```python
Dataset({
    features: ['tokens', 'ner_tags', 'langs', 'spans'],
    num_rows: 20000
})
```
- 那就只會輸出這個子資料集


> wikiann : 這個資料集的用途是進行命名實體識別訓練 NER 任務，其中模型的目標是從文本中識別和標記命名實體（主要例如人名、地名、組織名等，這也是最常見的）。我們可以使用訓練集來訓練 NER 模型，使用驗證集來調整模型的超參數，然後使用測試集來評估模型的效能，而且有非常多種語言的資料集。
![](C:\Users\davidliu.ELAND\Pictures\wikiann.png)

### 參考資料
- <https://ithelp.ithome.com.tw/articles/10295522>
- <https://huggingface.co/datasets/wikiann/viewer/en/validation>