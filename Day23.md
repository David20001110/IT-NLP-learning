# Day 23 - 使用 Datasets 庫 (2) - 清理資料集功能

#### 在 datasets 中提供了許多內鍵方法讓我們整理資料，因為內容比較多所以我們拆成半，今天說前半段

![](C:\Users\User\Pictures\datasets_2.png)
_以上圖出自 Hugging Face 官方_

範例資料集我就直接全部使用 Day21 使用到的 `wikiann` 的中文數據集
```python
from datasets import load_dataset

wikiann_datasets = load_dataset('wikiann', 'zh', split='train')
wikiann_datasets['train']
wikiann_dataset[:3]['tokens']
```
```python
Dataset({
    features: ['tokens', 'ner_tags', 'langs', 'spans'],
    num_rows: 20000
})

[['2', '0', '0', '9', '年', '：', '李', '民', '基', '《', 'E', 't', 'e', 'r', 'n', 'a', 'l', '#', 'S', 'u', 'm', 'm', 'e', 'r', '》'], 
 ['#', '澳', '門', '大', '學', '田', '家', '炳', '教', '育', '研', '究', '所'],
 ['#', '大', '维', '多', '利', '亚', '沙', '漠']]
```

### 1. shuffle (洗牌)
#### 洗牌的主要目的是引入隨機性，對資料的排列順序進行混合，更好地泛化到未見過的資料，因為如果資料集按特定順序排序，模型可能會學習到這種順序的特定模式，而不是真正的資料特徵。

```python
dataset_shuffle = wikiann_dataset.shuffle(seed=42)
print(dataset_shuffle[:3]['tokens'])
```
- seed 是一個設定隨機數生成器的參數，確保在相同的種子值下生成的隨機性操作將產生相同的結果
```python
[['#', '李', '相', '秀', '#', '朴', '英', '淑'], 
 ["'", "'", "'", '威', '爾', '·', '普', '爾', '特', "'", "'", "'"], 
 ["'", "'", "'", '丁', '戈', "'", "'", "'", '（', '配', '音', '員', '：', '山', '口', '真', '弓', '（', '日', '本', '）', '）']]

```
- 可以看到這次資料的前三筆和原本的前三筆資料不同

### 2. split (分割)
#### 昨天有說到把資料分成`train`和`test`，我們除了自己提前定義好之外也可以透過內建方法直接做分割
```python
data_split = wikiann_datasets.train_test_split(test_size=0.1)
print(data_split)
```
- 我們是用 `train_test_split`方法，給定要分割的比例 `0.1`，也就是 9:1 的分割方式

```python
DatasetDict({
    train: Dataset({
        features: ['tokens', 'ner_tags', 'langs', 'spans'],
        num_rows: 18000
    })
    test: Dataset({
        features: ['tokens', 'ner_tags', 'langs', 'spans'],
        num_rows: 2000
    })
})
```
- 這樣它就會將原本的 20000 筆資料，在拆成 18000 的 `train data`和 2000 的 `test data`

### 3. select (選取)
#### 它的作用是從資料集中選取指定範圍的樣本，並允許我們去控制資料集的大小和內容，以滿足需求。

```python
# 第一種
data_select = wikiann_datasets.select(range(1000))
print(data_select)

# 第二種
indices = [100, 200, 300, 400, 500]
data_select = wikiann_datasets.select(indices)
print(data_select)
```
- 第一種直接給定我們所要選擇的資料數量
- 第二種可以選擇我需要哪幾筆，這裡是選擇第100、200、300、400、500筆，共五筆資料

```python
Dataset({
    features: ['tokens', 'ner_tags', 'langs', 'spans'],
    num_rows: 1000
})
Dataset({
    features: ['tokens', 'ner_tags', 'langs', 'spans'],
    num_rows: 5
})
```

### 4. filter (過濾)
#### 可以根據指定的條件篩選資料集中的樣本然，後創建一個符合需求的新資料集
```python
data = wikiann_datasets.filter(lambda x: len(x["tokens"]) > 20)
print(data)
```
- lambda 是一種用來定義簡單的匿名函數的方式，以當下這種情況來說，x 是每一個資料，當 x 這個資料的 'tokens' 的長度大於 20 的情況 `(對於 lambda 想多了解的人可以再去自己找)`

也可以寫成這樣
```python
def longer(x):
    return len(x["tokens"]) > 20

data = wikiann_datasets.filter(longer)
```
- 獨立的 function，只是比較麻煩一點

```python
Dataset({
    features: ['tokens', 'ner_tags', 'langs', 'spans'],
    num_rows: 4789
})
```
- 這邊的結果呢就告訴我們，'token' 欄位長度大於 20 的資料有 4789 筆

#### 再舉一個 filter 的範例
```python
data = wikiann_datasets.filter(lambda x: x["tokens"][0] == '#')
print(data)
```
- 條件是當 x 這個資料的 'tokens' 陣列的第一個位置是 '#' 的情況
```python
Dataset({
    features: ['tokens', 'ner_tags', 'langs', 'spans'],
    num_rows: 6566
})
```
### 參考資料
- <https://huggingface.co/learn/nlp-course/zh-CN/chapter5/3?fw=pt>


