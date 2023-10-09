# Day 24 - 使用 Datasets 庫 (4) - 清理資料集功能

接著前一天的部分繼續說後半部

![](C:\Users\User\Pictures\datasets_3.png)
_以上圖出自 Hugging Face 官方_

#### 今天使用到的範例資料集會和前一天不一樣，我們使用 hugging face 官方 Course 提供的`加州大學機器學習儲存庫的藥物審查資料集`。

```python
!wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
!unzip drugsCom_raw.zip
```
- 先下載並提取數據

```python
from datasets import load_dataset

data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
print(drug_dataset)
```
- \t 在 python 中是 tab 的意思，這裡使用是因為 .tsv 檔欄位和欄位之間的分割是用 tab
```python
DatasetDict({
    train: Dataset({
        features: ['Unnamed: 0', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 161297
    })
    test: Dataset({
        features: ['Unnamed: 0', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 53766
    })
})
```

### 5. Rename (重新命名)
#### 它主要目的就是針對 column 的名稱做更改，在這資料集中的欄位`Unnamed: 0`，它是屬於病人的編號或是ID。

```python
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
print(drug_dataset)
```
- 使用 rename_column 的方法，前面參數是原有的欄位名稱，後面參數是新的欄位名稱。
```python
DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 161297
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 53766
    })
})
```
- 欄位名稱已更改成 `pattient_id`

### 6. Remove (移除)
#### 主要是刪除不需要的 column，如果發現這個欄位是做訓練或是使用的過程中完全不需要用到就可以刪除


```python
drug_dataset = drug_dataset.remove_columns(['rating', 'usefulCount'])
print(drug_dataset)
```
- 使用 remove_columns 的方法，將 column: `rating`和`usefulCount`刪除
```python
DatasetDict({
    train: Dataset({
        features: ['Unnamed: 0', 'drugName', 'condition', 'review', 'date'],
        num_rows: 161297
    })
    test: Dataset({
        features: ['Unnamed: 0', 'drugName', 'condition', 'review', 'date'],
        num_rows: 53766
    })
})
```
- 這樣這兩個欄位就不會出現在 drug_dataset 當中

### 7. map (映射)

#### 主要作用是對資料集中的每個樣本應用一個自定義的函數，然後返回一個新的資料集，其中包含了應用了函數處理的樣本。這個方法讓你能夠對資料集進行各種轉換、清理、特徵工程等操作。


```python
def lowercase_condition(example):
    return {"condition": example["review"].lower()}

drug_dataset.map(lowercase_condition)
```
- 這裡建立一個 function 用來將 column: review 全部都轉成小寫，然後把它帶入 map 的方法

![](C:\Users\User\Pictures\datasets_4.png)
- 在輸出欄位上會跑出一個 map 的進度條，會發現它的速度有點緩慢

#### map() 參數 : batched
map()方法有一個batched參數，如果設定為True , map 函數將會分批執行所需要進行的操作（批次大小是可配置的，但預設為1,000）。例如，先前對 review 進行轉小寫的 map 函數運行需要一些時間`（您可以從進度條中讀取所用時間）`。我們可以透過使用列表推導同時處理多個元素來加快速度，作法也很簡單。

```python
new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [o.lower() for o in x["review"]]}, batched=True
)
```
- 將 `x['review].lower()`，轉成 `[o.lower() for o in x["review"]]`，這是因為當我們將 batched 設為 True 時，map 處理的方式就會是批次處理也就是打包成 List，因此我們需要將處理的方式做一個轉換。
- 這時候去執行時會發現`速度上快很多`

#### 我們可以透過 在最前方加上 %time 去計算當前這行 code 花了多少時間，(記得在執行的時候要讓整個 map 都在同一行)
```python
%time new_drug_dataset = drug_dataset.map( lambda x: {"review": x["review"]} )
# Wall time: 25.3 s

%time new_drug_dataset = drug_dataset.map( lambda x: {"review": [o.lower() for o in x["review"]]}, batched=True )
# Wall time: 1.61 s
```
- 可以看到時間上的差距

#### map() 參數 : num_proc
它用於控制在 map 在運行期間同時處理多少個元素。每個進程都可以處理一個元素，因此 num_proc 指定了同時處理多少個元素的並行數，`(指定並行處理（parallel processing）的進程數量)`。

```python
%time new_drug_dataset = drug_dataset.map( lambda x: {"review": [o.lower() for o in x["review"]]}, batched=True, num_proc=8)
# Wall time: 3.66 s

%time new_drug_dataset = drug_dataset.map( lambda x: {"review": [o.lower() for o in x["review"]]}, batched=True, num_proc=4)
# Wall time: 3.49 s

%time new_drug_dataset = drug_dataset.map( lambda x: {"review": [o.lower() for o in x["review"]]}, batched=True, num_proc=2)
# Wall time: 2.14 s
```

> 使用 num_proc 以加快處理速度通常是一個好主意，只要您使用的函數還沒有自己帶有的進行某種多進程處理的方法，但並非情況都是如此，像上面測試的三組都沒有原本單純加上 `batched=True` 來的快

### 8. flatten (平坦)
#### 用於將資料集的嵌套結構展平，將嵌套的多維資料轉換為一個扁平的資料集，`(把它放在最後是因為在原本資料集中沒有可以實作 flatten 的地方)`

```python
from datasets import load_dataset

squad = load_dataset('squad', split='train')
print(squad)
```
- 先載入資料集`(斯坦福問答資料集（Stanford Question Answering Dataset）)`
```python
Dataset({
    features: ['id', 'title', 'context', 'question', 'answers'],
    num_rows: 87599
})
```
在上述所有 columns 中的 answers 這個 column 它其實還可以再拆分成 `text`和`answer_start`

```python
squad.flatten()
```
```python
Dataset({
    features: ['id', 'title', 'context', 'question', 'answers.text', 'answers.answer_start'],
    num_rows: 87599
})
```

### 參考資料
- <https://huggingface.co/learn/nlp-course/chapter5/3?fw=pt>



