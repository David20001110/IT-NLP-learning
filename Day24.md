# Day 24 - 使用 Datasets 庫 (3) - 清理資料集功能

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

#### 1. 全部轉成小寫
```python
def lowercase_condition(example):
    return {"condition": example["review"].lower()}

drug_dataset.map(lowercase_condition)
```
- 這裡建立一個 function 用來將 column: review 全部都轉成小寫，然後把它帶入 map 的方法

![](C:\Users\User\Pictures\datasets_4.png)
- 在輸出欄位上會跑出一個 map 的進度條，會發現它的速度有點緩慢

#### 2. 將所有HTML 字元進行轉義

```python
import html

text = "I&#039;m a transformer called BERT"
html.unescape(text)
# output: I'm a transformer called BERT
```
- 這是 html 模組中的 unescape 函數，這個函數會將 HTML 實體轉換回原始的字元。
```python
drug_dataset = drug_dataset.map(
    lambda x: {"review": html.unescape(x["review"])}
)

new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True
)
```