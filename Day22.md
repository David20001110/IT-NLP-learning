# Day 22 - 使用 Datasets 庫 (2) - 遠端和本地資料


#### Datasets 它提供了 loading scripts 來讓我們可以載入本地和遠端的資料集。

它支援以下幾種資料格式

|         資料格式         |  loading scripts  |                           舉例                           |
|:--------------------:|:-----------------:|:------------------------------------------------------:|
|      CSV & TSV       |        csv        |     load_dataset("csv", data_files="my_file.csv")      |
|      Text files      |       text        |     load_dataset("text", data_files="my_file.txt")     |
|  JSON & JSON Lines   |       json        |    load_dataset("json", data_files="my_file.jsonl")    |
|  Pickled DataFrames  |      pandas       | load_dataset("pandas", data_files="my_dataframe.pkl")  |
_表格取自 Hugging Face 官方_

#### 最主要就是要提供`格式名稱`和`檔案路徑 or URL`的參數
這邊要補充說明 `JSON` 和 `JSON Lines` 哪裡不一樣

- JSON: 它是屬於巢狀式的結構化資料，它使用物件和陣列的結構來表示資料
    ```python
    {
      "user": {
        "id": 1,
        "name": "John Doe",
        "email": "john.doe@example.com",
        "isStudent": true,
        "courses": [
          {
            "id": 101,
            "title": "Introduction to Programming",
            "instructor": "Jane Smith"
          },
          {
            "id": 102,
            "title": "Data Structures and Algorithms",
            "instructor": "Tom Brown"
          }
        ]
      }
    }
    ```
- JSON Line(簡稱 JSONL) : 每行都包含一個獨立的 JSON 物件。每行的資料是獨立的
    ```python
    {"id": "2834", "tokens": ["星", "巴", "克", "小", "圓", "零", "錢", "包"], "ner_tags": ["B-BRAND", "I-BRAND", "I-BRAND", "O", "O", "B-ITEM", "I-ITEM", "I-ITEM"]}
    {"id": "4516", "tokens": ["e", "x", "c", "e", "l", " ", "漸", "層", "魅", "色", "腮", "紅"], "ner_tags": ["B-BRAND", "I-BRAND", "I-BRAND", "I-BRAND", "I-BRAND", "O", "O", "O", "O", "O", "B-ITEM", "I-ITEM"]}
    {"id": "8103", "tokens": ["m", "e", "k", "o", "魔", "翹", "美", "型", "纖", "長", "睫", "毛", "膏"], "ner_tags": ["B-BRAND", "I-BRAND", "I-BRAND", "I-BRAND", "O", "O", "O", "O", "O", "O", "B-ITEM", "I-ITEM", "I-ITEM"]}  
    ```

### 遠端資料的載入

#### 載入一個遠端的 text 檔的資料
```python
from datasets import load_dataset

dataset_url = "https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt"
text_dataest = load_dataset('text', data_files=dataset_url)
print(text_dataest['train'][:5])
```
- 先給定格式名稱 `text`，再將遠端檔案以 `URL` 的方式傳遞給 load_dataset
```python
{
    'text': ['First Citizen:',
    'Before we proceed any further, hear me speak.',
    '',
    'All:',
    'Speak, speak.']
}
```
- txt 檔案的存取方式就是將每一行拆成一個 string，所以空白行也是一行

#### 載入一個遠端的 json 檔的資料
```python
from datasets import load_dataset

dataset_url = "https://github.com/crux82/squad-it/raw/master/SQuAD_it-train.json.gz"

squad_it_dataset = load_dataset("json", data_files=dataset_url, field="data")
print(squad_it_dataset)
```
- 這個 json 格式使用巢狀格式,所有文字都儲存在data檔案中。所以我們可以透過指定參數 field 來載入資料集
```python
DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 442
    })
})
```

### 本地資料的載入

```python
from datasets import load_dataset

squad_it_dataset = load_dataset("json", data_files="SQuAD_it-test.json", field="data")
print(squad_it_dataset)
```
- 就是將 URL 換成資料的名稱

```python
DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 48
    })
})
```

#### 但是我們真正想要的是包括`train`和`test`的 DatasetDict 對象
像是 SQuAD_it-train.json 和 SQuAD_it-test.json 建立成一個完整的 DatasetDict 對象，這樣的話就可以使用 Dataset.map() 函數同時處理訓練集和測試集。因此我們提供參數 data_files 的字典,將每個分割名稱映射到與該分割相關聯的資料

```python
from datasets import load_dataset

data_files = { "train" : "SQuAD_it-train.json" , "test" : "SQuAD_it-test.json" }
squad_it_dataset = load_dataset( "json" , data_files=data_files, field= "data" )
print(squad_it_dataset)
```

```python
DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 442
    })
    test: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 48
    })
})  
```
這就是我們需要的資料。我們可以應用各種預處理技術來清理資料、標記評論等。

#### 如果你不只 train 和 test，你還有分出一個 validation 的檔案，那你就可以拆成
  ```python
  data_files = { 
    "train" : "train.json" , 
    "test" : "test.json",
    "validation" : "validation.json"
  }
  ```

### 參考資料
- <https://huggingface.co/learn/nlp-course/zh-CN/chapter5/2?fw=pt>
