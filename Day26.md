# Day 25 - NER 模型訓練 (1)

#### NER 屬於 Token classification 的其中一種標記任務的分類，可以說　Token classification 的整個目的就是要為句子中的詞和字分配標籤，接下來我們將在 NER 任務上對預訓練模型 (BERT) 進行微調。`(會分為上下兩半)`

`在BERT上進行的微調`是指將預先訓練好的BERT模型進一步訓練以適應特定任務或領域的需求。BERT 是一個在大規模文本語言庫上進行預訓練的深度學習模型，但是它是一個通用的語言模型，可能需要進一步的訓練來適應特定的自然語言處理任務，那這個動作就稱為`微調`
- 這次訓練的數據集使用 `中文的 wikiann` 
- 預訓練模型使用 `bert-base-chinese`

 ![](C:\Users\User\Pictures\ner_1.png)
 輸入的詞句經過模型處理標示出具有實體代表的詞彙

#### 那就開始吧 `( 一樣打開 Colab )`


#### 1. 首先我們先下載相關的套件

   ```pyhton
   !pip install datasets evaluate transformers[sentencepiece]
   !pip install accelerate
   ```
   - `accelerate` 用於加速 PyTorch 模型訓練
   - `evaluate` 用於評估模型性能
   

#### 2. 載入資料集和查看數據資料
   ```python
   from datasets import load_dataset

   datasets = load_dataset('wikiann', 'zh')
   ```
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
   
   ```python
   label_list = datasets["train"].features["ner_tags"].feature.names
   # ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
   ```
   - 查看它的標籤列表
   - 這邊在說明一次標籤所對應的意思
     - O 不對應任何實體。
     - B-PER / I-PER 對應於人名實體的開頭/內部。
     - B-ORG / I-ORG 對應於組織名稱實體的開頭/內部。
     - B-LOC / I-LOC 對應於地名實體的開頭/內部。
   
   #### 我們也可以把他轉成 pandas 的格式隨機抽樣讓我們能比較好查看資料 `(這裡取七筆資料)`
   ```python
   from datasets import ClassLabel, Sequence
   import random
   import pandas as pd
   from IPython.display import display, HTML
   
   def show_random_elements(dataset, num_examples=7):
       picks = []
       for _ in range(num_examples):
           pick = random.randint(0, len(dataset)-1)
           while pick in picks:
               pick = random.randint(0, len(dataset)-1)
           picks.append(pick)
   
       df = pd.DataFrame(dataset[picks])
       for column, typ in dataset.features.items():
           if isinstance(typ, ClassLabel):
               df[column] = df[column].transform(lambda i: typ.names[i])
           elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
               df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
       display(HTML(df.to_html()))
   ```
   ```pyhton
   show_random_elements(datasets["train"])
   ```
   ![](C:\Users\User\Pictures\ner_2.png)

#### 3. 建立分詞器
```python
from transformers import BertTokenizerFast

model_checkpoint = "bert-base-chinese"
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
```
- 導入 BertTokenizerFast 類，使用 from_pretrained 方法加載 `bert-base-chinese`
   
#### 4. 處理數據

#### 對樣本資料進行分詞，放入標籤並標記ID，以便進行序列標註任務的訓練或評估。我們針對以下 function 拆開解析
```python
 label_all_tokens = True
 
 def tokenize_and_align_labels(examples):
     tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
     labels = []
     for i, label in enumerate(examples["ner_tags"]):
         word_ids = tokenized_inputs.word_ids(batch_index=i)
         previous_word_idx = None
         label_ids = []
         for word_idx in word_ids:
             if word_idx is None:
                 label_ids.append(-100)
             elif word_idx != previous_word_idx:
                 label_ids.append(label[word_idx])
             else:
                 label_ids.append(label[word_idx] if label_all_tokens else -100)
             previous_word_idx = word_idx
 
         labels.append(label_ids)
 
     tokenized_inputs["labels"] = labels
     return tokenized_inputs
```

- ```python
  tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
  ```
  - 使用分詞器進行分詞，`truncation=True`表示針對超過模型的最大長度會進行截斷，`is_split_into_words=True`表示原本的輸出文本已經是單詞狀態
- ```python
  word_ids = tokenized_inputs.word_ids(batch_index=i)
  ```  
  - 會回傳一個 list，裡面是獲取在分詞後的文本中標記所對應的詞或子詞的索引。`ex: [None, 0, 1, 2, 3, 4, None]`
- ```python
  label_ids = []
  for word_idx in word_ids:
     if word_idx is None:
         label_ids.append(-100)
     elif word_idx != previous_word_idx:
         label_ids.append(label[word_idx])
     else:
         label_ids.append(label[word_idx] if label_all_tokens else -100)
     previous_word_idx = word_idx
  ```
  - 根據`word_ids`標記中的每個索引與標籤的對應關係，將標籤轉換為`label_ids`。`ex: [-100, 0, 3, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, -100]`
- ```python
  tokenized_inputs["labels"] = labels
  ```
  - 將`labels列表`加入`tokenized_inputs`字典中的鍵`labels`下，以便將標籤與分詞後的輸入關聯起來。

#### 使用之前教過的 map 方法，將我們創建好的函數應用於數據集中的每一個數據資料，使用批次處理提高處理速度
```python
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
```
 ![](C:\Users\User\Pictures\ner_3.png)

### 參考資料
- <https://huggingface.co/learn/nlp-course/chapter7/2?fw=pt>