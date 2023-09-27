# Day 11 - 使用 Transformers (2) - Tokenizer

今天我們要針對昨天說到的三個部份的第一個部分 Tokenizer 來做說明

![](C:\Users\User\Pictures\tokenizer.png)
_以上圖出自 Hugging Face 官方_

Tokenizer 的主要功能是將自然語言文本轉換為機器可理解的形式，Tokenizer 接受原始文本作為輸入，並將其分解成詞彙或子詞（subwords）的序列，每個詞彙或子詞通常對應到一個唯一的數字 ID。這個轉換過程稱為 "tokenization"，它將文本轉換成機器可理解的形式，使得模型能夠處理它們。

---

從自然語言文本再轉到數字 ID 我們可以在拆開，來看下面這張圖
![](C:\Users\User\Pictures\tokenizer_2.png)
_以上圖出自 Hugging Face 官方_

### 接下來我們使用預訓練的 BERT 模型的範例來示範
#### 1. 首先將文字拆分為單字

通常這個動作稱為標記 (token)
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)
```
- 首先我們導入 Transformers 中的 AutoTokenizer 類別，這個類別允許你根據模型的名稱或模型 checkpoint 路徑來自動選擇並載入適當的 tokenizer。
- from_pretrained 方法會自動從 Hugging Face 模型庫下載並載入指定的 tokenizer，這裡我們使用預訓練的 BERT 模型。
- 使用載入的 tokenizer，將輸入序列 sequence 標記化為詞元

看一下結果
```python
['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']
```
- 對詞進行拆分後，直到獲得可以用其詞彙表表示的標記(token)。
- 每個詞元對應輸入序列中的一個單詞或一部分單詞，並且 ## 前綴表示子詞分割

#### 2. 再來將這些標記轉換為數字

```python
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
```
- 使用載入的 tokenizer 的 convert_tokens_to_ids 方法，將經過標記化的詞元列表 tokens 轉換為它們在詞彙表中的詞彙 ID。

```python
[7993, 170, 11303, 1200, 2443, 1110, 3014]
```
- 輸出將是一個整數列表，其中每個整數對應於輸入詞元列表中的詞元在模型詞彙表中的 ID。

#### 3. 將詞彙 ID 添加特殊標記


```python
final_input = tokenizer.prepare_for_model(ids)

print(final_input['input_ids'])
```
- 預訓練模型通常需要在輸入的文本之前和之後添加特殊標記，如 [CLS]（用於分類任務）和 [SEP]（用於分隔文本或標記句子邊界）。這些標記對於模型的正確操作非常重要。
- 這裡使用prepare_for_model方法，將詞彙 ID 清單ids轉換為適合輸入到模型的形式。

```python
[101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102]
```
- 最後的輸出呢會因應使用的`預訓練模型`而不同，我們使用的是 BERT Tokenizer，所以最後的 final 輸出前加上了 `101` 後面加上了 `102`

下一章我們把剩下的 Tokenizer 的部分講完

### 參考資料
- <https://zhuanlan.zhihu.com/p/448852278>
- <https://huggingface.co/learn/nlp-course/en/chapter2/2?fw=pt>



