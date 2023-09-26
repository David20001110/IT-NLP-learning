# Day 11 - 使用 Transformers (2) - Tokenizer

今天我們要針對昨天說到的三個部份的第一個部分 Tokenizer 來做說明

![](C:\Users\User\Pictures\tokenizer.png)
_以上圖出自 Hugging Face 官方_

Tokenizer 的主要功能是將自然語言文本轉換為機器可理解的形式，Tokenizer 接受原始文本作為輸入，並將其分解成詞彙或子詞（subwords）的序列，每個詞彙或子詞通常對應到一個唯一的數字 ID。這個轉換過程稱為 "tokenization"，它將文本轉換成機器可理解的形式，使得模型能夠處理它們。


**直接用範例說明(這裡我們使用)，拆成兩部分**  
`(畢竟我們主要要講 NER 所以呢我們這裡直接使用預訓練的 BERT 中文模型)`


```python
from transformers import AutoTokenizer

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```
- 首先我們導入 Transformers 中的 AutoTokenizer 類別，這個類別允許你根據模型的名稱或模型 checkpoint 路徑來自動選擇並載入適當的 tokenizer。
- from_pretrained 方法會自動從 Hugging Face 模型庫下載並載入指定的 tokenizer，這裡我們使用預訓練的 BERT 中文模型。


```python
raw_inputs = [
    "我好想要出去玩",
    "今天天氣好熱，不適合出門",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```
- 使用載入的 tokenizer 將 raw_inputs 中的文本進行 tokenization，我們來解釋其中的參數
    - raw_inputs 是要處理的文本列表
    - padding=True 表示要對文本進行填充，以使它們的長度相等，以便進行批次處理。
    - truncation=True 表示如果文本長度超過模型的最大限制，則要進行截斷。
    - return_tensors="pt" 表示要返回 PyTorch 張量（tensor）格式的輸出，張量可以選擇PyTorch、TensorFlow 或是普通NumPy。

那我們來看一下張量的結果輸出什麼
```python
{
    'input_ids': tensor([
        [101, 2769, 1962, 2682, 6206, 1139, 1343, 4381, 102, 0, 0, 0, 0, 0],
        [101, 791, 1921, 1921, 3706, 1962, 4229, 8024, 679, 6900, 1394, 1139, 7271, 102]]), 
    'token_type_ids': tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}
```
輸出是一個包含三個鍵的字典
- input_ids : 每個元素是一個整數，代表對應詞彙的 ID。兩個句子都轉換成了一個整數序列。這些 ID 是基於預訓練模型的詞彙表所生成的。
- token_type_ids : 它表示每個 token 屬於哪個句子。在這個例子中，所有的 token 都被標記為 0，表示它們屬於同一句子。這在處理具有多個句子的輸入時很有用，可以區分不同句子的 token `(可以提前設好 token_type_ids 加入到參數當中)`。
- attention_mask : 它表示模型在處理輸入時應該關注哪些部分。1 表示模型應該關注該位置的 token，0 表示模型應該忽略該位置的 token。注意力遮罩通常用於處理填充的部分，以確保模型不會關注填充 token`(=不要理會那些多餘的填充)`。