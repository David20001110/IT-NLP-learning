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

### 載入和保存

剛剛我們有使用到了載入這個方法 from_pretrained()，現在我們要來說保存 save_pretrained() 函数用於保存已經訓練或微調過的模型、tokenizer 和配置文件，以便以後再次使用。
```python
# 指定要保存的目錄路徑
save_directory = "my_fine_tuned_model"

# 使用 save_pretrained() 函數保存模型、分詞器和配置文件
tokenizer.save_pretrained(save_directory)
```
保存完成後，你將在指定目錄下看到分詞器文件，當然這個方法也可以使用在保存模型

這邊要順編補充一個就是我們除了可以使用 AutoTokenizer 這個類別之外，我們也可使用 BertTokenizer
```python
from transformers import AutoTokenizer

# 換成
from transformers import BertTokenizer
```
- AutoTokenizer 是通過模型的名稱或 checkpoint 路徑來自動選擇適當的 tokenizer。這使得它非常方便，因為你只需提供模型的名稱或路徑，而不必明確指定 tokenizer 的類型。
- BertTokenizer 則是一個專門用於 BERT 相關模型的 tokenizer。你需要明確知道你要使用的是哪種模型，然後實例化相應的 tokenizer
但是兩者在執行同一個 BERT 的 checkpoint 時不會有什麼差



