# Day 15 - 使用 Transformers (6) - 單一序列和模型批次不匹配(補充)

這一章節我們要補充如何將單個序列轉換成適合模型輸入的格式，以及處理維度或批次不符合的問題

#### 在講 Tokenizer 的第一天裡面我有舉到一個單個序列的例子，我們將它搬過來

```python
import torch
from transformers import AutoTokenizer

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
sequence = "Using a Transformer network is simple"

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
# ------------------------------- 以上是 Day11範例

# 轉換成張量
input_ids = torch.tensor(ids)
```

在這裡將產生出的 ids 經過 torch.tensor 轉換為 PyTorch 張量，以下是最後產生的 input_ids `這時候看似都沒有問題`
```python
tensor([ 7993,   170, 13809, 23763,  2443,  1110,  3014])
```

但假如我們將傳給 model 要輸出 logits 的時候我們來看看會發生什麼事
```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(checkpoint)
model(input_ids)
```
會出現以下錯誤
![](C:\Users\User\Pictures\error.png)


- 原因是因為模型的輸入必須是批次 (batch) 的資料，而不僅僅是單個句子的資料。
- 預期輸入是一個大小為 [batch_size, sequence_length] 的張量，其中 batch_size 是批次大小，sequence_length 是序列中 token 的數量。但在這裡，我們只提供了一個單獨的句子，因此缺少批次維度

#### 我們來仔細觀察 tokenizer 完整流程，會發現它不僅僅是將輸入 ID 的列表轉換成張量，它還在其上方添加了一個維度

我們將 sequence 直接傳入 tokenizer
```python
tokenized_inputs = tokenizer(sequence, return_tensors="pt")
print(tokenized_inputs["input_ids"])
```
可以看到輸出時多了一層 dimension
```python
tensor([[  101,  7993,   170, 13809, 23763,  2443,  1110,  3014,   102]])
```

#### 既然知道原因了那我們就將原本的範例重新調整就可以了
```python
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = BertForSequenceClassification.from_pretrained(checkpoint)

sequence = "Using a Transformer network is simple"

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
# --------------------------------------------
# 添加批次維度
# 作法 1
input_ids = torch.tensor(ids)
input_ids = input_ids.unsqueeze(0)

# 作法 2
input_ids = torch.tensor([ids])

# --------------------------------------------
output = model(input_ids)
print(output.logits)
```
- 這裡提供兩個作法都可以將單個序列轉換成批次的格式，以便將其傳遞給模型進行處理。

這樣最後就能得到 logits 了
```python
tensor([[-0.0946,  0.3242]], grad_fn=<AddmmBackward0>)
```

恭喜今天是三十天的一半了（￣︶￣）↗　

### 參考資料
- <https://zhuanlan.zhihu.com/p/564816807>
- <https://huggingface.co/learn/nlp-course/chapter2/5?fw=pt>
- 


