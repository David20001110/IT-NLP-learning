
# Day 16 - 使用 Transformers (7) - 處理多個序列(上半部)

這部分我們要來說明如何處理長度不同的多個序列或是序列太長的問題

### Batching (批次處理)

這個概念呢其實跟 Day12 最後給的完整範例有使用到，Batching 的概念是將多個句子一次性傳遞給模型的操作，那把它往外延伸在深度學習中的解釋是一種用於提高模型訓練效率和性能的技術。它涉及將多個數據樣本一次性輸入神經網絡，而不是一次處理單個樣本。

但是仍然可以建立一個批次，但這個批次只包含一個序列。例如，可以使用以下方式建立包含兩個相同序列的批次
```python
batched_ids = [ids, ids]
```

當嘗試將兩個或更多句子組合成批次時，這些句子可能具有不同的長度，我們就來看看會有甚麼問題
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequences = [
    "My mood is always so good",
    "Listening to my favorite music makes me feel so happy"
]
tokens = [tokenizer.tokenize(sequence) for sequence in sequences]
ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
print(ids[0])
print(ids[1])
```

```python
[1422, 6601, 1110, 1579, 1177, 1363]
[10817, 1158, 1106, 1139, 5095, 1390, 2228, 1143, 1631, 1177, 2816]
```
- 這裡我們得到兩個不同長度的 input IDs

接著我們去建立張量
```python
import torch

input_ids = torch.tensor(ids)
```

就會發現出錯了
![](C:\Users\User\Pictures\tensor_error.png)
- 是因為所有的 array 或是 tensor 都必須是矩形

#### 為了解決這個問題，我們要使用填充 (padding) 的方式來確保張量具有矩形的形狀，方法是向那些較短的句子中添加一個特殊的詞，這個詞通常被稱為"填充標記"（padding token）。

我們可以在 tokenizer.pad_token_id 找到 padding 的 token ID。
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print(tokenizer.pad_token_id)
```
這邊就會輸出它在做 padding 時會補上的數字是甚麼，這邊是0`(通常也都是0)`
```python
0
```

那其實我們要做 padding 的方法很簡單，細心的人應該可以發現在 Day12 時我有使用到。
```python
sequences = [
    "My mood is always so good",
    "Listening to my favorite music makes me feel so happy"
]
inputs = tokenizer(sequences, padding=True)
print(inputs['input_ids'])
```
- 直接在做 tokenizer 的時候，加入 padding=True 這個參數，它就會在做 tokenizer 時直接加上 padding 的 token ID 

```python
[
    [101,  1422, 6601, 1110, 1579, 1177, 1363,  102,    0,    0,    0,    0,   0], 
    [101, 10817, 1158, 1106, 1139, 5095, 1390, 2228, 1143, 1631, 1177, 2816, 102]
]
```
- 可以看到它的 input_ids 就會是一個矩形形狀了

下一章我們把剩下的的部分講完

### 參考資料
- <https://zhuanlan.zhihu.com/p/564816807>
- <https://huggingface.co/learn/nlp-course/chapter2/5?fw=pt>
