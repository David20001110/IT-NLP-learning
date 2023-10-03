
# Day 17 - 使用 Transformers (8) - 處理多個序列(下半部)

#### 當我們已經知道要怎麼透過 padding 的方法矩形張量，那我們就可以將它傳入模型進行批量處理。

但是假如我們將兩個句子分開傳遞給模型和一起傳入模型做批次處理，會發現有一個地方不太一樣。
```python
import torch
from transformers import BertForSequenceClassification


ids_1 = torch.tensor([[101, 1422, 6601, 1110, 1579, 1177, 1363, 102]])
ids_2 = torch.tensor([[101, 10817, 1158, 1106, 1139, 5095, 1390, 2228, 1143, 1631, 1177, 2816, 102]])
all_ids = torch.tensor(
    [[101, 1422, 6601, 1110, 1579, 1177, 1363, 102, 0, 0, 0, 0, 0], 
    [101, 10817, 1158, 1106, 1139, 5095, 1390, 2228, 1143, 1631, 1177, 2816, 102]]
  )

model = BertForSequenceClassification.from_pretrained('bert-base-cased')

print(model(ids_1).logits)
print(model(ids_2).logits)
print(model(all_ids).logits)
```
- 這邊的例子我們直接使用昨天產生出的 input_ids，記得ids_1、ids_2 要多加一層 dimension

我們來看看結果
```python
ids_1 -> tensor([[0.2902, 0.8102]], grad_fn=<AddmmBackward0>)
ids_2 -> tensor([[0.2497, 0.7787]], grad_fn=<AddmmBackward0>)

all_ids -> tensor([[0.2761, 0.7023],
                   [0.2497, 0.7787]], grad_fn=<AddmmBackward0>)>)
```
- 可以看到有做 padding 的第一個 sequence，在單一序列和批量處理的時候得到的答案完全不一樣

#### 這是因為 Transformer 模型的關鍵特徵是注意力層，它對每個 token 進行上下文處理。這些注意力層會考慮到 padding token。為了在將不同長度的單獨句子傳遞到模型時或在傳遞帶有相同句子和填充的批次時獲得相同的結果，我們需要告訴這些注意力層忽略 padding token。這是通過使用 attention mask 來實現的，所以接下來我們要來講 attention mask。

### Attention masks (注意力遮罩)

注意力遮罩用於控制模型在處理序列數據時的關注範圍。當模型進行注意力計算時，它會參考注意力遮罩，以確定哪些位置的信息應該被納入計算，哪些位置的信息應該被忽略。

1表示應注意相應的標記，0表示不應注意相應的標記
```python
attention_mask = torch.tensor([
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])
output = model(all_ids, attention_mask=attention_mask)
print(output.logits)
```
- 設定注意力遮罩
- 將 attention_mask 以參數的方式傳入 model

```python
tensor([[0.2902, 0.8102],
        [0.2497, 0.7787]], grad_fn=<AddmmBackward0>)
```

- 這樣我們就能得到跟單個序列時一樣的 output

### Longer sequences (長序列)

#### Transformer 模型通常有處理序列的長度限制，大多數模型能夠處理的最大序列長度為 512 或 1024 個 tokens，如果試圖傳遞更長的序列給模型，它們可能會崩潰。

這個問題 Hugging Face 有提供兩個解決方法

1. **使用支持更長序列的模型**：一些特殊模型被設計來處理非常長的序列，例如 Longformer 和 LED。如果你的任務需要處理非常長的序列，它們具有更長的支持序列長度，有興趣建議查看這些模型`(我目前沒研究過)`。
2. **截斷序列**：另一種方法是截斷長度超過模型限制的序列。這可以通過指定 `max_sequence_length` 參數來實現，這將保留序列的前部分，並將其縮減為模型可以處理的最大長度。
    ```
   sequence = sequence[:max_sequence_length]
   ```
   
### 參考資料
- <https://huggingface.co/learn/nlp-course/chapter2/5?fw=pt>
- <https://zhuanlan.zhihu.com/p/448852278>