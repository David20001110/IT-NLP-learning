# Day 10 - 使用 Transformers

在 Day5 的時候我們有提到 Hugging Face 的 Transformers 函式庫的一點介紹和例子，那這章會在深入它的相關使用和比較詳細的介紹。`(裡面的內容會基於Huggingface 推出的NLP 課程)`

為了因應數以百萬計到數千萬數十億參數的 Transformer 模型，訓練和部屬都是一項很龐大的任務，Transformers 可以解決這個問題，它提供一個API，透過它可以載入、訓練、微調和保存任何Transformer模型、輸入數據的處理。

### pipeline () 函數

pipeline 是在 Transformers 函式庫中一個基本也強大的函數，它提供了一個簡單的接口，使我們可以輕鬆的執行各種 NLP 任務`(它將模型與其必要的處理步驟連接起來)`，而且不用大量的程式碼就可以做到。

可以執行的 NLP 任務有以下，更多的可以到 [task summary](https://huggingface.co/docs/transformers/v4.33.2/en/task_summary) 查看
- 文本分類（Text Classification）
- 命名實體識別（Named Entity Recognition）
- 情感分析（Sentiment Analysis）
- 文本生成（Text Generation）
- 文本摘要（Text Summarization）
- 問答系统（Question-Answering）
- 語言翻譯（Translation）
- 文本匹配（Text Matching）
- 文本生成（Text Generation）
- 語音識別（Speech Recognition）

---

開始之前呢，可以先打開我在 Day4 介紹過的 Google Colab 來做使用

第一步先安裝 Transformers 函式庫
```python
!pip install transformers
```
或是可以安裝開發版本(先建議)
```python
!pip install transformers[sentencepiece]
```
直接用範例來說明，這裡初始化一个情感分析模型，可以用於分析文本的情感級性，正面 or 負面。
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")  # 情感分析 (是用英文短語)
classifier([
    "Your kindness and generosity are truly heartwarming.",
    "Their rude behavior and disrespect left a sour taste in my mouth."
])
```

當第一次運行的時候會先去下載 model 和 tokenizer 並且快取下來。
![](C:\Users\davidliu.ELAND\Pictures\model_download_1.jpg)

輸出的結果它就會告訴我們這個是正面還是負面的，以及所得 的分數多少。
![](C:\Users\davidliu.ELAND\Pictures\model_download_2.jpg)

### pipeline 內部的運作流程

接下來我們要來剖析它內部的處理流程。

>要講它內部組成部分的原因是因為，在之後使用自己訓練完的模型時我們不一定只單靠 pipeline 這個方法就可以很好的去使用，所以把它連接起來的部分在細拆來介紹。


可以拆成以下三個部分
1. 分詞器的預處理
2. 模型的傳遞輸入
3. 對生成的輸出進行後處理

![](C:\Users\davidliu.ELAND\Pictures\pipeline_2.png)
_以上圖出自 Hugging Face 官方_

用實際的範例來說就是，首先先使用分詞器將原始文字轉換為模型可以理解的數字，經過模型之後產生出 logits，最後將這輸出的結果在轉換為標籤和分數。

![](C:\Users\davidliu.ELAND\Pictures\pipeline_3.png)
_以上圖出自 Hugging Face 官方_

> 接下來幾天我們會再深入講這三個部分，畢竟還有很多天 ^_^

### 參考資料
- <https://zhuanlan.zhihu.com/p/448852278>
- <https://huggingface.co/learn/nlp-course/en/chapter2/2?fw=pt>
