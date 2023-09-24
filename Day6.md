# NLP常用的Transformer模型 -- 簡介


**"Transformer模型"** 是一種深度學習架構，最初由Google於2017年提出。它是一種用於處理序列數據的神經網絡架構，特別在自然語言處理任務中表現出色。Transformer的核心思想是注意力機制（Attention Mechanism），它允許模型專注於序列中不同位置的重要信息，從而實現了在長序列中建立上下文感知的能力。

### Transformer的發展史

下面這張圖是來自[Hugging Face官網](https://huggingface.co/)的 Transformer 發展史

![](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_chrono.svg)

Transformer 模型最初是為了解決翻譯任務而設計的，但 Transformer 模型的成功啟發了更廣泛的應用領域，隨後推出了幾個有影響力的模型。

_(以下取自官方網站)_
- 2018 年 6 月 : GPT 第一個預訓練的 Transformer 模型，用於各種 NLP 任務並獲得極好的結果
- 2018 年 10 月 : BERT , 另一個大型預訓練模型，該模型主要在產生更好的句子摘要
- 2019 年 2 月 : GPT-2 , GPT 的改進（並且更大）版本
- 2019 年 10 月 : DistilBERT , BERT 的提煉版本，速度提高60%，記憶體減輕40%，但仍保留 BERT 97% 的效能
- 2019 年 10 月 :  BART 和 T5 , 兩個使用與原始 Transformer 模型相同架構的大型預訓練模型（第一個這樣做）
- 2020 年 5 月 : GPT-3 , GPT-2 的更大版本，無需微調即可在各種任務上表現良好

### Transformers 是語言模型

上述的所有Transformer 模型（GPT、BERT、BART、T5 等）都被訓練成語言模型。這意味著他們已經以無監督學習的方式接受了大量原始文本的訓練。無監督學習是一種訓練類型，其中目標是根據模型的輸入自動計算的。這意味著不需要人工來標記數據！

這種類型的模型可以對其訓練過的語言進行統計理解，但對於特定的實際任務並不是很有用。因此，一般的預訓練模型會經歷一個稱為遷移學習的過程。在此過程中，模型在給定任務上以監督方式（即使用人工註釋標籤）進行微調。

- 任務的一個例子是閱讀n個單字的句子，預測下一個單字。這被稱為因果語言建模，因為輸出取決於過去和現在的輸入。

![](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/causal_modeling.svg)

- 另一種是 masked language modeling，模型要去預測出遮住的單字

![](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/masked_modeling.svg)

---

上面這一段話我知道可能會有點看不懂(Transformers 是語言模型)，這段是官方教學的其中一段，我這邊先解釋幾個名詞的意思，再用比較短的話講述上述的重點

- **預訓練**：預訓練是指在大規模的文本數據上訓練模型，使其學習通用的語言表示。這一階段模型是無監督地學習的，
因此不需要人工標記的數據。預訓練的目標是讓模型理解自然語言中的語法、語義和文本結構。BERT、GPT等模型都是通過預訓練來學習語言表示的。  


- **微調**：微調是指在特定任務上使用`預訓練模型`，並根據任務的需要對模型進行有監督的調整。
這通常涉及到使用人工標註的數據，如文本分類、命名實體識別等任務。微調過程重點在將預訓練的通用語言理解能力轉化為`特定任務`的性能。


- **遷移學習**：遷移學習是一種學習方式，它強調了在不同但相關任務之間共享知識的能力。在NLP中，預訓練和微調就是一種遷移學習的應用。
預訓練模型在大規模數據上學到了通用的語言知識，然後通過微調來應用這些知識到特定任務，從而加速了模型在特定任務上的收斂和性能提升。


> 這段話的重點是強調了Transformer模型的兩個關鍵階段，預訓練和微調。理解這兩個概念對於有效利用預訓練模型來解決各種NLP任務至關重要。
這是深度學習在NLP領域取得成功的關鍵之一，因為它允許模型利用大規模的無監督數據進行學習，並將這些學習應用到各種具體任務中。

#說一下下一篇會講的是 Transformer 的架構

### 參考資料

- <https://zhuanlan.zhihu.com/p/535100411>
- <https://huggingface.co/learn/nlp-course/en/chapter1/4>