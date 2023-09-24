Transformer 模型架構主要由兩個區塊組成，左側是 Encoder(編碼器)，右側是 Decoder(解碼器)

![](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_blocks.svg)

(_這邊先簡單說明，下一章節回詳細解析兩個部份的內部結構_)
- **Encoder** : 編碼器會接收輸入的序列，負責將輸入序列（例如文本或其他序列數據）轉換為一系列特徵表示（模型用來理解輸入序列的內部結構、語法和語義信息的方式之一）。編碼器通常由多個相同結構的層（通常是自注意力層和前饋神經網絡層）堆疊而成。每一層都可以捕捉輸入序列的不同層次的信息，並生成相應的特徵表示。
- **Decoder** : 解碼器會使用編碼器產生的特徵表示還有其他輸入去生成輸出序列，解碼器通常由多個相同結構的層組成，但相對於編碼器，解碼器可能還包括額外的自注意力層，用於關注輸入序列的不同部分。

這兩個部分也可以獨立使用，會取決於 什麼類型的task
- Encoder-only models : 適用於需要理解輸入的任務，如句子分類和命名實體識別。
- Decoder-only models : 適用於生成任務，如文字生成。
- Encoder-decoder models(sequence-to-sequence models) : 適用於需要根據輸入進行產生的任務，如翻譯或摘要。

### Self-Attention 機制

它是 Transformer 模型的最重要的核心組件之一，用於處理序列數據，無論是在編碼器還是解碼器中。它的作用就像是模型的注意力系統，可以動態地為輸入序列中的每個元素分配不同的注意力權重，以捕捉元素之間的關係。這個自注意力機制由三部分組成，分別是查詢（Query）、鍵（Key）、和值（Value），模型通過計算它們之間的關係來生成注意力權重。這個機制讓模型可以同時處理不同元素之間的長程和短程依賴關係。

![https://ithelp.ithome.com.tw/upload/images/20230922/20160436g8qrM6EwwM.jpg](https://ithelp.ithome.com.tw/upload/images/20230922/20160436g8qrM6EwwM.jpg)
_圖片來自:Hung-yi Lee老師(YT)_

> 下方代表一整個 Sequence，假設他輸入有四個 vector，那輸出也會是四個 output vector，但這四個 output vector 不是單一考慮一個小的範圍而是考慮一整個 sequence 才得到的，所以最後就是考慮一整個 Sequence 的資訊然後決定要輸出怎麼樣的 output(所以中間才會連那麼多線)

### 參考資料
- <https://zhuanlan.zhihu.com/p/526155983>
- <https://www.youtube.com/watch?v=hYdO9CscNes>
- <https://huggingface.co/learn/nlp-course/en/chapter1/4>