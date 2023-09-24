# Day 2 - NLP (自然語言處理)是什麼？

讓機器、電腦擁有理解人類說話的語言的能力，就是自然語言處理，它能夠以自然語言文字或語音來查詢資料。這也稱為「語言輸入」。
以一個簡單的例子讓大家可以比較理解，例如 Amazon’s 的 Alexa 或是 Apple’s 的 Siri ，當我們問它問題時不僅能了解我們的要求還能用自然語言回覆我們，NLP 就是他們的核心技術。  

NLP 可套用在書面文字和語音，並適用於所有人類語言。NLP 提供的工具其他範例包括 Web 搜尋、電子郵件垃圾郵件篩選、文字或語音的自動翻譯、文件彙總、情感分析，以及文法 / 拼字檢查。
(Google 使用 NLP 來改進其搜索引擎結果，而 Facebook 等社交網絡則使用它來檢測和過濾一些不好的言論)

<br>

### 為什麼自然語言處理很重要
NLP 是日常生活中不可或缺的一部分，並且隨著語言技術應用於各種產業
- 醫療照護業：醫療照護系統全都移轉到電子病例，NLP 可用於分析及取得健康記錄的新見解。  


- 金融業：貿易商會使用 NLP 技術，從公司文件和新聞發行版本自動挖掘資訊，擷取與投資組合和交易決策相關的資訊。


- 還有像是Google 使用 NLP 來改進其搜索引擎結果，而 Facebook 等社交網絡則使用它來檢測和過濾不好的言論。

<br>

### 電腦如何學會語言
自然語言處理透過這兩個步驟，將複雜的語言轉化為電腦容易處理、計算的形式。早期是人工訂定規則，現在則是讓機器自己學習(機器學習)。
> 第一步是斷詞、理解詞；第二步則是分析句子，包含語法及語義的自動解析

<img src="https://aiacademy.tw/wp-content/uploads/2020/01/ma-natural-language-processing-03.jpg" width="80%">

<br>

### NLP 相關任務
可以將NLP主要分成四大分類
1. 序列標註任務 : 詞性標註任務、命名實體標註任務`(這次主要的介紹重點)`
2. 分類任務 : 文本分類任務、情感分類任務
3. 句子關係判斷 : 句法分析任務
4. 生成式任務 : 翻譯任務、文本摘要任務

<br>

### NLP 程式庫與開發環境
*(以下是一些熱門 NLP 程式庫的範例。)*

**TensorFlow 與 PyTorch**：這是兩個最受歡迎的深度學習工具程式。他們可以自由用於研究和商業用途。他們支援多種語言，主要語言是 Python。

**AllenNLP**：這是 PyTorch 和 Python 中實作之高階 NLP 元件 (例如簡單聊天機器人) 的程式庫。文件極佳。

**HuggingFace**(後面會著重說明)：此公司在 TensorFlow 和 PyTorch 中發布數百種不同的預先訓練深層學習 NLP 模型，以及 TensorFlow 和 PyTorch 中的外掛程式軟體工具程式，讓開發人員能夠快速評估不同預先訓練模型在特定任務上執行的效能。

**Spark NLP**：Spark NLP 是一個適用於 Python、Java 和 Scala 程式設計語言之進階 NLP 的開源文字處理庫。其目標是為自然語言處理流程管道提供應用程式設計介面 (API)。它提供預先訓練的神經網路模式、流程管道和內嵌項目，同時支援訓練客製化模型。

**SpaCy NLP**： SpaCy 是免費的開放原始碼 Python 進階 NLP 資料庫，它是專門用來幫助建立能夠處理和理解大量文本的應用程式。SpaCy 因高度直覺易用而廣為人知，可以處理常見 NLP 專案中所需的許多任務。

<br>

### 參考資料
- <https://aiacademy.tw/what-is-nlp-natural-language-processing/>
- <https://www.deeplearning.ai/resources/natural-language-processing/>
- <https://www.oracle.com/tw/artificial-intelligence/what-is-natural-language-processing/>
- <https://zhuanlan.zhihu.com/p/109122090>
