# 開源機器學習社群平台-Hugging Face

### Hugging Face 是甚麼

Hugging Face 是一家軟體公司和開源社區，專注於自然語言處理和人工智慧領域的研究和開發。該社區成立於2016年，以其對NLP領域的貢獻而聞名，特別是在預訓練NLP模型、數據集、工具和資源方面。他們提供的 Hugging Face Hub 平台現在已經有超過300,000個訓練模型、60,000個數據集 。 

它已經是目前業界做自然語言處理應用的最主流工具了。它主要的支援三種深度學習框架，分別是 PyTorch、TensorFlow 和 JAX，並可以在它們之間輕鬆切換 。

<br>

### 關於 Hugging Face 的主要幾個重點

### 1. Hugging Face Hub(開源社區)  

社區成員可以分享程式碼、模型、數據和知識，共同協作解決NLP相關的問題。這個社區推動了NLP研究和應用的發展，促進了知識和經驗的交流與分享。

_(上面的欄位)_
1. **Models** : 它允許瀏覽、搜索或是獲取各種 NLP 模型，這些模型包括預訓練、微調後的模型以及其他 NLP 相關模型。
2. **Datasets** : 用於管理和分享 NLP 相關的數據集，這些數據可用於模型訓練和評估
3. **Spaces** : 允許用戶創建自己的工作區或空間。每個空間都是一個獨立的項目環境，類似於一個版本控制的儲存庫，用戶可以在其中組織和管理模型、權重、程式碼、文件等資源。
4. **Docs** : 是 Hugging Face Hub 上的文件系統，用於編寫和分享與模型、工具、庫和技術相關的文件和教程。  


![](https://magicpandaengineer.blob.core.windows.net/img/2022ironman/2022ironman-day01-01.png)
### 2. Transformers 庫

參考 [Transformers Github](https://github.com/huggingface/transformers)

Huggingface Transformers 是用於建構、訓練和使用 NLP 模型的 Python 庫，它基於 Transformer 架構。主要用途包括加載和使用預訓練的NLP模型（如 BERT、GPT-2、RoBERTa 等），進行適應特定任務，執行文字處理和生成，以及評估模型表現等。`(下一章會仔細說明Transformer)`

Transformers 支援在 PyTorch, TensorFlow 和 JAX 框架上的互通性，開發者可以根據自己的偏好選擇使用哪個框架。

舉一個簡單的例子，透過這個函式庫我想調用預訓練模型，可以方便地對模型進行調用，只需要一個模型的名字，就可以取得模型檔
```python
from transformers import AutoModel

model = AutoModel.from_pretrained(model_name)
# model_name 可以選擇剛剛提到的Hugging Face Hub提供的模型，ex: bert-base-uncased
```

### 3. Datasets 庫

參考 [Datasets Github](https://github.com/huggingface/datasets)

這個函式庫提供了大量用於NLP研究和開發的資料集。這些資料集可用於模型訓練、評估和基準測試，涵蓋了多種語言和NLP任務。

使用者可以輕鬆載入資料集，無需編寫複雜的資料載入和剩餘程式碼。
資料載入非常簡單，而且資料集可以以不同的資料格式（如PyTorch、TensorFlow、Pandas等）載入。

舉一個簡單的例子，去加載一個現有的數據集，可以方便的使用，只需要數據集名稱，就可以取得數據
```python
from datasets import load_dataset

datasets = load_dataset(dataset_name)
# dataset_name 一樣可以選擇Hugging Face Hub提供的, ex: conll2003

# 轉換為 Pandas 數據格式
import pandas as pd

pandas_dataframe = pd.DataFrame(dataset["train"])
```

### 4. Tokenizers 庫

用於在自然語言處理任務中進行文字分詞和標記化，主要用途是將文字資料分割成標記（tokens），並進行適當的編碼，用於將文字資料轉換生成模型有效的輸入形式以便輸入到NLP模型中。

用範例來說明分詞跟有效的輸入型式  
```python
文句: “今天天氣真好，陽光明媚，適合外出散步。”

第一步先分詞 -> 我們需要將文本分割成標記或詞彙。在這個例子中，可能會被分散以下標記
["今天", "天气", "真好", "，", "阳光", "明媚", "，", "适合", "出去", "散步", "。"]

第二步編碼 -> 們需要將每個標記編碼成數字，以便模型能夠理解。這可以使用詞彙表來完成
[101, 1234, 5678, 321, 9876, 5432, 321, 2345, 6789, 98765, 1357]
```

舉一個簡單的例子
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(tokenize_name)

# 進行分詞
text='今天天氣真好，陽光明媚，適合外出散步。'
tokens = tokenizer.tokenize(example)
```

### 參考資料
- <https://www.zhihu.com/tardis/zm/art/390814850?source_id=1003#:~:text=Transformer%20transformers%20%E6%8C%87%E7%9A%84%E6%98%AF,%E6%84%9F%E7%9F%A5%E7%9A%84%E7%9A%84contextual%20embedding%E3%80%82>
- <https://zhuanlan.zhihu.com/p/535100411>
- <https://ithelp.ithome.com.tw/articles/10291757>
