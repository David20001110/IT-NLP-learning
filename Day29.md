# Day 29 - NER 模型檔案

#### 這篇我使用我之前上傳到 Hugging Face 的模型檔案來解說，那因為在模型訓練那部份我沒講到要如何上船模型，因此等鐵人賽結束後我會再補充回去。

我們來看看上傳模型之後會有哪些內容

#### (先來說最重要的)
1. files : 模型的檔案
![](C:\Users\User\Pictures\modelhub_2.png)
- README.md：README 文件包含了模型的基本資訊，有關模型的描述性文件。
- config.json：包含了有關模型配置的 JSON 檔案。它描述了模型的架構、超參數和其他配置資訊。
- pytorch_model.bin：這是 PyTorch 模型的二進位權重文件，包含了訓練完成的模型參數。
- special_tokens_map.json：這個 JSON 檔案描述了特殊標記（如[PAD]、[CLS]、[SEP]等）的映射和設定資訊。
- tokenizer.json：此 JSON 檔案包含了有關標記器（tokenizer）的配置信息，用於將文字轉換為模型的輸入格式。
- tokenizer_config.json：這個 JSON 檔案也包含有關標記器（tokenizer）的設定信息，通常與tokenizer.json檔案相關。
- Training_args.bin：二進位訓練文件，包含訓練參數和訓練過程的配置資訊。它可以用於還原模型的設定。
- vocab.txt：此文字檔案包含了模型訓練的詞彙表，包括模型期間遇到的所有詞彙。
#### 但這些全部的檔案內容我們在使用時也會全部一起載入，基本上也不用太深入地去理解

1. Model card
![](C:\Users\User\Pictures\modelhub_1.png)