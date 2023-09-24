# 相關開發環境與設定

這裡介紹的開發環境都是我自己有試過的。  
由於是要訓練自然語言處理的模型，這會蠻耗時的，因此如果電腦有GPU是最好的，如果沒有的話沒關係大家可以使用我等等介紹的雲端環境


## 本地端開發
### 1. PyCharm

PyCharm 是著名的開發環境 ( IDE )，專門用於Python，但也支持其他編程語言，例如javascript，kotlin或CoffeeScript 和其他工具，例如html或css。  

它屬於跨平台的，具有Windows，MacOS和Linux版本，它有兩種版本，必須付費的高級版本和另一個免費增值版或社區版本是免費的。

我之所以使用 PyCharm 的原因是因為它內建有強大的調適工具、支援多種 Python 框架和函式庫，其功能和性能使得編寫、調試和管理 Python 程式碼變得更加方便。特別適用於 Python 開發人員和團隊，幫助提高生產力、減少錯誤，並加速開發過程，而且付費版的功能真的對於開發強大許多，但基本的免費版就蠻夠用了。  
_( 相關安裝可以到[官網](https://www.jetbrains.com/pycharm/download/?section=windows)查詢 )_

### 2. Jupyter Notebook

這個是我給新手比較推薦的，可以前往 [Anaconda 網站](https://www.anaconda.com/download)  
Jupyter Notebook 是一種開源的互動式計算環境，相當彈性，特別適用於數據科學、機器學習和科學計算等領域，可以自行安裝套件、模塊化套件。

<img src="https://miro.medium.com/v2/resize:fit:1200/1*R5uM8zw8uhW4-HC4F1v9IA.png">

## 雲端開發

### Google Colab 
_(我之後我以這個雲端環境來做開發測試，所以會簡單介紹操作)_  
Google Colab 是一個免費的雲端開發環境，專為Python程式語言和機器學習（Machine Learning）領域設計。它為用戶提供了方便、彈性和可協作的工具，讓我們可以輕鬆開展數據科學和機器學習專案，重點是有免費的GPU可以使用，但當然不是吃到飽的哈哈，它會給予免費版的忘記是12GB還是16GB的量。

1. 建立 Colab 筆記本  
   按新增->更多，點擊 Google Colaboratory 應用程式
2. 打開後可以先更改檔名
3. 在程式碼儲存格中輸入程式碼語句
4. 點左側的箭頭去執行程式碼
5. 這裡是可以選擇使用 GPU 提升效能  
   可以點選下圖中RAM及磁碟使用狀態旁的黑色 icon後，點選下方查看資源，即可知道目前資源使用情況，點選變更執行階段類型可以選擇GPU

### 參考資料
- <https://ithelp.ithome.com.tw/m/articles/10292946>
- <https://ithelp.ithome.com.tw/articles/10291739>
- <https://www.linuxadictos.com/zh-TW/pycharm-un-potente-ide-para-crear-programas-con-python.html>
- <https://medium.com/python4u/google-colab-%E6%95%99%E5%AD%B8-2-%E5%BB%BA%E7%AB%8B%E5%8F%8A%E4%BD%BF%E7%94%A8-colab-%E7%AD%86%E8%A8%98%E6%9C%AC%E7%B7%A8%E5%AF%AB-python-8b0d1a5b920d>


