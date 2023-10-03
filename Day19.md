# Day 19 - 標記資料工具介紹

#### 我們在講載入數據集也就是講 datasets 庫相關的內容前我先來分享一個我用來標記資料的工具。

#### 我要介紹的工具是 Doccano，它是一個開源的文本標註工具和平台，用於自然語言處理和機器學習項目中的文本數據標註，然後它支援情感分析，命名實體識別，文本摘要等任務。

#### 我這邊的安裝是透過 Docker 的方式去安裝的，如果不知道甚麼是 Docker 或是還沒安裝 Docker 是甚麼的可以去參考一下[這個網站](https://blog.kennycoder.io/2020/01/12/Docker-%E4%BB%8B%E7%B4%B9%E5%8F%8A%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8/#:~:text=%E5%B0%B1%E5%AE%89%E8%A3%9D%E5%A5%BD%E4%BA%86%EF%BD%9E-,Windows10%20%E5%AE%89%E8%A3%9D,-%E9%A6%96%E5%85%88%20Windows%20%E5%AE%89%E8%A3%9D)

### 1. 安裝 Doccano
打開終端可以根據以下的 Container 建立程式碼
```shell
docker pull doccano/doccano
docker container create --name doccano \
  -e "ADMIN_USERNAME=admin" \
  -e "ADMIN_EMAIL=admin@example.com" \
  -e "ADMIN_PASSWORD=password" \
  -v doccano-db:/data \
  -p 8000:8000 doccano/doccano
```

### 2. 啟動 Doccano
```shell
docker start doccano
```
接著在網址上輸出 http://localhost:8000/ 就可以開啟了

### 3. 完整使用 Doccano 進行標記

![](C:\Users\User\Pictures\doccano_1.png)
- 打開後會長這樣，接著點 LOGIN

![](C:\Users\User\Pictures\doccano_2.png)
- 然後我們將剛剛上面打上的`ADMIN_USERNAME`和`ADMIN_PASSWORD`就可以登入了

![](C:\Users\User\Pictures\doccano_3.png)
- 點 Create

![](C:\Users\User\Pictures\doccano_4.png)
- 這裡就會出現許多 NLP 相關的任務，那因為我們要創建 NER 的標記資料所以我們選擇 `Sequence Labeling`

![](C:\Users\User\Pictures\doccano_5.png)
- 打上 project name、description、Tags`(這三個可以打上自己需要的我這邊就隨便舉個例子)`，接著就可以按 Create 建立一個新的 NER 標註項目，


![](C:\Users\User\Pictures\doccano_6.png)
- 最左側是一系列功能頁面。Home 這邊是doccano提供的一系列教程，其他的頁面可以設定項目



![](C:\Users\User\Pictures\doccano_7.png)
- 我們直接點 Dataset 然後選 Import Dataset，來看看可以載入哪幾種格式的文本
    - Textfile：Textfile 格式是最簡單的文本格式，每個文本文件通常包含一個文本段落或文檔，文本段落之間用換行符或其他分隔符分隔。
    - Textline：Textline 格式將每個文本片段或文檔放在單獨的行上，每行代表一個文本單位。
    - JSONL：JSONL 格式使用 JSON 來組織文本數據，每行都是一個獨立的 JSON 對象，每個對象通常包含文本和標籤等信息。
    - CoNLL：CoNLL 格式通常用於命名實體識別和詞性標註等任務。它使用列來組織文本數據，每列包含不同的信息，例如詞彙、詞性標籤、命名實體標籤等。

![](C:\Users\User\Pictures\doccano_8.png)
- 這裡我選擇使用 JSONL 的格式，下面它會給你範例讓你知道最後輸出格式結果大概會長什麼樣子
- 然後最下面選擇你要上傳被標記的資料

![](C:\Users\User\Pictures\doccano_9.png)
- 接下來我們可以點選左側欄位的 Labels 為我們的標籤加上顏色的識別這樣在進行標記的時候比較清楚

![](C:\Users\User\Pictures\doccano_10.png)
- 接著我們就可以回到我們的標記資料對他們進行標記，標記完後記得點選勾勾或按 enter，表示標記完成

### 4. 匯出標記完的資料

![](C:\Users\User\Pictures\doccano_11.png)
- 最後當我們把所有資料都標記完的話，我們就可以回到 Dataset 點選 Export Dataset 來匯出我們的檔案 (如果有資料沒標記完但是想先匯出要勾選 `Export only approved documents`)


#### 這樣就是完整的人工標記資料啦



