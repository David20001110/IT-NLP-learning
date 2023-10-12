# Day 28 - NER 模型評估和驗證

#### 昨天我們已經透過`train()`方法將模型訓練完後，我們需要了解它在未見過的資料上的表現。使用驗證集進行評估可以幫助您確定模型的泛化能力，即模型是否能夠在新數據上表現良好。

```python
trainer.evaluate()
```
- `evaluate()`方法的用意是在訓練過程中對模型進行評估，以了解模型在驗證集上的表現和表現。
- 這裡的驗證資料是我們在建立物件`Trainer`時設定的`eval_dataset`參數來進行評估

```python
{'eval_loss': 0.03453173115849495,
 'eval_precision': 0.9602925809822361,
 'eval_recall': 0.9704329461457233,
 'eval_f1': 0.9653361344537815,
 'eval_accuracy': 0.9932748124026618,
 'eval_runtime': 4.4921,
 'eval_samples_per_second': 106.409,
 'eval_steps_per_second': 13.357}
```
#### 在說明上面輸出結果中比較重要的幾個重點之前我們先需要來解釋一個名詞 `混淆矩陣`

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*Z54JgbS4DUwWSknhDCvNTQ.png)  
_以上圖來自 [Sarang Narkhede](https://medium.com/@narkhedesarang)_

**混淆矩陣（Confusion Matrix）** : 是在分類問題中用於評估模型性能的一種表格，它以矩陣的形式展示了模型對不同類別樣本的分類結果。混亂矩陣通常用於二分類和多分類任務，以幫助分析模型的表現和錯誤分類情況。

四格分別代表
1. TP (True Positives)：模型正確識別的正類別樣本的數量。  
    (舉例: 預測小帥為陽性，他真的為陽性)

2. TN (True Negatives)：模型正確辨識的負類別樣本的數量。  
   (舉例: 預測大壯不是陽性，他真的不是陽性)

3. FP (False Positives)：模型將實際負類別樣本錯誤分類為正類別的數量。  
   (舉例: 預測大美是陽性，結果他不是陽性)

4. FN (False Negatives)：模型將實際正類別樣本錯誤分類為負類別的數量。  
   (舉例: 預測小美不是陽性，結果她是陽性)

那接下來要來說與混淆矩陣有關的四個分數
1. Accuracy (準確率) : 模型預測正確數量所佔整體的比例。  
   ![](C:\Users\User\Pictures\output_1.png)

2. Precision (精確率) : 被預測為 Positive 的資料中，有多少是真的 Positive。  
   ![](C:\Users\User\Pictures\output_2.png)

3. Recall (召回率) : 原本 Positive 的資料中被預測出來的有多少。  
   ![](C:\Users\User\Pictures\output_3.png)

4. F1-score (F1值) : Precision 與 Recall 的調和平均數。(用於平衡準確度和召回率之間的權衡)  
   ![](C:\Users\User\Pictures\output_4.png)
    
#### **eval_loss (驗證損失)** : 是除了四個分數之外很重要的評估指標，損失值是一個數值，通常是非負數，在訓練過程中，模型的目標是最小化損失值，即使模型的預測結果也接近實際標籤。因此，較低的驗證損失通常表示模型對驗證資料的預測準確比較。

我們昨天在定義一個計算評估指標的函數這裡就可以使用到了
```python
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
```
我們來使用這個函數從模型的預測結果和實際標籤中計算出效能指標，以便更全面地了解模型在測試資料上的表現表現。
```python
predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
results
```
- ```python
  predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
  predictions = np.argmax(predictions, axis=2)
  ```
  - 我們使用`predict`的方法來將產生模型預測的結果: `predictions`和 原本標記資料的答案: `labels`
  - 將`predictions`經過`argmax`函數回傳每一個標記類別的索引，也就是回傳完整的預測結果
- ```python
   true_predictions = [
       [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
       for prediction, label in zip(predictions, labels)
   ]
   true_labels = [
       [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
       for prediction, label in zip(predictions, labels)
   ]
  ```
  - 將原本標記答案以及預測結果去除特殊標記`-100`
- ```python
   results = metric.compute(predictions=true_predictions, references=true_labels)
   ```
  - 最後利用`compute_metrics`計算各個標籤的性能指標`(這裡我拿我之前做過的評估來做示範)`   
    ![](C:\Users\User\Pictures\output_5.png)
    - 這裡除了會輸出`overall`的分數之外，也會把我們的每一個標籤的 `precision`、`recall`、`f1值` 計算出來以及每一個標籤的總數有多少，這樣可以更讓我們清楚知道自己哪一個標籤的表現比較好或是比以較差，以讓我們可以反覆地去調整。

### 參考資料
- <https://hackmd.io/@CynthiaChuang/Common-Evaluation-MetricAccuracy-Precision-Recall-F1-ROCAUC-and-PRAUC>
- <https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62>
