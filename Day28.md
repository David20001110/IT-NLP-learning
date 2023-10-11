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




