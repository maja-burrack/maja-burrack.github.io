---
layout: post
title: Precision-Recall Curve for Imbalanced Datasets
---

## ROC-curve vs PR-curve
For a binary classification model with positive class $1$ and negative class $0$, and predictions (or probabilities) in the interval $(0, 1)$, the ROC curve is the true positive rate (TPR) plotted against the false positive rate (FPR) at all classification thresholds.

<details open>
  <summary><b>Definitions of TPR and FPR</b></summary>
  
  The TPR is the ratio between the number of positives correctly classified as positive and the total number of positives.
  
  The FPR is the ratio between the number of positives incorrectly classified as negative and the total number of negatives.
  
  As formulas, we have
  
  $$\textrm{TPR} = \frac{\textrm{TP}}{\textrm{TP} + \textrm{FN}} $$
  
  and
  
  $$ \textrm{FNR} = \frac{\textrm{FP}}{\textrm{FP} + \textrm{TN}}$$
</details>