---
layout: post
title: ROC Curve vs Precision-Recall Curve for Imbalanced Datasets
subtitle: I explain the difference between ROC and precision-recall curves, highlighting their use in evaluating classifiers with imbalanced data.
---

Area under receiver-operating characteristic (ROC) curve (`auc`) and area under precision-recall (PR) curve (`aucpr`) are popular metrics for evaluating the performance of a binary classification model.

For such a classification model with positive class $1$ and negative class $0$, and predictions (or probabilities) in the interval $(0, 1)$:
- the ROC curve is the true positive rate (TPR) plotted against the false positive rate (FPR) at all classification thresholds, and
- the PR-curve is the precision plotted against the recall for all classification thresholds.

<details>
  <summary><b>Definitions of TPR and FPR</b></summary>
  
  The $TPR$ is the ratio between the number of positives correctly classified as positive and the total number of positives.
  <br/><br/>
  The $FPR$ is the ratio between the number of positives incorrectly classified as negative and the total number of negatives.<br>
  <br/>
  As formulas, we have
  $$TPR = \frac{TP}{TP + FN} $$
  $$FNR = \frac{FP}{FP + TN}$$
</details>

<details>
	<summary><b>Definitions of precision and recall</b></summary>
	$$ \textrm{precision} = \frac{TP}{TP + FP}$$
	$$ \textrm{recall} = \frac{TP}{TP + FN}$$
</details>
<br>
`aucpr` is better suited for imbalanced data, because it focuses more on the positive class than `auc` does. For severely imbalanced data (letting the positive class be the minority class, as is standard), the FPR will remain low even if many false positives are present, as the large number of true negatives will dilute the impact of false positives. As a result, a classifier might yield a high `auc` even when it poorly identifies the positive class. 

`aucpr`, on the other hand, is more sensitive to the performance on the positive class as the number of true negatives doesn't directly influence the result.

## Balanced Data
For a balanced dataset with half of the data belonging to each class, a genuinely random classifier will predict positive and negative classes with equal probability. 

Thus, the ROC curve will look like a straight diagonal line from points (0,0) to (1,1). The PR curve with look like a straight horizontal line through (0, 0.5) and (1, 0.5). 

For such curves, the areas underneath them will be 0.5 for each. Hence, the area the closer to 1, the better. If the area is less than 0.5, the model is decidedly bad. It performs worse than random guesses!

## Imbalanced Data
A no-skill classifier on imbalanced data will still have an area under the ROC curve of 0.5, but the area under the PR curve will be much less. If the positive class is the minority class and has probability $p$ of occurring, then the `aucpr` will be $p$.

If we take a simple no-skill classifier that predicts $p$ everywhere, then for threshold $t \leq p$, the classifier will predict the positive class for all instances. In that case, $TP = P$ and $FP = N$. For thresholds strictly greater than $p$, the classifier will predict the negative class for all instances.

So, for threshold $t \leq p$:

$$\textrm{precision} = \frac{TP}{TP + FP} = \frac{P}{P + N} = p$$

$$\textrm{recall} = \frac{TP}{TP + FN} = \frac{P}{P + 0}=1$$

And for threshold $t > p$:

$$ \textrm{precision} = \frac{TP}{TP + FP} = \frac{0}{0 + 0} = \textrm{undefined (or considered }0)$$

$$ \textrm{recall} = \frac{TP}{TP + FN} = \frac{0}{0 + P} = 0$$

The PR curve can be considered as horizontal line at precision $=p$ from recall $0$ to $1$ (when the recall is 1, the precision can be considered to drop to). The area under a horizontal line at $p$ from 0 to 1, is just $p$. So a "good" model on imbalanced data might have `aucpr` less than 0.5, but it should at least be greater than $p$. 