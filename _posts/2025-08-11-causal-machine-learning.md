---
layout: post
title: Causal Machine Learning
subtitle:
---

I learned young that I could affect the world around me with my actions. If I threw my spoon on the floor as a toddler, an adult would hand it back to me. In adulthood, I find people to be preoccupied with perceiving their influence on the world. Corporate meetings are filled with individuals justifying their initiatives with numbers quantifying the value of their efforts: the marketing manager saying "our new promotion strategies have resulted in increased sales of  EUR 100,000" or the HR manager explaining "completing our training program increases the employee's productivity by 8%". These are causal statements about the effect of some intervention. Making precise statements about causal effects is difficult, but not impossible. Here, I will present different methods for estimating the effect of an intervention.

Formally, we are interested in estimating the Conditional Average Treatment Effect (CATE). This is a measure of how much the outcome changes if we change something (give "treatment") conditional on some other factors. For simplicity, we will assume the treatment is binary, but it doesn't have to be. If we were to sell a product at difference prices, that would be a continuous treatment.

The formal definition of the CATE is:

$$
\tau(x) = E [ Y(1) - Y(0) \mid X = x],
$$

where $Y(1)$ is the outcome with the treatment, and $Y(0)$ is the outcome without the treatment.

Unfortunately, we can't usually observe the CATE directly, because that would require us to assign both treatment and no-treatment to the same unit. Since we can't observe it, we need methods for estimating it. 

There are several frameworks for estimating the CATE. Here, I will present three--in order of the simplest to the most complex. Common to all of them is that they employ machine learning methods to predict the outcome $Y$. The choice of machine learning algorithm is irrelevant when explaining these frameworks, but tree-based methods like random forests or boosting trees are popular. The frameworks differ in how they use the fitted model to estimate the CATE. The more complex frameworks will fit several models instead of one--some of them fitting other variables than just the outcome. 

The choice of machine learning algorithm is commonly called the base-learner while the framework for estimating the CATE is called a meta-learner.

## The simplest method
The S-learner is the simplest meta-learner. The "S" stands for "single" because we use a single model to estimate the outcome $Y$ from a set of input variables including the treatment variable $T$. By including the treatment variable as a feature, we can use the same model to make two prediction--one assuming treatment and one assuming no treatment. Then the CATE is just the difference between the two. 

In mathematical terms, we train one model (a base learner) that predicts the outcome $Y$ based on a set of features $X$ and a treatment variable $T$:

$$
\mu(x, t) = E [ Y | T=t, X=x ].
$$

Just to be precise, we denote the model as $\hat{\mu}$. The model $\hat{\mu}$ is an estimator for the true expected outcome $\mu$.
Then we define the CATE estimate as:

$$
\hat{\tau}_S(x) = \hat{\mu}(x, 1) - \hat{\mu}(x, 0).
$$

That's all there is to the S-learner. Just fit one model on the outcome and use it to estimate what the outcome would have been under each treatment assignment.

The S-learner is simple, but perhaps too simple in some cases. If the treatment is relatively weak compared to other features in the model, the base learner might treat it as noise, leading to overly conservative CATE estimates.

## The two-model method
The T-learner introduces more complexity than the S-learner by using two models to estimate the CATE (hence, the "T" in T-learner. "T" for "two") instead of only one. 

We obtain two models by splitting the data into treatment group and control group. Then we fit separate models estimating the outcome on each of these datasets:

$$
\begin{aligned}
\mu_0(x) &= E [ Y(0) \mid  X=x], \\
\mu_1(x) &= E [ Y(1) \mid X=x]. 
\end{aligned}
$$

Here, we don't include the treatment as a feature (it's constant for each model, so it wouldn't add any information). The fitted model $\hat{\mu}_0$ estimates the outcome without treatment (control), and the model $\hat{\mu}_1$ estimates the outcome with treatment, so we can define the CATE as:

$$
\hat{\tau}_T(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x).
$$

Compared to the S-learner, the T-learner's strength is that it fits models on the two groups separately, which should make it easier to learn the structure of each. It is less likely that the treatment effect will be shrunk to 0.

However, the T-learner might not perform well if the data is imbalanced. If there are few units in the treatment group and many units in the control group, we must be careful not to overfit the data for the treatment group. But a complex model for the control group and simple model for the treatment group, could lead to unreasonable CATE estimates that don't capture the true treatment effect well. 

## The "X"-shaped method
The X-learner {% cite xlearner %} is more involved than the S- and T-learner. It has several stages and includes a model for the propensity score (the estimated probability that treatment is given). 

Exactly like the T-learner, we start with fitting two separate models--one on the control group and one on the treatment group:

$$
\begin{aligned}
\mu_0(x) &= E [ Y(0) \mid X=x], \\
\mu_1(x) &= E [ Y(1) \mid X=x]. 
\end{aligned}
$$

Next, we use these two models to estimate the individual treatment effects for each group:

$$
\begin{aligned}
&\text{For the control group: } \quad &\hat{\tau}_0^{(i)} &= \hat{\mu}_1(X_i^0) - Y_i^0, \\
&\text{For the treatment group: } \quad &\hat{\tau}_1^{(i)} &= Y_i^1 - \hat{\mu}_0(X_i^1).
\end{aligned}
$$

To clarify, $X_i^0$ and $Y_i^0$ are the features and the observed outcome, respectively, for a unit $i$ in the **control group**, while $X_i^1$ and $Y_i^1$ are the features and observed outcome for a unit $i$ in the **treatment group**. 

Notice how this resembles an X-shape because in the first line, we are using the model trained on the treatment group to estimate the counterfactual $\mu_1(X_i^0)$ for the control group, and vice versa in the second line.

For each group, we are asking "what would the outcome have been if the unit had received the opposite treatment?" and then comparing it to the actual outcome $Y_i$. We call $\hat{\tau}_0^{(i)}$ and $\hat{\tau}_1^{(i)}$ the imputed treatment effects. 

Next, we employ machine learning again using the imputed treatment effects as the target variables. Hence, we obtain two models, $\hat{\tau}_0(x)$ and $\hat{\tau}_1(x)$, where the first one is trained only on the control group, and the second is trained only on the treatment group. 

These models are both estimates of the CATE. Now the issue is how to choose between the two. The X-learner doesn't choose, and instead combines them in a final estimator for the CATE. We do this by taking the weighted average of the two models, where the weight is the propensity score $e(x)$ of the treatment:

$$
e(x) = \mathbb{P}(T = 1 \mid X = x)
$$

We don't know the true propensity score, but we can estimate it with yet another model. Letting $\hat{e}(x)$ denote the model for the propensity score, we can write the final CATE estimator as:

$$
\hat{\tau}_X(x) = \hat{e}(x)\hat{\tau}_0(x) + (1-\hat{e}(x))\hat{\tau}_1(x).
$$

This balances the two group-specific estimates based on how likely a unit is to belong to each group. 

If the data is imbalanced, and we choose a relatively simple model for $\hat{\mu}_1(x)$  to prevent overfitting, the model $\hat{\tau}_0(x)$ will be relatively poor (pay attention to the subscripts). However, since we have more observations in the control group, the propensity score $\hat{e}(x)$ will be small. Thus, $\hat{\tau}_0(x)$ won't matter as much in the final CATE estimate.

## Endnote
All three learners are implemented in the python packages [CausalML](https://causalml.readthedocs.io/en/latest/about.html) (from Uber) and [EconML](https://www.pywhy.org/EconML/index.html) (from Microsoft). 

Other meta-learners exist, some of which are implemented in one (or both) of the packages. If you are interested in learning more about them, the package documentation is a good place to start. 

For a more general introduction to causal inference, I would recommend books by Judea Pearl and the online textbook [Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/landing-page.html). The latter contains a chapter about meta-learners, which describes all three meta-learners mentioned here. 

---

## References
{:.no_toc}
{% bibliography --cited %}