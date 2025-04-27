---
layout: post
title: Maximum a posteriori
subtitle:
---

In Bayesian inference, we consider parameters to be random variables and data to be fixed. This is in stark contrast to classical (Frequentist) statistics where data is considered random and the parameters fixed. When we perform [Bayesian inference]({{site.baseurl}}/blog/bayesian-inference) , we estimate a probability distribution for each parameter, rather than a single number. However, sometimes we might be interested in a just one number, so how should be define this number? A common choice is the value that maximizes the posterior. This estimator is called the *maximum a posteriori (MAP)*-estimator:

$$
\begin{align}
\hat{\theta}_{\text{MAP}} &:= \underset{\theta}{\arg \max} \left \{ p(\theta \mid y) \right \} \label{1} \\ 
    &=\underset{\theta}{\arg \max} \left \{ \frac{p(y \mid \theta) p(\theta)}{p(y)}\right \} \label{2} \\
    &= \underset{\theta}{\arg \max} \left \{ p(y \mid \theta) p(\theta) \right \} \label{3}
\end{align}
$$

To get from \eqref{1} to \eqref{2}, we have used [Bayes' theorem]({{site.baseurl}}/blog/bayesian-inference), and we arrive at \eqref{3} by noting that $p(y)$ does not depend on $\theta$. 

Since $\log$ is monotonically increasing, we can also maximize the log of the posterior instead. Hence,

$$
\begin{align}
\hat{\theta}_{\text{MAP}} &= \underset{\theta}{\arg \max} \left \{ \log \left[ p(y \mid \theta) p(\theta) \right] \right \} \\
    &= \underset{\theta}{\arg \max } \left \{ \log p(y \mid \theta) + \log p(\theta) \right \}
\end{align}
$$

## Maximum Likelihood Estimation
The MAP-estimator is similar, but different to the (Frequentist) maximum likelihood estimator (MLE):

$$
\begin{align}
\hat{\theta}_{\text{MLE}} &= \underset{\theta}{\arg \max} \left \{ p(y \mid \theta ) \right \}.
\end{align}$$

If we consider a standard linear regression formulation

$$
y_{i} = \beta^{T}\mathbf{x}_{i} + \epsilon_i, \quad i = 1, ..., n,
$$

where $y_{i}$ is the scalar response for the $i$-th observation; $\mathbf{x}_{i}$ is a $1 \times p$ vector of predictor variables; $\beta$ is an $p \times 1$ vector of parameters; and $\epsilon_i$ is the unobserved error of each observation (we will assume $\epsilon_i \sim\mathcal{N}(0, \sigma^2)$), then the likelihood function is

$$
\begin{aligned}
p(y \mid \beta, \sigma^2) 
	&= \prod_{i=1}^{n}\frac{1}{\sqrt{2\pi \sigma^2}} \exp \left[ \frac{-1}{2\sigma^2}(y_i - \beta^T \mathbf{x_i})^2\right] \\
	
    &= (2\pi\sigma^2)^{-n/2} \exp \left[\frac{-1}{2\sigma^2} \sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2 \right].
\end{aligned}
$$

Then the maximum likelihood estimate for $\beta$ is:

$$
\begin{align}
\hat{\beta}_{\text{MLE}} 
    &= \underset{\beta}{\arg \max} \left \{ \log \left[ (2\pi\sigma^2)^{-n/2} \exp \left[\frac{-1}{2\sigma^2} \sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2 \right] \right] \right \} \\

    &= \underset{\beta}{\arg \max} \left \{  \log ((2\pi\sigma^2)^{-n/2}) + \left[\frac{-1}{2\sigma^2} \sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2 \right] 
    \right \} \\

    &= \underset{\beta}{\arg \max} \left \{  -\frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2 \right \} \label{likelihood}\\

    &= \underset{\beta}{\arg \min} \left \{  \sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2 \right \}.

\end{align}
$$

## Maximum a posteriori with normal prior
To obtain an MAP estimate of $\beta$, we need to choose a prior. The choices are many, but if we choose a normal prior for *each* $\beta_j$ with identical variance $\tau^2$

$$
p(\beta) = \prod_{j=1}^{p} \frac{1}{\sqrt{2 \pi \tau^2}} \exp \left[ \frac{-\beta_j^2}{2\tau^2}\right]
$$

then we obtain the following MAP:

$$
\begin{align}
\hat{\beta}_{MAP} 

    &= \underset{\beta}{\arg \max} \left \{ 
            \log \left[ (2\pi\sigma^2)^{-n/2} \exp \left[\frac{-1}{2\sigma^2} \sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2 \right] \right] 
            + \log \left[ \prod_{j=1}^{p} \frac{1}{\sqrt{2 \pi \tau^2}} \exp \left[ \frac{-\beta_j^2}{2\tau^2}\right]\right]
        \right \} \\

    &= \underset{\beta}{\arg \max} \left \{ 
        -\frac{1}{2\sigma^2}\sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2  + \log \left[ \prod_{j=1}^{p} \frac{1}{\sqrt{2 \pi \tau^2}} \exp \left[ \frac{-\beta_j^2}{2\tau^2}\right]\right] \right \} \\
    
    &= \underset{\beta}{\arg \max} \left \{ 
            -\frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2 + \log \left[  (2\pi \tau^2)^{-p/2} \exp \left[ \frac{-1}{2\tau^2}\sum_{j=1}^p \beta_j^2 \right] \right]
        \right \}\\

    &= \underset{\beta}{\arg \max} \left \{ 
            -\frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2 +
                \log\left[(2 \pi \tau^2)^{-p/2}\right] -  \frac{1}{2\tau^2} \sum_{j=1}^p \beta_{j}^2
        \right \} \\

    &= \underset{\beta}{\arg \max} \left \{ 
            -\frac{1}{2\sigma^2} \left[  \sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2
                +  \frac{\sigma^2}{\tau^2} \sum_{j=1}^p \beta_{j}^2
            \right] 
        \right \} \\

    &= \underset{\beta}{\arg \min} \left \{ 
             \sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2
                +  \lambda \sum_{j=1}^p \beta_{j}^2
        \right \} \\
        
    &= \underset{\beta}{\arg \min} \left \{ 
             \sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2
                +  \lambda \lVert \beta \lVert_2^2
        \right \},
\end{align}
$$

where $\lambda = \frac{\sigma^2}{\tau^2}$ and $\lVert \cdot \lVert_2$ denotes the L2-norm. We recognize this a L2-regularization (also called Ridge regression in this context), which penalizes very large coefficients $\beta_j$. 

## Maximum a posteriori with Laplace prior
Say we choose a Laplace prior for each $\beta_j$ with same location $\mu=0$ and scale $b$ for each, then

$$
\begin{align}
p(\beta) &= \prod_{j=1}^p \frac{1}{2b} \exp \left[ -\frac{| \beta_j - \mu |}{b} \right] \\
    &= \prod_{j=1}^p \frac{1}{2b} \exp \left[ -\frac{| \beta_j|}{b} \right],
\end{align}
$$

and the MAP for $\beta$ is

$$
\begin{align}
\hat{\beta}_{\text{MAP}} 
    &= \underset{\beta}{\arg \max} \left \{ 
            \log \left[ (2\pi\sigma^2)^{-n/2} \exp \left[\frac{-1}{2\sigma^2} \sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2 \right] \right] 
            + \log \prod_{j=1}^p \frac{1}{2b} \exp \left[ -\frac{| \beta_j |}{b} \right]
        \right \} \\
    
    &= \underset{\beta}{\arg \max} \left \{ 
        -\frac{1}{2\sigma^2}\sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2 
        + \log \frac{1}{(2b)^p} - \frac{1}{b}\sum_{j=1}^p |\beta_j |
        \right \} \\

    &= \underset{\beta}{\arg \max} \left \{ 
        -\frac{1}{2\sigma^2}\sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2 
        - \frac{1}{b}\sum_{j=1}^p |\beta_j |
        \right \} \\   

    &= \underset{\beta}{\arg \max} \left \{ 
        -\frac{1}{2\sigma^2}\left[ \sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2 
        + \frac{2\sigma^2}{b}\sum_{j=1}^p |\beta_j  |
        \right]
        \right \} \\ 

    &= \underset{\beta}{\arg \min} \left \{ 
        \sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2 
        + \frac{2\sigma^2}{b}\sum_{j=1}^p |\beta_j |
        \right \} \\ 

    &= \underset{\beta}{\arg \min} \left \{ 
        \sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2 
        + \lambda \lVert \beta \lVert_1
        \right \}, \\ 

\end{align}
$$

where $\lambda=\frac{2\sigma^2}{b}$ and $\lVert \cdot \lVert_1$ denotes the L1-norm. As above, we recognize this as a specific type of regularization, namely L1-regularization (also called LASSO regression in this context), which promotes sparsity in $\beta$.

## Conclusion
