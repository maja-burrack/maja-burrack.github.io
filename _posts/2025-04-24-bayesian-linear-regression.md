---
layout: post
title: Bayesian Linear Regression
subtitle: I provide a complete Bayesian analysis for a standard linear regression model with a Normal-inverse-gamma prior, demonstrating how the conjugate structure yields closed-form expressions for both the posterior distribution and the posterior predictive.
---

Consider a standard linear regression formulation with

$$
y_{i} = \beta^{T}\mathbf{x}_{i} + \epsilon_i, \quad i = 1, ..., n
$$

where $y_{i}$ is the scalar response for the $i$-th observation $\mathbf{x}_{i}$ is a $1 \times p$ vector of predictor variables; $\beta$ is an $p \times 1$ vector of parameters; and $\epsilon_i$ is the unobserved error of each observation. We will assume $\epsilon_i \sim\mathcal{N}(0, \sigma^2)$. It follows that $y_i \sim \mathcal{N}(\beta^T \mathbf{x}_i, \sigma^2)$. 

Recall, we perform [Bayesian Inference]({{site.baseurl}}/blog/bayesian-inference) by specifying a prior $p(\beta, \sigma^2)$ on the parameters and a likelihood $p(y \mid \beta, \sigma^2)$, and then deriving (or approximating) the posterior $p(\beta, \sigma^2 \mid y)$ using Bayes' Theorem:

$$
p(\beta, \sigma^2 \mid y) = \frac{
	\overbrace{p(y \mid \beta, \sigma^2)}^{\textrm{likelihood}}
	\overbrace{p(\beta, \sigma^2)}^{\textrm{prior}}
}{
	\underbrace{\int\int p(y \mid \beta, \sigma^2)p(\beta, \sigma^2) d\beta d\sigma^2}_{\textrm{evidence}}
}.
$$

## Posterior
Since $y_i \sim \mathcal{N}(\beta^T \mathbf{x}_i, \sigma^2)$, the likelihood is

$$
\begin{align*}
p(y \mid \beta, \sigma^2) 
	&= \prod_{i=1}^{n}\frac{1}{\sqrt{2\pi \sigma^2}}\exp \left[ \frac{-1}{2\sigma^2}(y_i - \beta^T \mathbf{x_i})^2\right] \\
			  &= (2\pi\sigma^2)^{-n/2} \exp \left[\frac{-1}{2\sigma^2} \sum_{i=1}^{n} (y_i- \beta^T \mathbf{x_i})^2 \right].

\end{align*}
$$

We can write this in terms of matrices

$$
y = \begin{bmatrix}
	y_1 \\
	\vdots \\
	y_n
\end{bmatrix}, \quad 
X = \begin{bmatrix}
	\mathbf{x_1} \\
	\vdots \\
	\mathbf{x_n}
\end{bmatrix} = 
\begin{bmatrix}
	x_{11} & \dots & x_{1p} \\
	\vdots & \ddots & \vdots \\
	x_{n1} & \dots & x_{np}
\end{bmatrix}, \quad
\beta = \begin{bmatrix}
	\beta_1 \\
	\vdots \\
	\beta_p
\end{bmatrix}
$$

as 

$$
p(y \mid \beta, \sigma^2) = (2\pi\sigma^2)^{-n/2} \exp \left[ \frac{-1}{2\sigma^2} (y-X\beta)^T(y-X\beta)\right].
$$

It remains to choose a prior for $\beta$ and $\sigma^2$. We know that [the Normal-inverse-gamma distribution is the conjugate prior to the Normal likelihood]({{site.baseurl}}/blog/conjugate-prior-normal-inverse-gamma), so that would be a natural choice. So we let

$$
\begin{gathered}
	\beta \sim \mathcal{N}(\mu_0, \sigma^2\Lambda_0^{-1}) \\
	\sigma^2 \sim \textrm{InvGamma}(a_0, b_0), 
\end{gathered}
$$

where $\sigma^2\Lambda_0^{-1}$ is the covariance matrix. The covariance matrix is symmetric and positive-definite by definition, so it follows that $\Lambda_0$ is too. The densities for each of these priors are

$$
\begin{gathered}
	p(\beta \mid \sigma^2) = (2\pi\sigma^2)^{-p/2}| \Lambda_0 |^{1/2} \exp \left[ -\frac{1}{2}(\beta-\mu_0)^T \Lambda_0 (\beta-\mu_0) \right], \quad \textrm{and} \\
	p(\sigma^2) = \frac{b_0^{a_0}}{\Gamma(a_0)}(\sigma^2)^{-(a_0+1)}\exp \left[ \frac{-b_0}{\sigma^2} \right].
\end{gathered}
$$

Now, we want to obtain the posterior $p(\beta, \sigma^2 \mid y)$ given by

$$
\begin{align}
p(\beta, \sigma^2 \mid y) 

	&= \frac{
			p(y \mid \beta, \sigma^2)p(\beta, \sigma^2)
		}{
			\int\int p(y \mid \beta, \sigma^2)p(\beta, \sigma^2) d\beta d\sigma^2
		} \\
	&= \frac{
			p(y \mid \beta, \sigma^2)p(\beta \mid \sigma^2)p(\sigma^2)
			}{
			\int \int p(y \mid \beta, \sigma^2)p(\beta \mid \sigma^2)p(\sigma^2) d\beta d\sigma^2
			}. \label{eq:bayes}
\end{align}
$$

Let's start with the joint probability in the numerator of $\eqref{eq:bayes}$:

$$
\begin{align}
\begin{split}
p(y \mid \beta, \sigma^2)p(\beta \mid \sigma^2)p(\sigma^2) 
	&= (2\pi\sigma^2)^{-n/2} \exp \left[ \frac{-1}{2\sigma^2} (y-X\beta)^T(y-X\beta)\right] \\
	
	&\times (2\pi\sigma^2)^{-p/2}|\Lambda_0|^{1/2} \exp \left[-\frac{1}{2}(\beta-\mu_0)^T\Lambda_0(\beta-\mu_0) \right] \\
	
	&\times \frac{b_0^{a_0}}{\Gamma(a_0)}(\sigma^2)^{-(a_0+1)}\exp \left[ \frac{-b_0}{\sigma^2} \right].
\end{split}
% \label{eq:joint-probability}
\end{align}
$$

Collecting the exponents, this is equal to:

$$
\begin{align}
(2\pi\sigma^2)^{-n/2} (2\pi\sigma^2)^{-p/2}|\Lambda_0|^{1/2} \frac{b_0^{a_0}}{\Gamma(a_0)}(\sigma^2)^{-(a_0+1)} \exp 
	\left[ 
		\frac{-1}{2\sigma^2} (y-X\beta)^T(y-X\beta)
		-\frac{1}{2}(\beta-\mu_0)^T\Lambda_0(\beta-\mu_0) 
		-\frac{b_0}{\sigma^2} 
	\right]
\label{eq:joint-probability}
\end{align}
$$


Using the trick $y-X\beta = y - X\hat{\beta} + X\hat{\beta} - X\beta$, where $\hat{\beta} = (X^T X)^{-1}X^T y$ is the ordinary least squares (OLS) estimator (assuming $X^TX$ is invertible), we can write the following:

$$
\begin{align*}

	(y-X\beta)^T(y-X\beta)
	
		&= (y - X\hat{\beta} + X\hat{\beta} - X\beta)^T(y - X\hat{\beta} + X\hat{\beta} - X\beta)\\
		
		&= (y - X\hat{\beta})^T(y-X\hat{\beta}) + (\hat{\beta}-\beta)^TX^TX(\hat{\beta}-\beta) + \underbrace{2(y-X\hat{\beta})^T(X\hat{\beta}-X\beta)}_{0},
		
\end{align*}
$$

where the last term equals $0$, because

$$
\begin{align*}
    (y-X\hat{\beta})^T(X\hat{\beta}-X\beta) 
        &= (y-X\hat{\beta})^TX(\hat{\beta}-\beta) \\
		&= (y-X((X^T X)^{-1}X^T y))^TX(\hat{\beta}-\beta) \\
		&= (X^Ty - X^TX(X^T X)^{-1}X^T y)^T(\hat{\beta}- \beta) \\
		&= \underbrace{(X^Ty - X^Ty)}_{0}{}^T(\hat{\beta}-\beta) \\
		& = 0.
\end{align*}
$$

Now we can use this result to write

$$
\begin{multline*}
	(y-X\beta)^T(y-X\beta) + (\beta-\mu_0)^T\Lambda_0 (\beta- \mu_0) \\
		= (y - X\hat{\beta})^T(y-X\hat{\beta}) 
			+ (\hat{\beta}-\beta)^TX^TX(\hat{\beta}-\beta) 
			+ (\beta-\mu_0)^T\Lambda_0 (\beta- \mu_0) \\
		= (y - X\hat{\beta})^T(y-X\hat{\beta})
			+\hat{\beta}^TX^TX\hat{\beta}+\beta^TX^TX\beta 
			- 2\hat{\beta}^TX^TX\beta + \beta^T\Lambda_0\beta 
			+ \mu_0^T\Lambda_0\mu_0 -2\mu_0^T\Lambda_0\beta \\
		= (y - X\hat{\beta})^T(y-X\hat{\beta})
			+\underbrace{\beta^T(\underbrace{X^TX+\Lambda_0}_{\Lambda_N})\beta -2(\underbrace{\mu_0^T\Lambda_0+\hat{\beta}^TX^TX}_{b_N^T})\beta
			}_{\beta^T\Lambda_N\beta - 2b_N^T\beta}
			+\hat{\beta}^TX^TX\hat{\beta}+\mu_0^T\Lambda_0\mu_0,
\end{multline*}
$$

and by completing the square in $\beta$, and letting $\mu_N = \Lambda_N^{-1}b_N$, we obtain

$$
\begin{multline*}
	(y-X\beta)^T(y-X\beta) + (\beta-\mu_0)^T\Lambda_0 (\beta- \mu_0) \\
		= (y-X\hat{\beta})^T(y-X\hat{\beta})
			+ {\color{red}\beta^T\Lambda_N\beta - 2b_N^T\beta}
			+ \hat{\beta}^TX^TX\hat{\beta} 
			+ \mu_0^T\Lambda_0\mu_0 \\
		= (y-X\hat{\beta})^T(y-X\hat{\beta})
			+ {\color{red}(\beta -\Lambda_N^{-1}b_N)^T\Lambda_N(\beta-\Lambda_N^{-1}b_N)
			-b_N^T\Lambda_N^{-1}b_N}
			+ \hat{\beta}^TX^TX\hat{\beta} 
			+ \mu_0^T\Lambda_0\mu_0 \\
		= (y-X\hat{\beta})^T(y-X\hat{\beta})
			+ {\color{red}(\beta -\mu_N)^T\Lambda_N(\beta-\mu_N)
			-b_N^T\mu_N}
			+ \hat{\beta}^TX^TX\hat{\beta} 
			+ \mu_0^T\Lambda_0\mu_0 \\
		= {\color{blue}{(y-X\hat{\beta})^T(y-X\hat{\beta})}}
			+ {\color{red}(\beta -\mu_N)^T\Lambda_N(\beta-\mu_N)
			-\mu_N^T\Lambda_N\mu_N}
			+ {\color{blue}{\hat{\beta}^TX^TX\hat{\beta}}}
			+ \mu_0^T\Lambda_0\mu_0.\\
\end{multline*}
$$

Finally, let's have a look at the first and second-to-last terms (colored blue above) using the substitution with the OLS estimator:

$$
\begin{aligned}
 (y-X\hat{\beta})^T(y-X\hat{\beta}) 
			+\hat{\beta}^TX^TX\hat{\beta}
		&= (y-X\hat{\beta})^T(y-X\hat{\beta}) 
			+ \hat{\beta}^TX^TX(X^TX)^{-1}X^Ty \\
		&= (y-X\hat{\beta})^T(y-X\hat{\beta})
			+ \hat{\beta}^TX^Ty \\
		&= y^Ty+\hat{\beta}^TX^TX\hat{\beta}-2\hat{\beta}^TX^Ty + \hat{\beta}^TX^Ty\\
		&= y^Ty + \hat{\beta}^TX^Ty - 2\hat{\beta}^TX^Ty + \hat{\beta}^TX^Ty \\
		&= y^Ty
\end{aligned}
$$

Plugging this in, we get

$$
(y-X\beta)^T(y-X\beta) + (\beta-\mu_0)^T\Lambda_0 (\beta- \mu_0) 
	= y^Ty 
		+ (\beta -\mu_N)^T\Lambda_N(\beta-\mu_N)
		-\mu_N^T\Lambda_N\mu_N
		+ \mu_0^T\Lambda_0\mu_0,
$$

where $\Lambda_N = X^TX + \Lambda_0$ and $\mu_N = \Lambda_N^{-1}(\mu_0^T\Lambda_0 + X^Ty)$. 

Now we can go back to \eqref{eq:joint-probability} and rewrite the exponent as

$$
\begin{aligned}
	\left[ -\frac{1}{2\sigma^2}(y-X\beta)^T(y-X\beta) - \frac{1}{2\sigma^2}(\beta-\mu_0)^T\Lambda_0(\beta-\mu_0) - \frac{b_0}{\sigma^2} \right]

	&= -\frac{1}{2\sigma^2}\left[ (y-X\beta)^T(y-X\beta) + (\beta-\mu_0)^T\Lambda_0(\beta-\mu_0)  + 2b_0\right] \\

	&= -\frac{1}{2\sigma^2} \left[
		(\beta -\mu_N)^T\Lambda_N(\beta-\mu_N)
		+ y^Ty 
		-\mu_N^T\Lambda_N\mu_N
		+ \mu_0^T\Lambda_0\mu_0 
		+ 2b_0 
		\right] \\
	&= -\frac{1}{2\sigma^2} \left[
		(\beta -\mu_N)^T\Lambda_N(\beta-\mu_N)
		+ 2b_N
		\right],
\end{aligned}
$$

where 

$$
b_N = b_0 + \tfrac{1}{2}(y^Ty - \mu_N^T\Lambda_N\mu_N + \mu_0^T\Lambda_0\mu_0).
$$

Hence, the joint probability can be written as

$$
\begin{align}
	p(y, \beta, \sigma^2) 
		&= (2\pi\sigma^2)^{-n/2} (2\pi\sigma^2)^{-p/2}|\Lambda_0|^{1/2} 
		\frac{b_0^{a_0}}{\Gamma(a_0)}(\sigma^2)^{-(a_0+1)}
		\exp \left[ 
			-\frac{1}{2\sigma^2}(\beta -\mu_N)^T\Lambda_N(\beta-\mu_N) 
			-\frac{b_N}{\sigma^2} 
		\right],
\end{align}
$$

and by rearranging a bit as:

$$
\begin{align}
\begin{split}
	p(y, \beta, \sigma^2) 
		&= (2\pi\sigma^2)^{-p/2}|\Lambda_0|^{1/2}  \exp \left[ -\frac{1}{2\sigma^2}(\beta -\mu_N)^T\Lambda_N(\beta-\mu_N)  \right] \\
		& \quad \times (\sigma^2)^{-(a_0+n/2+1)}\exp \left[ -\frac{b_N}{\sigma^2}\right] \\
		&\quad \times (2\pi)^{-n/2}\frac{b_0^{a_0}}{\Gamma (a_0)}.
\end{split} \label{eq:joint-lines}
\end{align}
$$


Therefore, by recognizing the kernel of the joint distribution, we can identify the posterior distributions as follows:

$$
p(\beta \mid \sigma^2, y) \sim \mathcal{N}(\mu_N, \sigma^2 \Lambda_N^{-1})
\quad \text{and} \quad
p(\sigma^2 \mid y) \sim \text{InvGamma}(a_N, b_N),
$$

where the updated posterior parameters are:

$$
\begin{aligned}
\Lambda_N &= X^T X + \Lambda_0, \\
\mu_N &= \Lambda_N^{-1}(X^T y + \Lambda_0 \mu_0), \\
a_N &= a_0 + \frac{n}{2}, \\
b_N &= b_0 + \frac{1}{2}(y^T y + \mu_0^T \Lambda_0 \mu_0 - \mu_N^T \Lambda_N \mu_N).
\end{aligned}
$$

## Marginal Likelihood

Let's turn to the marginal likelihood in the denominator of \eqref{eq:joint-probability}:

$$
p(y) = \int \int p(y \mid \beta, \sigma^2)p(\beta \mid \sigma^2)p(\sigma^2)d\beta d\sigma^2.
$$

Because we have managed to write the joint probability in a rather convenient form, evaluating this integral turns out to not be too hard. Let's start with integrating out $\beta$. Only the first line in \eqref{eq:joint-lines} has terms in $\beta$, and it looks very similar to the pdf of a multivariate normal distribution. In fact, if we write 

$$
\Sigma^{-1} = \frac{1}{\sigma^2}\Lambda_N,
$$

then we immediately know that 

$$
\begin{align}
\int \exp \left[ -\frac{1}{2\sigma^2}(\beta -\mu_N)^T\Lambda_N(\beta-\mu_N)  \right] d\beta
&= (2\pi)^{p/2}|\Sigma|^{1/2} \\
&= (2\pi)^{p/2}|\sigma^2\Lambda_N^{-1}|^{1/2} \\
&= (2\pi\sigma^2)^{p/2}|\Lambda_N|^{-1/2}.\label{eq:beta-integral}
\end{align}
$$

This follows from the definition of a multivariate normal distribution with mean $\mu_N$ and covariance matrix $\Sigma^{-1}$. 

Hence, integrating \eqref{eq:joint-lines} over $\beta$ and plugging in \eqref{eq:beta-integral}, we get:

$$
\begin{align}
\int p(y, \beta, \sigma^2) d\beta 
	&= (2\pi\sigma^2)^{-p/2}|\Lambda_0|^{1/2}(2\pi\sigma^2)^{p/2}|\Lambda_N|^{-1/2} 
		\times (\sigma^2)^{-(a_0+n/2+1)}\exp \left[ -\frac{b_N}{\sigma^2}\right] 
		\times (2\pi)^{-n/2}\frac{b_0^{a_0}}{\Gamma (a_0)} \\

	&= |\Lambda_0|^{1/2}
	|\Lambda_N|^{-1/2} 
		\times (\sigma^2)^{-(a_0+n/2+1)}\exp \left[ -\frac{b_N}{\sigma^2}\right] 
		\times (2\pi)^{-n/2}\frac{b_0^{a_0}}{\Gamma (a_0)}. \\
\end{align}
$$

Similarly for integrating out $\sigma^2$, we need only worry about the second line, which looks awfully a lot like the pdf of an inverse gamma distribution. Letting $a_N = a_0 +n/2$, we immediately know that 

$$
\int (\sigma^2)^{-(a_0+n/2+1)}\exp \left[ -\frac{b_N}{\sigma^2}\right] d\sigma^2 

= \frac{\Gamma(a_N)}{b_N^{a_N}}.
$$

So we can finally obtain the evidence:

$$
\int \int p(y, \beta, \sigma^2) d\beta d\sigma^2 = (2\pi)^{-n/2}\sqrt{
	\frac{
		|
		\Lambda_0
		|
		}{
		|
		\Lambda_N
		|
		}}
	\frac{
		b_0^{a_0}
		}{
		b_N^{a_N}
		}
	\frac{
		\Gamma(a_N)
		}{
		\Gamma(a_0)
		}.
$$

Thus, we have shown that the posterior $p(\beta, \sigma^2 \mid y)$ has a closed form solution under our choice of prior. Generally, the evidence does not have closed form and we would usually have to use approximation methods to compute it. 

## Posterior predictive
Let's also compute the posterior predictive $p(y^* \mid y)$ given by

$$
\begin{align}
p(y^* \mid y)  
	&= \int \int p(y^* \mid X^*, \beta, \sigma^2) p(\beta, \sigma^2 \mid y) d\beta d\sigma^2 \\
	
	&= \int \int p(y^* \mid X^*, \beta, \sigma^2) p(\beta \mid \sigma^2, y) p(\sigma^2 \mid y) d\beta d\sigma^2 \\
&= \int \underbrace{\left[ 
		\int p(y^* \mid X^*, \beta, \sigma^2) p(\beta \mid \sigma^2, y) d\beta
	\right]}_{p(y^* \mid \sigma^2, y)}
	p(\sigma^2 \mid y) d\sigma^2.
\end{align}
$$

I have included $X^*$ in the formula to make it clear that the posterior predictive depends on *new data points* $X^*$. The posterior predictive answers the question:

<p style="text-align: center; font-style: italic;">
  Given the data I've seen, and my prior beliefs, what is the distribution over new outputs $y^*$, at a new inputs $X^*$?
</p>

Let's start by computing the inner integral:

$$
\int p(y^* \mid X^*, \beta, \sigma^2) p(\beta \mid \sigma^2, y) d\beta
	= \int \mathcal{N}(y^* \mid X^*\beta, \sigma^2 I) \mathcal{N}(\beta \mid \mu_N, \sigma^2 \Lambda_N^{-1}) d\beta.
$$

We can compute this without directly integrating it by using well-known a theorem[^1]:

>**Theorem**
>Let $z$ follow a multivariate normal distribution
>$$z \sim \mathcal{N}(\mu, \Sigma).$$
>Suppose we can partition $z$ as $z=(z_a, z_b)$, and similarly $\mu=(\mu_{a}, \mu_b)$ and $$\Sigma = \begin{bmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb}\end{bmatrix}.$$ Then
>- the marginal distribution of $z_a$ is $z_a \sim \mathcal{N}(\mu_a, \Sigma_{aa})$,
>- the marginal distribution of $z_b$ is $z_b \sim \mathcal{N}(\mu_b, \Sigma_{bb})$, and
>- the conditional distribution of $z_b$ given $z_a$ is $$z_b \mid z_a \sim \mathcal{N}(\mu_b + \Sigma_{ba}\Sigma_{aa}^{-1}(z_a-\mu_a), \Sigma_{bb}-\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab})$$

We can apply this theorem by constructing $z=(\beta, y^*)$,  with $\mu = (\mu_{N}, X^*\mu_{N})$, and

$$
\Sigma = \begin{bmatrix}
	\Sigma_{\beta\beta} & \Sigma_{\beta y^*} \\
	\Sigma_{y^* \beta} & \Sigma_{y^*y^*}
\end{bmatrix} =
\begin{bmatrix}
	\sigma^2 \Lambda_N^{-1} & \sigma^2\Lambda_N^{-1}X^{*T} \\
	\sigma^2 X^*\Lambda_N^{-1} & \sigma^2(I+X^*\Lambda_N^{-1}X^{*T})
\end{bmatrix}.
$$

Then it follows from the theorem that $\beta \sim \mathcal{N}(\mu_N, \sigma^2\Lambda_N^{-1})$,  $y^* \sim \mathcal{N}(X^*\mu_N, \sigma^2(I+X^*\Lambda_N^{-1}X^{*T}))$, and 

$$
y^* \mid \beta \sim \mathcal{N}(\mu_{y^*\mid \beta}, \Sigma_{y^* \mid \beta}),
$$

where

$$
\begin{aligned}
\mu_{y^* \mid \beta} 
	&= X^*\mu_N + \sigma^2 X^*\Lambda_N^{-1}(\sigma^2\Lambda_N^{-1})^{-1}(\beta-\mu_N), \\
	&= X^*\mu_N + X^*(\beta - \mu_N) \\
	&= X^*\beta, \\ \\

\Sigma_{y^*\mid \beta} 
	&= \sigma^2(I+X^*\Lambda_N^{-1}X^{*T})-\sigma^2X^*\Lambda_N^{-1}(\sigma^2\Lambda_N^{-1})^{-1}\sigma^2\Lambda_N^{-1}X^{*T} \\
	&= \sigma^2 (I+X^*\Lambda_N^{-1}X^{*T})-\sigma^2X^*\Lambda_N^{-1}X^{*T} \\
	&= \sigma^2I.

\end{aligned}
$$

Notice that we have constructed $z$ in such a way that it is immediately apparent that 

$$
\begin{aligned}
p(y^* \mid\sigma^2, y) 
	&= \int p(y^* \mid X^*, \beta, \sigma^2) p(\beta \mid \sigma^2, y) d\beta \\
	&= \int \mathcal{N}(y^* \mid X^*\beta, \sigma^2 I) \mathcal{N}(\beta \mid \mu_N, \sigma^2 \Lambda_N^{-1}) d\beta \\
	&= \mathcal{N}(X^*\mu_N, \sigma^2(I+X^*\Lambda_N^{-1}X^{*T})). 
\end{aligned}
$$

Now computing the outer integral:

$$
\begin{aligned}
	p(y^* \mid y)
		&= \int \mathcal{N}(y^* \mid X^*\mu_N, \sigma^2(I+X^*\Lambda_N^{-1}X^{*T}))p(\sigma^2 \mid y) d\sigma^2 \\

		&= \int \mathcal{N}(y^* \mid X^*\mu_N, \sigma^2\underbrace{(I+X^*\Lambda_N^{-1}X^{*T})}_{M_0})\cdot \textrm{InvGamma}(\sigma^2 \mid a_n, b_n) d\sigma^2 \\

		&= \int (2 \pi \sigma^2)^{-p/2} | M_0|^{-1/2}
			\exp \left[ -\frac{1}{2\sigma^2} (y^* - X^*\mu_N)^T M_0^{-1}(y^*-X^*\mu_N) \right]
			\cdot \frac{b_N^{a_N}}{\Gamma (a_N)} (\sigma^2)^{-(a_N + 1)} \exp \left[ -\frac{b_N}{\sigma^2}\right] d\sigma^2 \\

		&= \frac{b_N^{a_N}}{\Gamma(a_N)(2 \pi )^{p/2} | M_0 |^{1/2}}
			\int 
			(\sigma^2)^{-(a_N + 1 + p/2)}
			
			\exp \left[ -\frac{1}{2\sigma^2} \left( (y^* - X^*\mu_N)^T M_0^{-1}(y^*-X^*\mu_N) + 2b_N\right) \right] d\sigma^2.
\end{aligned}
$$

For the next step we do integration by substitution, using the substitutions $u=\frac{1}{\sigma^2} \Rightarrow  \| d\sigma^2 \| = \tfrac{1}{u^2}du$. 

Then we get:

$$
\begin{aligned}
	p(y^* \mid y)
		&= \frac{b_N^{a_N}}{\Gamma(a_N)(2 \pi )^{p/2} | M_0 |^{1/2}}
			\int
			u^{a_N + 1 + p/2}

			\exp \left[ -\frac{u}{2} \left( (y^* - X^*\mu_N)^T M_0^{-1}(y^*-X^*\mu_N) + 2b_N\right) \right] \frac{1}{u^2} du \\
		
		&= \frac{b_N^{a_N}}{\Gamma(a_N)(2 \pi )^{p/2} | M_0 |^{1/2}}
			\int
			u^{a_N - 1 + p/2}

			\exp \left[ -\frac{u}{2} \left( (y^* - X^*\mu_N)^T M_0^{-1}(y^*-X^*\mu_N) + 2b_N\right) \right] du \\
		
		&= \frac{b_N^{a_N}}{\Gamma(a_N)(2 \pi )^{p/2} | M_0 |^{1/2}}
			\int
			u^{a_N + p/2 -1}

			\exp \left[ - \underbrace{\left( b_N + \frac{1}{2}(y^* - X^*\mu_N)^T M_0^{-1}(y^*-X^*\mu_N) \right)}_{B}u \right] du \\
\end{aligned}
$$

Now, defining $B := b_N + \frac{1}{2}(y^{\*} - X^{\*}\mu_N)^T M_0^{-1}(y^{\*}-X^{\*}\mu_N)$ for ease of writing, and once again doing integration by substitution by substituting with $v=Bu \Rightarrow du = B^{-1}dv$, we obtain:

$$
\begin{aligned}
	p(y^* \mid y)
		&= \frac{b_N^{a_N}}{\Gamma(a_N)(2 \pi )^{p/2} | M_0 |^{1/2}}
			\int
			\left(\frac{v}{B}\right)^{a_N + p/2 -1}

			\exp \left[ - v \right] B^{-1} dv \\

		&= \frac{b_N^{a_N}}{\Gamma(a_N)(2 \pi )^{p/2} | M_0 |^{1/2}}
			B^{-(a_N+p/2)+1}
			\int
			v^{a_N + p/2 -1}

			\exp \left[ - v \right] B^{-1} dv \\

		&= \frac{b_N^{a_N}}{\Gamma(a_N)(2 \pi )^{p/2} | M_0 |^{1/2}}
			B^{-(a_N+p/2)}
			\underbrace{\int
			v^{(a_N + p/2) -1}

			\exp \left[ - v \right] dv}_{\Gamma(a_N + p/2)} \\

\end{aligned}
$$

We recognize the integral as the Gamma function $\Gamma(a_N + p/2)$. 

Lastly, doing some final rearranging:

$$
\begin{aligned}

	p(y^* \mid y)
		&= \frac{
				b_N^{a_N} \Gamma(a_N + p/2)
			}{
				\Gamma(a_N)(2 \pi )^{p/2} | M_0 |^{1/2}
			} 
			B^{-(a_N+p/2)}
			\\
		
		
		&= \frac{
				b_N^{a_N} \Gamma(a_N + p/2)
			}{
				\Gamma(a_N)(2 \pi )^{p/2} | M_0 |^{1/2}
			} 
			\left( b_N + \frac{1}{2}(y^* - X^*\mu_N)^T M_0^{-1}(y^*-X^*\mu_N) \right)^{-(a_N+p/2)}
			\\

		&= \frac{
				b_N^{a_N} \Gamma(a_N + p/2)
			}{
				\Gamma(a_N)(2 \pi )^{p/2} | M_0 |^{1/2}
			} 
			\left( b_N(1 + \frac{1}{2b_N}(y^* - X^*\mu_N)^T M_0^{-1}(y^*-X^*\mu_N) ) \right)^{-(a_N+p/2)}
			\\
		
		&= \frac{
				b_N^{a_N} \Gamma(a_N + p/2) b_N^{-(a_N + p/2)}
			}{
				\Gamma(a_N)(2 \pi)^{p/2} | M_0 |^{1/2}
			} 
			\left( 1+ \frac{1}{2a_N}(y^* - X^*\mu_N)^T \left(\tfrac{b_N}{a_N} M_0 \right)^{-1}(y^*-X^*\mu_N) \right)^{-(a_N+p/2)}
			\\

		&= \frac{
				b_N^{-p/2} \Gamma(a_N + p/2)
			}{
				\Gamma(a_N)(2 \pi)^{p/2} | M_0 |^{1/2}
			} 
			\left( 1+ \frac{1}{2a_N}(y^* - X^*\mu_N)^T \left(\tfrac{b_N}{a_N} M_0 \right)^{-1}(y^*-X^*\mu_N) \right)^{-(a_N+p/2)}
			\\

		&= \frac{
				\Gamma(a_N + p/2) 
			}{
				\Gamma(a_N)(2 \pi a_N)^{p/2} | \tfrac{b_N}{a_N} M_0  |^{1/2}
			} 
			\left( 1+ \frac{1}{2a_N}(y^* - X^*\mu_N)^T \left(\tfrac{b_N}{a_N} M_0 \right)^{-1}(y^*-X^*\mu_N) \right)^{-(a_N+p/2)},
			\\

\end{aligned}
$$

we end up with a result that we recognize as a multivariate Student's $t$-distribution with:
- degrees of freedom $2a_N$,
- mean $X^{*}\mu_N$,
- shape matrix $ \frac{b_N}{a_N} M_0 = \frac{b_N}{a_N} ( I + X^{*} \Lambda_N^{-1} X^{\*T}) $.

## Conclusion
Given a Normal likelihood with conjugate Normal-inverse-gamma prior on $(\beta, \sigma^2)$:

$$
\beta \mid \sigma^2 \sim \mathcal{N}(\mu_0, \sigma^2 \Lambda_0^{-1}), \quad \sigma^2 \sim \text{InvGamma}(a_0, b_0),
$$

we obtain closed-form posteriors:

$$
\beta \mid \sigma^2, y \sim \mathcal{N}(\mu_N, \sigma^2 \Lambda_N^{-1}), \quad \sigma^2 \mid y \sim \text{InvGamma}(a_N, b_N),
$$

with updated parameters:

$$
\begin{aligned}
\Lambda_N &= X^T X + \Lambda_0, \\
\mu_N &= \Lambda_N^{-1}(X^T y + \Lambda_0 \mu_0), \\
a_N &= a_0 + \tfrac{n}{2}, \\
b_N &= b_0 + \tfrac{1}{2}(y^T y + \mu_0^T \Lambda_0 \mu_0 - \mu_N^T \Lambda_N \mu_N).
\end{aligned}
$$

The marginal likelihood integrates out both parameters:

$$
p(y) = (2\pi)^{-n/2}\sqrt{
	\frac{
		|
		\Lambda_0
		|
		}{
		|
		\Lambda_N
		|
		}}
	\frac{
		b_0^{a_0}
		}{
		b_N^{a_N}
		}
	\frac{
		\Gamma(a_N)
		}{
		\Gamma(a_0)
		}.,
$$

and the posterior predictive for new $y^{\*}$ given new $X^{\*}$ is:

$$
y^* \mid X^*, y \sim \text{Student-}t_{2a_N} \left( X^{*} \mu_N,\ 
\frac{b_N}{a_N} \left(1 + X^{*} \Lambda_N^{-1} X^* \right) \right).
$$

Conjugacy gives us full analytical tractability: posteriors, predictive distribution, and model evidence — all in closed form.

---

[^1]: Wessermann, L. (2004). *All of Statistics: A Concise Course in Statistical Inference*, p. 40