---
layout: post
title: Bayesian Linear Regression
subtitle: TODO
---
Consider a standard linear regression formulation with

$$
y_{i} = \mathbf{\beta}^{T}\mathbf{x}_{i} + \epsilon_i, \quad i = 1, ..., n
$$

where $y_i$ is the scalar response for the $i$th observation; $\mathbf{x}_i$ is a $1 \times p$ vector of predictor variables; $\mathbf{\beta}$ is an $p \times 1$ vector of parameters; and $\epsilon_i$ is the unobserved error of each observation. As is customary, $x_1=1$. We will assume $\epsilon_i \sim\mathcal{N}(0, \sigma^2)$. It follows that $y_i \sim \mathcal{N}(\beta^T \mathbf{x}_i, \sigma^2)$. 

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
\begin{align}
p(y \mid \beta, \sigma^2) &= \prod_{i=1}^{n}\frac{1}{\sqrt{2\pi \sigma^2}}\exp \left[ \frac{-1}{2\sigma^2}(y_i - \mathbf{\beta}^T \mathbf{x_i})^2\right] \\
			  &= (2\pi\sigma^2)^{-n/2} \exp \left[\frac{-1}{2\sigma^2} \sum_{i=1}^{n} (y_i- \mathbf{\beta}^T \mathbf{x_i})^2 \right].
\end{align}
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
p(y \mid \beta, \sigma^2) = (2\pi\sigma^2)^{-n/2} \exp \left[ \frac{-1}{2\sigma^2} (y-X\mathbf{\beta})^T(y-X\mathbf{\beta})\right].
$$

It remains to choose a prior for $\beta$ and $\sigma^2$. We know that [the Normal-inverse-gamma distribution is the conjugate prior to the Normal likelihood](Conjugate%20prior%20for%20the%20univariate%20Normal%20distribution.md), so that would be a natural choice. So we let

$$
\displaylines{
	\beta \sim \mathcal{N}(\mu_0, \sigma^2\Lambda_0^{-1}) \\
	\sigma^2 \sim \textrm{InvGamma}(a_0, b_0), \\
}
$$

where $\sigma^2\Lambda_0^{-1}$ is the covariance matrix. The covariance matrix is symmetric and positive-definite by definition, so it follows that $\Lambda_0$ is too. The densities for each of these priors are

$$
\begin{equation}
\displaylines{
	p(\beta \mid \sigma^2) = (2\pi\sigma^2)^{-p/2}\begin{vmatrix}\Lambda_0\end{vmatrix}^{1/2} \exp \left[-\frac{1}{2}(\beta-\mu_0)^T\Lambda_0(\beta-\mu_0) \right], \quad \textrm{and} \\
	p(\sigma^2) = \frac{b_0^{a_0}}{\Gamma(a_0)}(\sigma^2)^{-(a_0+1)}\exp \left[ \frac{-b_0}{\sigma^2} \right].
}
\end{equation}
$$

Now, we want to obtain the posterior $p(\beta, \sigma^2 \mid y)$ given by

$$
\begin{align*}
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
			}.
\end{align*}
$$

Let's start with the joint probability $p(y, \beta, \sigma^2)$ in the numerator of (?):

$$
\begin{align*}
p(y \mid \beta, \sigma^2)p(\beta \mid \sigma^2)p(\sigma^2) 
	&= (2\pi\sigma^2)^{-n/2} \exp \left[ \frac{-1}{2\sigma^2} (y-X\mathbf{\beta})^T(y-X\mathbf{\beta})\right] \\
	
	&\times (2\pi\sigma^2)^{-p/2}\begin{vmatrix}\Lambda_0\end{vmatrix}^{1/2} \exp \left[-\frac{1}{2}(\beta-\mu_0)^T\Lambda_0(\beta-\mu_0) \right] \\
	
	&\times \frac{b_0^{a_0}}{\Gamma(a_0)}(\sigma^2)^{-(a_0+1)}\exp \left[ \frac{-b_0}{\sigma^2} \right].
\end{align*}
$$

Using the trick $y-X\beta = y - X\hat{\beta} + X\hat{\beta} - X\beta$, where $\hat{\beta}$ is the Moore-Penrose pseudoinverse $\hat{\beta} = (X^T X)^{-1}X^T y$, we can write the following:

$$
\begin{align}

	(y-X\beta)^T(y-X\beta)
	
		&= (y - X\hat{\beta} + X\hat{\beta} - X\beta)^T(y - X\hat{\beta} + X\hat{\beta} - X\beta)\\
		
		&= (y - X\hat{\beta})^T(y-X\hat{\beta}) + (\hat{\beta}-\beta)^TX^TX(\hat{\beta}-\beta) + \underbrace{2(y-X\hat{\beta})^T(X\hat{\beta}-X\beta)}_{0},
		
\end{align}
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
\begin{equation}
\displaylines{
	(y-X\beta)^T(y-X\beta) + (\beta-\mu_0)^T\Lambda_0 (\beta- \mu_0) 
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
}
\end{equation}
$$

and by completing the square in $\beta$, and letting $\mu_N = \Lambda_N^{-1}b_N$, we obtain

$$
\begin{align}
	(y-X\beta)^T(y-X\beta) + (\beta-\mu_0)^T\Lambda_0 (\beta- \mu_0) 
		&= (y-X\hat{\beta})^T(y-X\hat{\beta})
			+ {\color{red}\beta^T\Lambda_N\beta - 2b_N^T\beta}
			+ \hat{\beta}^TX^TX\hat{\beta} 
			+ \mu_0^T\Lambda_0\mu_0 \\
		&= (y-X\hat{\beta})^T(y-X\hat{\beta})
			+ {\color{red}(\beta -\Lambda_N^{-1}b_N)^T\Lambda_N(\beta-\Lambda_N^{-1}b_N)
			-b_N^T\Lambda_N^{-1}b_N}
			+ \hat{\beta}^TX^TX\hat{\beta} 
			+ \mu_0^T\Lambda_0\mu_0 \\
		&= (y-X\hat{\beta})^T(y-X\hat{\beta})
			+ {\color{red}(\beta -\mu_N)^T\Lambda_N(\beta-\mu_N)
			-b_N^T\mu_N}
			+ \hat{\beta}^TX^TX\hat{\beta} 
			+ \mu_0^T\Lambda_0\mu_0 \\
		&= \underbrace{(y-X\hat{\beta})^T(y-X\hat{\beta})}_{t_1}
			+ {\color{red}(\beta -\mu_N)^T\Lambda_N(\beta-\mu_N)
			-\mu_N^T\Lambda_N\mu_N}
			+ \underbrace{\hat{\beta}^TX^TX\hat{\beta}}_{t_4}
			+ \mu_0^T\Lambda_0\mu_0.\\
\end{align}
$$

Finally, let's have a look at the first and second-to-last terms $t_1$ and $t_4$ using the substitution with the Moore-Penrose pseudoeinverse:

$$
\begin{align}
	t_1 + t_4 
		&= (y-X\hat{\beta})^T(y-X\hat{\beta}) 
			+\hat{\beta}^TX^TX\hat{\beta} \\
		&= (y-X\hat{\beta})^T(y-X\hat{\beta}) 
			+ \hat{\beta}^TX^TX(X^TX)^{-1}X^Ty \\
		&= (y-X\hat{\beta})^T(y-X\hat{\beta})
			+ \hat{\beta}^TX^Ty \\
		&= y^Ty+\hat{\beta}^TX^TX\hat{\beta}-2\hat{\beta}^TX^Ty + \hat{\beta}^TX^Ty\\
		&= y^Ty + \hat{\beta}^TX^Ty - 2\hat{\beta}^TX^Ty + \hat{\beta}^TX^Ty \\
		&= y^Ty
\end{align}
$$

Putting it together, we get

$$
(y-X\beta)^T(y-X\beta) + (\beta-\mu_0)^T\Lambda_0 (\beta- \mu_0) 
	= y^Ty 
		+ (\beta -\mu_N)^T\Lambda_N(\beta-\mu_N)
		-\mu_N^T\Lambda_N\mu_N
		+ \mu_0^T\Lambda_0\mu_0,
$$

where $\Lambda_N = X^TX + \Lambda_0$ and $\mu_N = \Lambda_N^{-1}(\mu_0^T\Lambda_0 + X^Ty)$. 

Now we can go back to (?) and rewrite the sum of the exponents as

$$
\displaylines{
	\left[-\frac{1}{2\sigma^2}(y-X\beta)^T(y-X\beta)\right] + \left[-\frac{1}{2\sigma^2}(\beta-\mu_0)^T\Lambda_0(\beta-\mu_0)\right] + \left[-\frac{b_0}{\sigma^2}\right] \\

	= -\frac{1}{2\sigma^2}\left[ (y-X\beta)^T(y-X\beta) + (\beta-\mu_0)^T\Lambda_0(\beta-\mu_0)  + 2b_0\right] \\

	= -\frac{1}{2\sigma^2} \left[
		(\beta -\mu_N)^T\Lambda_N(\beta-\mu_N)
		+ y^Ty 
		-\mu_N^T\Lambda_N\mu_N
		+ \mu_0^T\Lambda_0\mu_0 
		+ 2b_0 
		\right] \\
	= -\frac{1}{2\sigma^2} \left[
		(\beta -\mu_N)^T\Lambda_N(\beta-\mu_N)
		+ 2b_N
		\right],
}
$$

where 

$$
b_N = b_0 + \tfrac{1}{2}(y^Ty - \mu_N^T\Lambda_N\mu_N + \mu_0^T\Lambda_0\mu_0).
$$

Hence, the joint probability can be written as

$$
\begin{align}
	p(y, \beta, \sigma^2) 
		&= (2\pi\sigma^2)^{-n/2} (2\pi\sigma^2)^{-p/2}\begin{vmatrix}\Lambda_0\end{vmatrix}^{1/2} 
		\frac{b_0^{a_0}}{\Gamma(a_0)}(\sigma^2)^{-(a_0+1)}
		\exp \left[ 
			-\frac{1}{2\sigma^2}(\beta -\mu_N)^T\Lambda_N(\beta-\mu_N) 
			-\frac{b_N}{\sigma^2} 
		\right] \\
		
		&= (2\pi\sigma^2)^{-p/2}\begin{vmatrix}\Lambda_0\end{vmatrix}^{1/2}  \exp \left[ -\frac{1}{2\sigma^2}(\beta -\mu_N)^T\Lambda_N(\beta-\mu_N)  \right] \\
		&\times (\sigma^2)^{-(a_0+n/2+1)}\exp \left[ -\frac{b_N}{\sigma^2}\right] \\
		&\times (2\pi)^{-n/2}\frac{b_0^{a_0}}{\Gamma (a_0)}.
\end{align}
$$

Let's turn to the denominator of (?):

$$
\int \int p(y \mid \beta, \sigma^2)p(\beta \mid \sigma^2)p(\sigma^2)d\beta d\sigma^2.
$$

Because we have managed to write the joint probability in a rather convenient form, evaluating this integral turns out to not be too hard. Let's start with integrating out $\beta$. Only the first line in (?) has terms in $\beta$, and it looks very similar to the pdf of a multivariate normal distribution. In fact, if we write 

$$
\Sigma^{-1} = \frac{1}{\sigma^2}\Lambda_N,
$$

then we immediately know that 

$$
\displaylines{
\int \exp \left[ -\frac{1}{2\sigma^2}(\beta -\mu_N)^T\Lambda_N(\beta-\mu_N)  \right] d\beta
= (2\pi)^{p/2}\begin{vmatrix}\Sigma\end{vmatrix}^{1/2} \\
= (2\pi)^{p/2}\begin{vmatrix}\sigma^2\Lambda_N^{-1}\end{vmatrix}^{1/2} \\
= (2\pi\sigma^2)^{p/2}\begin{vmatrix}\Lambda_N\end{vmatrix}^{-1/2}.
}
$$

This follows from the definition of a multivariate normal distribution with mean $\mu_N$ and covariance matrix $\Sigma^{-1}$. 

Hence

$$
\begin{align}
\int p(y, \beta, \sigma^2) d\beta 
	&= (2\pi\sigma^2)^{-p/2}\begin{vmatrix}\Lambda_0\end{vmatrix}^{1/2}(2\pi\sigma^2)^{p/2}\begin{vmatrix}\Lambda_N\end{vmatrix}^{-1/2}  \\
		&\times (\sigma^2)^{-(a_0+n/2+1)}\exp \left[ -\frac{b_N}{\sigma^2}\right] \\ 
		&\times (2\pi)^{-n/2}\frac{b_0^{a_0}}{\Gamma (a_0)} \\

	&= \begin{vmatrix}\Lambda_0\end{vmatrix}^{1/2}
	\begin{vmatrix}\Lambda_N\end{vmatrix}^{-1/2}  \\
		&\times (\sigma^2)^{-(a_0+n/2+1)}\exp \left[ -\frac{b_N}{\sigma^2}\right] \\ 
		&\times (2\pi)^{-n/2}\frac{b_0^{a_0}}{\Gamma (a_0)}. \\
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
		\begin{vmatrix}
		\Lambda_0
		\end{vmatrix}
		}{
		\begin{vmatrix}
		\Lambda_N
		\end{vmatrix}
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
Let's also compute the posterior predictive $p(\hat{y} \mid y)$ given by

$$
\begin{align}
p(\hat{y} \mid y)  
	&= \int \int p(\hat{y} \mid \hat{X}, \beta, \sigma^2) p(\beta, \sigma^2 \mid y) d\beta d\sigma^2 \\
	
	&= \int \int p(\hat{y} \mid \hat{X}, \beta, \sigma^2) p(\beta \mid \sigma^2, y) p(\sigma^2 \mid y) d\beta d\sigma^2 \\
&= \int \underbrace{\left[ 
		\int p(\hat{y} \mid \hat{X}, \beta, \sigma^2) p(\beta \mid \sigma^2, y) d\beta
	\right]}_{p(\hat{y} \mid \sigma^2, y)}
	p(\sigma^2 \mid y) d\sigma^2.
\end{align}
$$

I have included $\hat{X}$ in the formula to make it clear that the posterior predictive depends on *new data points* $\hat{X}$. The posterior predictive answers the question:

>Given the data I've seen, and my prior beliefs, what is the distribution over new outputs $\hat{y}$, at a new inputs $\hat{X}$?"

Let's start by computing the inner integral:

$$
\int p(\hat{y} \mid \hat{X}, \beta, \sigma^2) p(\beta \mid \sigma^2, y) d\beta
	= \int \mathcal{N}(\hat{y} \mid \hat{X}\beta, \sigma^2 I) \mathcal{N}(\beta \mid \mu_N, \sigma^2 \Lambda_N^{-1}) d\beta.
$$

We can compute this without directly integrating it by using well-known theorem:

>[!info] **Theorem**
>Let $z$ follow a multivariate normal distribution
>$$z \sim \mathcal{N}(\mu, \Sigma).$$
>Suppose we can partition $z$ as $z=(z_a, z_b)$, and similarly $\mu=(\mu_{a}, \mu_b)$ and $$\Sigma = \begin{bmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb}\end{bmatrix}.$$ Then
>- the marginal distribution of $z_a$ is $z_a \sim \mathcal{N}(\mu_a, \Sigma_{aa})$,
>- the marginal distribution of $z_b$ is $z_b \sim \mathcal{N}(\mu_b, \Sigma_{bb})$, and
>- the conditional distribution of $z_b$ given $z_a$ is $$z_b \mid z_a \sim \mathcal{N}(\mu_b + \Sigma_{ba}\Sigma_{aa}^{-1}(z_a-\mu_a), \Sigma_{bb}-\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab})$$

We can apply this theorem by constructing $z=(\beta, \hat{y})$,  with $\mu = (\mu_{N}, \hat{X}\mu_{N})$, and

$$
\Sigma = \begin{bmatrix}
	\Sigma_{\beta\beta} & \Sigma_{\beta \hat{y}} \\
	\Sigma_{\hat{y} \beta} & \Sigma_{\hat{y}\hat{y}}
\end{bmatrix} =
\begin{bmatrix}
	\sigma^2 \Lambda_N^{-1} & \sigma^2\Lambda_N^{-1}\hat{X}^T \\
	\sigma^2 \hat{X}\Lambda_N^{-1} & \sigma^2(I+\hat{X}\Lambda_N^{-1}\hat{X}^T)
\end{bmatrix}.
$$

Then it follows from the theorem that $\beta \sim \mathcal{N}(\mu_N, \sigma^2\Lambda_N^{-1})$,  $\hat{y} \sim \mathcal{N}(\hat{X}\mu_N, \sigma^2(I+\hat{X}\Lambda_N^{-1}\hat{X}^T))$, and 

$$
\hat{y} \mid \beta \sim \mathcal{N}(\mu_{\hat{y}\mid \beta}, \Sigma_{\hat{y} \mid \beta}),
$$

where

$$\begin{align}
\mu_{\hat{y} \mid \beta} 
	&= \hat{X}\mu_N + \sigma^2 \hat{X}\Lambda_N^{-1}(\sigma^2\Lambda_N^{-1})^{-1}(\beta-\mu_N), \\
	&= \hat{X}\mu_N + \hat{X}(\beta - \mu_N) \\
	&= \hat{X}\beta, \\ \\

\Sigma_{\hat{y}\mid \beta} 
	&= \sigma^2(I+\hat{X}\Lambda_N^{-1}\hat{X}^T)-\sigma^2\hat{X}\Lambda_N^{-1}(\sigma^2\Lambda_N^{-1})^{-1}\sigma^2\Lambda_N^{-1}\hat{X}^T \\
	&= \sigma^2 (I+\hat{X}\Lambda_N^{-1}\hat{X}^T)-\sigma^2\hat{X}\Lambda_N^{-1}\hat{X}^T \\
	&= \sigma^2I.

\end{align}
$$

Notice that we have constructed $z$ in such a way that it is immediately apparent that 

$$
\displaylines{
p(\hat{y} \mid\sigma^2, y) 
	= \int p(\hat{y} \mid \hat{X}, \beta, \sigma^2) p(\beta \mid \sigma^2, y) d\beta 
	= \int \mathcal{N}(\hat{y} \mid \hat{X}\beta, \sigma^2 I) \mathcal{N}(\beta \mid \mu_N, \sigma^2 \Lambda_N^{-1}) d\beta \\
	= \mathcal{N}(\hat{X}\mu_N, \sigma^2(I+\hat{X}\Lambda_N^{-1}\hat{X}^T)). 
}
$$

Now computing the outer integral:

$$
\begin{align}
	p(\hat{y} \mid y) 
		&= \int \mathcal{N}(\hat{y} \mid \hat{X}\mu_N, \sigma^2(I+\hat{X}\Lambda_N^{-1}\hat{X}^T))p(\sigma^2 \mid y) d\sigma^2 \\
		&= \int \mathcal{N}(\hat{y} \mid \hat{X}\mu_N, \sigma^2(I+\hat{X}\Lambda_N^{-1}\hat{X}^T))\cdot \textrm{InvGamma}(\sigma^2 \mid a_n, b_n) d\sigma^2 \\
		&= \int -\frac{1}{2\sigma^2} 
\end{align}.
$$
