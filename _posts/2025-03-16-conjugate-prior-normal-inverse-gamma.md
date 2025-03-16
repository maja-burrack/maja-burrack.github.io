---
layout: post
title: Conjugate Prior for the Univariate Normal Distribution
subtitle: I show that the Normal-inverse-gamma distribution is a conjugate prior to the normal distribution with unknown mean and variance.
---

Assume we have a univariate Gaussian dataset $y=\{y_{1}, ..., y_n\}$ with unknown mean $\mu$ and unknown variance $\sigma^2$, i.e.

$$
y_{i} \sim \mathcal{N}(\mu, \sigma^2), \quad i=1, ..., n,
$$

which has probability density function

$$
p(y_{i} \mid \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}}\exp \left[ -\frac{1}{2 \sigma^2} (y_i - \mu)^2 \right].
$$

If we want to estimate the parameters $\mu$ and $\sigma^2$ using [Bayesian methods]({{site.baseurl}}/bayesian-inference), we multiply the likelihood $p(y \mid \theta)$ of the data with the prior distribution $p(\theta)$ of the parameters and then normalize to obtain the posterior distribution of the parameters conditioned on the data, as given by Bayes' Theorem:

$$
p(\mu, \sigma^2 \mid y) = \frac{p(y\mid \mu, \sigma^2)p(\mu, \sigma^2)}{\int \int p(y\mid \mu, \sigma^2)p(\mu, \sigma^2)d\mu d\sigma^2}.
$$

A prior is _conjugate_ to the likelihood if the posterior belongs to the same family of probability distributions, i.e. has the same functional form.

A conjugate prior for the Normal distribution is the Normal-inverse-gamma specified as:

$$
\displaylines{
\mu \mid \sigma^2 \sim \mathcal{N}(\mu; \mu_{0}, \sigma^2) \\
\sigma^2 \sim \textrm{InvGamma}(\sigma^2 ; a, b)
}
$$

with probability density functions

$$
p(\mu \mid \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}}\exp \left[ -\frac{1}{2 \sigma^2} (\mu - \mu_0)^2 \right],
$$

and

$$
p(\sigma^2) = \frac{b^{a}}{\Gamma(a)}(\sigma^2)^{-(a+1)}\exp \left[ -b / \sigma^2 \right].
$$

To show that this is in fact conjugate to the Normal distribution, we need to show that the posterior $p(\mu, \sigma^2 \mid y)$ is proportional to $\mathcal{N}(\mu; \mu_n, \sigma^2) \cdot \textrm{InvGamma}(\sigma^2 ; a_n, b_n)$ for some parameters $\mu_n, a_n$ and $b_n.$ Let's do this.

#### Prior

The prior is the joint probability distribution $p(\mu, \sigma^2)$:

$$
\begin{align*}
p(\mu, \sigma^2) &= p(\mu \mid \sigma^2)p(\sigma^2) \\
    &= \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left[-\frac{1}{2\sigma^2}(\mu - \mu_{0})^2 \right] \cdot \frac{b^{a}}{\Gamma(a)}(\sigma^2)^{-(a+1)}\exp \left[ -b / \sigma^2 \right] 
\end{align*}
$$

#### Likelihood

$$
\begin{align*}
p(y \mid \mu, \sigma^{2}) &= \prod_{i=1}^{n} p(y_{i} \mid \mu, \sigma^{2}) \\
    &= \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi \sigma^{2}}} \exp \left[ -\frac{1}{2\sigma^{2}}(y_{i}-\mu)^2 \right] \\
    &= (2 \pi \sigma^{2})^{-n/2}\exp \left[-\frac{1}{2\sigma^2}\sum_{i=1}^{n} (y_{i}-\mu)^2 \right]
\end{align*}
$$

#### Posterior

By Bayes' Theorem, we get

$$
\begin{align*}
p(\mu, \sigma^2 \mid y) &\propto p(y \mid \mu, \sigma^2)p(\mu, \sigma^2) \\
    &= p(y \mid \mu, \sigma^2) p(\mu \mid \sigma^2) p(\sigma^2).
\end{align*}
$$

Substituting in the probability density functions, we get

$$
\begin{align*}
    p(\mu, \sigma^2 \mid y) &= (2 \pi \sigma^{2})^{-n/2}\exp \left[-\frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_{i}-\mu)^2 \right] \\
    &\times \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left[-\frac{1}{2\sigma^2}(\mu - \mu_{0})^2 \right] \\
    &\times \frac{b^{a}}{\Gamma(a)}(\sigma^2)^{-(a+1)}\exp \left[ -b / \sigma^2 \right]\\
    &\propto (\sigma^{2})^{-n/2}\exp \left[-\frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_{i}-\mu)^2 \right] \\
    &\times (\sigma^2)^{-1/2} \exp \left[-\frac{1}{2\sigma^2}(\mu - \mu_{0})^2 \right] \\
    &\times (\sigma^2)^{-(a+1)}\exp \left[ -b / \sigma^2 \right] \\
    &= (\sigma^2)^{-1/2}(\sigma^2)^{-(a+n/2+1)} \exp \left[-\frac{1}{2\sigma^2}\left( \sum_{i=1}^{n}(y_i - \mu)^2 + (\mu-\mu_0)^2 + 2b\right) \right].
\end{align*}
$$

Now we will use a trick and write $y_i-\mu = y_i - \bar{y} + \bar{y} - \mu$, where $\bar{y}=\tfrac{1}{n}\sum_{i=1}^n y_i$.

Then

$$
\begin{align*}
\sum_{i=1}^n (y_i - \mu)^2 &= \sum_{i=1}^n(y_i - \bar{y} + \bar{y} - \mu)^2 \\
    &= \sum_{i=1}^n (y_i - \bar{y})^2 + n(\bar{y}-\mu)^2
\end{align*}
$$

(see the appendix [A1](#A1) for more steps showing how to get this result).

Substituting with this in the exponential we obtain

$$
\begin{align*}
    p(\mu, \sigma^2 \mid y) & \propto (\sigma^2)^{-1/2}(\sigma^2)^{-(a+n/2+1)} \exp \left[-\frac{1}{2\sigma^2}\left(  \sum_{i=1}^n (y_i - \bar{y})^2 + n(\bar{y}-\mu)^2 + (\mu-\mu_0)^2 + 2b\right) \right] \\
    &= (\sigma^2)^{-(a+n/2+1)} \exp\left[-\frac{1}{\sigma^2} \left( \frac{1}{2}\sum_{i=1}^n (y_i - \bar{y})^2  + b\right) \right] \\
    &\times (\sigma^2)^{-1/2} \exp \left[ -\frac{1}{2\sigma^2} (n(\bar{y}-\mu)^2 + (\mu-\mu_0)^2) \right].
\end{align*}
$$

Now, by completing the square a few times, we can write

$$
\begin{align*}
n(\bar{y}-\mu)^2 + (\mu-\mu_0)^2 & = n\bar{y}^2 + n\mu^2 - 2n\mu\bar{y} + \mu^2 + \mu_0^2 - 2\mu\ \\
    &= (n+1)\mu^2 -2(n\bar{y}+\mu_0)\mu + n\bar{y}^2 + \mu_0^2 \\
    &= (n+1)\left( \mu^2 - \frac{2(n\bar{y} + \mu_0)\mu}{n+1} \right) + n\bar{y}^2 + \mu_0^2 \\
    &= (n+1)\left( \left(\mu^2 - \frac{n\bar{y} + \mu_0}{n+1} \right)^2 - \left( \frac{n\bar{y}+\mu_0}{n+1} \right)^2 \right) + n\bar{y}^2 + \mu_0^2 \\
    &= (n+1)\left(\mu^2 - \frac{n\bar{y} + \mu_0}{n+1} \right)^2 - \frac{(n\bar{y}+\mu_0)^2}{n+1} + n\bar{y}^2 + \mu_0^2 \\
    &= (n+1)\left(\mu^2 - \frac{n\bar{y} + \mu_0}{n+1} \right)^2 - \frac{(n\bar{y}+\mu_0)^2 - (n+1)(n\bar{y}^2 + \mu_0^2)}{n+1} \\
    &= (n+1)\left(\mu^2 - \frac{n\bar{y} + \mu_0}{n+1} \right)^2 + \frac{n(\bar{y}-\mu_0)^2}{n+1}.
\end{align*}
$$

That was a bit tedious, but now we can substitute with this in the exponent to get

$$
\begin{align*}
    p(\mu, \sigma^2 \mid y) & \propto (\sigma^2)^{-(a+n/2+1)} \exp\left[-\frac{1}{\sigma^2} \left( \frac{1}{2}\sum_{i=1}^n (y_i - \bar{y})^2  + b\right) \right] \\
    &\times (\sigma^2)^{-1/2} \exp \left[ -\frac{1}{2\sigma^2} (n(\bar{y}-\mu)^2 + (\mu-\mu_0)^2) \right] \\
    &= (\sigma^2)^{-(a+n/2+1)} \exp\left[-\frac{1}{\sigma^2} \left( \frac{1}{2}\sum_{i=1}^n (y_i - \bar{y})^2  + b\right) \right] \\
    &\times (\sigma^2)^{-1/2} \exp \left[ -\frac{1}{2\sigma^2} \left( (n+1)\left(\mu^2 - \frac{n\bar{y} + \mu_0}{n+1} \right)^2 + \frac{n(\bar{y}-\mu_0)^2}{n+1} \right) \right] \\
    &= (\sigma^2)^{-(a+n/2+1)} \exp\left[-\frac{1}{\sigma^2} \left( \frac{1}{2}\sum_{i=1}^n (y_i - \bar{y})^2 + \frac{n(\bar{y}-\mu_0)^2}{2(n+1)} + b\right) \right] \\
    &\times (\sigma^2)^{-1/2} \exp \left[ -\frac{n+1}{2\sigma^2}\left(\mu^2 - \frac{n\bar{y} + \mu_0}{n+1} \right)^2\right].
\end{align*}
$$

This is proportional to $\textrm{InvGamma}(\sigma^2 ; a_n, b_n) \cdot \mathcal{N}(\mu ; \mu_n, \sigma^2)$, where

$$
a_n = a + \frac{n}{2} + 1 \\
b_n = \frac{1}{2}\sum_{i=1}^n (y_i - \bar{y})^2 + \frac{n(\bar{y}-\mu_0)^2}{2(n+1)} + b \\
\mu_n = \frac{n\bar{y} + \mu_0}{n+1}
$$

and so we are finally done, having showed that the Normal-inverse-gamma distribution is conjugate to the normal likelihood.

# Appendix

## <a name=A1></a>A1.

$$
\begin{align*}
\sum_{i=1}^n (y_i - \mu)^2 &= \sum_{i=1}^n(y_i - \bar{y} + \bar{y} - \mu)^2 \\
    &= \sum_{i=1}^n \left( (y_i - \bar{y})^2 + (\bar{y}-\mu)^2 + 2(y_i-\bar{y})(\bar{y}-\mu)) \right) \\
    &= \sum_{i=1}^n \left( (y_i - \bar{y})^2 + (\bar{y}-\mu)^2 + 2y_i \bar{y} - 2y_i \mu - 2\bar{y}^2 + 2\bar{y}\mu \right) \\
    &= \sum_{i=1}^n (y_i - \bar{y})^2 + n(\bar{y}-\mu)^2 + 2\bar{y}\sum_{i=1}^n y_i - 2 \mu \sum_{i=1}^n y_i - 2n\bar{y}^2 + 2n\mu\bar{y} \\
    &= \sum_{i=1}^n (y_i - \bar{y})^2 + n(\bar{y}-\mu)^2 + 2n\bar{y}^2 - 2n \mu \bar{y} - 2n\bar{y}^2 + 2n\mu\bar{y} \\
    &= \sum_{i=1}^n (y_i - \bar{y})^2 + n(\bar{y}-\mu)^2
\end{align*}
$$
