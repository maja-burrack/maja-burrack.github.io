---
layout: post
title: Metropolis-Hastings Algorithm
subtitle: 
---

The Metropolis-Hastings (MH) algorithm is a popular [Markov Chain Monte Carlo (MCMC)](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) method for sampling from any probability distribution. It's especially useful when we can't sample from the distribution directly, which is often the case in Bayesian statistics, when we want to sample from a posterior distribution. I have [previously shown]({{site.baseurl}}/blog/bayesian-linear-regression) that we can obtain the posterior of a linear regression model analytically under a specific choice of prior, but even so, it requires a lot of (tedious) work to obtain the closed form expressions. With the Metropolis-Hastings algorithm, we can sample from the posterior under *any* choice of prior. It also works for models more complicated than linear regression.

The algorithm works by constructing a Markov chain, i.e. a sequence of samples $(x_1, x_2, ..., x_n)$, whose distribution converges to the target distribution $p(x)$ as $n \to \infty$. There are many ways to construct such sequences, but Metropolis-Hastings is one of the most popular and relatively simple methods. 

## How it works
Let $p(x)$ be the target distribution we want to sample from. Then to construct the sequence, proceed as follows:

<!-- 1. Initialization: 
    1. Pick a starting point $x_0$.
    2. Choose a *proposal distribution* $q(x^\ast \mid x)$, which proposes candidate states based on the current state.
2. For $t= 0, 1, 2, ...$:
    1. Draw a random number $u \sim \textrm{Uniform}(0,1)$.
    2. *Proposal step*: Sample a candidate state $x^\ast$ from the proposal distribution $q(x^\ast \mid x_{t})$.
    3. *Acceptance step*: Define the *acceptance probability* $A(x^\ast, x_t)$ as: $$A(x^\ast, x_{t}) = \min \left(1,\frac{p(x^\ast)q(x_{t} \mid x^\ast )}{p(x_{t})q(x^\ast \mid x_{t})} \right).$$
       Then set the next value of the sequence $x_{t+1}$:
       $$
       x_{t+1} = \begin{cases} x^\ast &\text{if } A(x^\ast, x_{t}) \geq u, \\ x_{t} &\text{otherwise.} \end{cases}
       $$ -->

<ol>
  <li>Initialization:
    <ol type="a">
      <li>Pick a starting point $x_0$.</li>
      <li>Choose a <em>proposal distribution</em> $q(x^\ast \mid x)$, which proposes candidate states based on the current state.</li>
    </ol>
  </li>
  <li>For <span>\( t = 0, 1, 2, \dots \)</span>:
    <ol type="a">
      <li>Draw a random number <span>\( u \sim \mathrm{Uniform}(0,1) \)</span>.</li>
      <li><em>Proposal step:</em> Sample a candidate state <span>\( x^\ast \)</span> from the proposal distribution <span>\( q(x^\ast \mid x_t) \)</span>.</li>
      <li><em>Acceptance step:</em> Define the <em>acceptance probability</em> <span>\( A(x^\ast, x_t) \)</span> as:<br>
        <span>
          \[ \begin{equation}
          A(x^\ast, x_t) = \min \left(1,\frac{p(x^\ast)q(x_t \mid x^\ast)}{p(x_t)q(x^\ast \mid x_t)} \right) \label{eq:acceptance}
          \end{equation}
          \]
        </span>
        Then set the next value of the sequence <span>\( x_{t+1} \)</span>:<br>
        <span>
          \[
          x_{t+1} = 
          \begin{cases} 
          x^\ast & \text{if } A(x^\ast, x_t) \geq u, \\
          x_t & \text{otherwise}
          \end{cases}
          \]
        </span>
      </li>
    </ol>
  </li>
</ol>

In very few words: if the current state of the sequence is $x_t$, then we accept a candidate state $x^\ast \sim q(x^\ast \mid x_t)$ with probability $A(x^\ast, x_t).$ If we reject the candidate state, we simply retain the current state for the next step in the sequence.

Remark: Since we use the **ratio** of the target distribution in the acceptance step, any normalizing factor cancels out, so we don’t need to be able to sample from the target distribution directly. A function proportional to the target distribution will do, and this is one of the strengths of the algorithm.

## Why it works
The algorithm works because the sequence $x_1, x_2, ...$ converges to the target distribution $p(x)$, as noted earlier. But why does it converge to the target distribution? The proof has two steps:

1. Show that the Markov chain has a unique stationary distribution $\pi(x)$.
2. Show that $\pi(x) = p(x)$, i.e. the unique stationary distribution equals the target distribution.

Markov chains are fully characterized by a transition probability $P(x^\ast \mid x)$ (or a transition matrix $P$ in the discrete case) that encodes the probability of transitioning from one state $x$ to another state $x^\ast$. For the first step, we will use some properties of Markov chains:

1. $\pi(x)$ is **stationary** if it satisfies the condition of **detailed balance**: $$\pi(x)P(x^\ast \mid x)=\pi(x^\ast)P(x \mid x^\ast).$$
2. $\pi(x)$ is **unique** (and the chain converges to it), if the Markov chain is **aperiodic** and **irreducible**.

Let's start by showing that $p(x)$ satisfies detailed balance:

$$\begin{equation}
p(x) P(x^\ast \mid x) = p(x^\ast) P(x \mid x^\ast ). \label{eq:detailed-balance}
\end{equation}$$

If we let $P(x^\ast \mid x) = q(x^\ast \mid x)A(x^\ast \mid x)$, in effect separating the transition from one state to another into a proposal step and an acceptance step, then the detailed balance condition \eqref{eq:detailed-balance} becomes:

$$p(x) q(x^\ast \mid x)A(x^\ast, x) = p(x^\ast)q(x \mid x^\ast )A(x, x^\ast ),$$

and with some rearranging:

$$\begin{equation}
\frac{A(x^\ast, x)}{A(x, x^\ast)} = \frac{p(x^\ast)q(x \mid x^\ast )}{p(x) q(x^\ast \mid x)}. \label{eq:detailed-balance-acceptance}
\end{equation}$$

Now refer back to the definition \eqref{eq:acceptance} of the acceptance probability to see that \eqref{eq:detailed-balance-acceptance} holds. By construction, either $A(x^\ast, x)$ or $A(x, x^\ast)$ must be equal to 1. If $A(x, x^\ast) = 1$, then we must have that $p(x^\ast)q(x \mid x^\ast) \leq p(x)q(x^\ast \mid x)$, by definition of $A(x, x^\ast)$, and hence 

$$ \frac{A(x^\ast, x)}{A(x, x^\ast)} = A(x^\ast, x) = \min \left(1,\frac{p(x^\ast)q(x \mid x^\ast)}{p(x)q(x^\ast \mid x)} \right), $$

which indeed matches the ratio

$$\frac{p(x^\ast)q(x \mid x^\ast )}{p(x) q(x^\ast \mid x)},$$

when $p(x^\ast)q(x \mid x^\ast) \leq p(x)q(x^\ast \mid x)$. Similarly for the case $A(x^\ast, x) = 1$. 

In conclusion, **detailed balance holds by construction,** and so $p(x)$ is stationary distribution.

To ensure that $p(x)$ is the *unique* stationary distribution (i.e., that the chain converges to it from any starting point), the Markov chain must be irreducible and aperiodic. This comes down to our choice of proposal distribution and there are many choices that would fulfill these requirements. The latter condition holds for a random walk on any proper distribution, except for trivial exceptions, while irreducibility holds as long as we choose a proposal distribution that is able to (eventually) jump to any state[^BDA3]. In practice, we assess convergence of the Markov chain using trace plots, autocorrelation diagnostics, or other convergence metrics (more on this later).

___
[^BDA3]: Bayesian Data Analysis, p. 279.