---
layout: post
title: Metropolis-Hastings Algorithm
subtitle: Anintroduction to the Metropolis-Hastings algorithm, one of the core methods behind Bayesian inference via MCMC. I explore how it works, why it works, and how to implement it — with references for further learning.
---

I've been doing a bit of a deep-dive into Bayesian statistics recently; mainly focusing on the basic theory that sometimes gets brushed over or assumed as known facts when learning how to use specific libraries or tools for doing Bayesian statistics (personally, I have been fiddling with the python library [PyMC](https://pymc.org)). 

I have [previously shown]({{site.baseurl}}/blog/bayesian-linear-regression) that we can obtain the posterior of a linear regression model analytically under a specific choice of prior (a so-called *conjugate prior*), but even so, it requires a lot of (tedious) work to obtain the closed form expressions. This can be fine for a simple model, where the "good" priors to use are known, but what if we want to fit a more complex model, where the conjugate prior is unknown or non-existent, or what if we have good reason to want to use a prior that isn't conjugate? Fortunately, we can sample from arbitrary probability distributions using computational methods. One of these is by using [Markov Chain Monte Carlo (MCMC)](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) methods, which is the rabbit-hole I am currently in. It isn't the only way to sample (or sometimes approximate) a probability distribution, but it is one of the most popular ones.

The libraries I have looked at all use MCMC methods to sample from the posterior (read: fit the model), so in order to better understand how these libraries work, let us have a closer look at one of those methods: the Metropolis-Hastings algorithm. The algorithm works by constructing a Markov chain, i.e., a sequence of samples $(x_1, x_2, ..., x_n)$, whose distribution converges to the target distribution $p(x)$ as $n \to \infty$. There are many ways to construct such sequences, but Metropolis-Hastings is one of the most popular and relatively simple methods. 

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

**Remark**: Because the acceptance probability involves a ratio of the target distribution values, any normalizing factor cancels out, so we don’t need to be able to sample from the target distribution directly. A function proportional to the target distribution will do, and this is one of the strengths of the algorithm.

## Why it works
The algorithm works because the sequence $x_1, x_2, ...$ converges to the target distribution $p(x)$, as noted earlier. But why does it converge to the target distribution? The proof has two steps:

1. Show that the Markov chain has a unique stationary distribution $\pi(x)$.
2. Show that $\pi(x) = p(x)$, i.e., the unique stationary distribution equals the target distribution.

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

Now refer back to the definition \eqref{eq:acceptance} of the acceptance probability to see that \eqref{eq:detailed-balance-acceptance} holds: by construction, either $A(x^\ast, x)$ or $A(x, x^\ast)$ must be equal to 1. If $A(x, x^\ast) = 1$, then we must have that $p(x^\ast)q(x \mid x^\ast) \leq p(x)q(x^\ast \mid x)$, by definition of $A(x, x^\ast)$, and hence 

$$ \frac{A(x^\ast, x)}{A(x, x^\ast)} = A(x^\ast, x) = \min \left(1,\frac{p(x^\ast)q(x \mid x^\ast)}{p(x)q(x^\ast \mid x)} \right), $$

which indeed matches the ratio

$$\frac{p(x^\ast)q(x \mid x^\ast )}{p(x) q(x^\ast \mid x)},$$

when $p(x^\ast)q(x \mid x^\ast) \leq p(x)q(x^\ast \mid x)$. Similarly for the case $A(x^\ast, x) = 1$. 

In conclusion, **detailed balance holds by construction,** and so $p(x)$ is stationary distribution.

To ensure that $p(x)$ is the *unique* stationary distribution (i.e., that the chain converges to it from any starting point), the Markov chain must be irreducible and aperiodic[^1] (some sources will thrown the word "ergodic" around instead). This comes down to our choice of proposal distribution and there are many choices that would fulfill these requirements. The latter condition holds for a random walk on any proper distribution, except for trivial exceptions, while irreducibility holds as long as we choose a proposal distribution that is able to (eventually) jump to any state {% cite BDA3 -l 279 %}. 

Also, it isn't immediately obvious how many iterations we would need before the chain converges. If we set $n$ too small, the sequence won't converge, but if we set $n$ too high, we will run out of time (or computing resources) before it happens. Similarly, if the parameter space is explored too slowly (i.e., with overly small proposal steps), the chain may fail to converge., we won't obtain a converging sequence. In practice, we assess convergence of the Markov chain using trace plots, autocorrelation diagnostics, or other convergence metrics (outside the scope of this post). Most implementations also use some strategies (e.g. discarding the first parts of the sequence, often called *warm-up*) to aid convergence. See {% cite BDA3 -l 281-4 %} for a brief discussion of some of these strategies.

## Implementation
I wrote a quick little python implementation:
```python
{% include code/metropolis-hastings.py %}
```
You can find many examples of how to implement the algorithm online, including some toy examples where you get to see the algorithm work, so instead of repeating what has been done many times before, I'll link to some resources here:
- [The Metropolis-Hastings algorithm](https://blog.djnavarro.net/posts/2023-04-12_metropolis-hastings/) by Danielle Navarro. She uses an example probability distribution to show how the algorithm works, and it's very accessible. She implements the algorithm in R, but as a python-user myself, I still found the code very easy to understand. 
- [The Metropolis Hastings Algorithm](https://stephens999.github.io/fiveMinuteStats/MH_intro.html) by Matthew Stephens. This contains a short and very simple toy example, also written in R. The algorithm implemented is actually the Metropolis algorithm, which is a special case where the proposal distribution is symmetric, i.e., $q(x \mid x^\ast) = q(x^\ast \mid x)$. Thus, the ratio $\tfrac{p(x^\ast )q(x \mid x^\ast)}{p(x)q(x^\ast \mid x)}$ reduces to $\tfrac{p(x^\ast)}{p(x)}$. 

## Final remarks
PyMC doesn't use the Metropolis-Hastings algorithm as its default sampler. Instead it uses the No-U-Turn Sampler (NUTS) {% cite hoffmanGelmanNUTS2011 %}, which is an extension of the Hamilton Monte Carlo algorithm. It's outside the scope of this post, but just wanted to quickly mention that NUTS is preferred, because 1) it has more efficient random walk strategy, and 2) it automatically tunes some parameters.
___


## References
{% bibliography --cited %}

## Footnotes
[^1]: I don't really explain what *aperiodic* means, so I will give a brief description here: it basically means that the Markov chain doesn't get trapped in cycles. If it were possible to return to a state every, say, fourth step (i.e., at steps 4, 8, 16, ...), then that state would periodic (i.e., not aperiodic). This would be an issue, because then the Markov chain might "oscillate", so to speak, and never settle in a steady (stationary) state. More formally, a state $x_i$ is aperiodic if the greatest common divisor of all possible return times to that state is 1. And if all states of an irreducible chain are aperiodic, then the whole chain is aperiodic.

