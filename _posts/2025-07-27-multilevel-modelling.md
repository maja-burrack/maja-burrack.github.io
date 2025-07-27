---
layout: post
title: Multilevel (or Hierarchical, or Mixed) Modelling
subtitle: The one in which the author writes a surprisingly long introduction to multilevel modelling (with math and all) and then proceeds to show how to fit one in both R and Julia with frequentist and Bayesian methods, respectively, despite not being very proficient in either language (because every company she has ever worked at has been python-houses). 
---
#### Contents
{:.no_toc}
* TOC
{:toc}

## Theory and motivation
Multilevel models are extensions of regression specifically suitable for data that is structured in groups or with different granularities. I often come across data structured in this way with data available at different levels or granularities, In the retail industry, for example, you might have data at product-level, brand-level, store-level, retailer-level etc. That poses some challenges; some of which multilevel modelling solves.

So how might we go about modelling something on an individual level, when some of the data is at a group-level?

One strategy would be to fit one model completely ignoring the group indicators but potentially including group-level predictors:

$$
\begin{equation}
y_i = \alpha + \beta X_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2),
\end{equation}
$$

where $X$ is a $n \times k$ matrix of predictors ($n$ is the number of observations, $k$ is the number of predictors). $X_i$ is then the vector of length $k$ representing the $i$th row of $X$ and $\beta$ is a column vector of length $k$ (note: $\beta$ is generally a vector, while $\alpha$ is a scalar. We will keep this notation throughout).

Such a model ignores the group-level variation beyond that explained by the group-level predictors (provided we include these in the predictors $X$). Sometimes this is fine, but if we expect the groups to be different in a way that *isn't* captured by the group-level predictors, then this model might suit our needs.

An alternative strategy is to create separate models for each group:

$$
\begin{equation}
y_{ij} = \alpha_{j} + \beta_{j} X_{ij} + \epsilon_{ij}, \quad \epsilon_{ij} \sim \mathcal{N}(0, \sigma_j^2),
\end{equation}
$$

where the $j$'s index the groups $J$. Here, the predictors $X_{ij}$ would not include any group-level predictors, nor indicators, as these would be constant within each group, and so it wouldn't make sense to include them. This is like the opposite end of the spectrum[^1].

Perhaps we can settle some place in the middle between these two extremes by allowing the intercepts or slopes (or both) to vary between the groups.

Varying-intercepts and varying-slopes are not unique to multilevel models (which we haven't even discussed yet). It's possible to specify such a model without making it "multilevel":

If we add group indicators to the model, this would essentially result in a model with varying intercepts for each group:

$$
\begin{equation}
y_i = \sum_{j} \gamma_j I_{j} + \beta X_i + \epsilon_{i}, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2).
\end{equation}
$$

Here, each $\gamma_j$ is the group $j$'s intercept. We have dropped the global intercept $\alpha$, because if we were to include it, it would cause perfect [multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity). If we wanted to include it, we should drop one of the indicators. In a similar vein, the predictors $X_i$ in this case cannot include any group-level predictors as these would be collinear with the group indicators.

If we also add interactions between the indicators and predictors, we get varying slopes for each group:

$$
\begin{equation}
y_i = \sum_{j} \gamma_j I_{j} + \sum_j \beta_j X_i I_j + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2) \label{with-indicators},
\end{equation}
$$

effectively a separate slope for each group. For the same reason as above, we have dropped the global slope $\beta$. This is an example of a varying-intercepts, varying-slopes model. We can write a general *varying-intercepts, varying-slopes model* as:

$$
\begin{equation}
y_i = \alpha_{j[i]} + \beta_{j[i]}X_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2),
\end{equation}
$$

where $j[i]$ is the group $j$ that contains the unit $i$. 

A limitation of \eqref{with-indicators} is that we cannot include group-level predictors, as they are collinear with group indicators, as mentioned earlier. It turns out we can circumvent this problem, and include both group indicators and group predictors, if we build a **multilevel model**.

## The multiple levels in multilevel models
If we have a model with varying coefficients, *and* a model for those varying coefficients, it's a multilevel model. The model for the varying coefficients could include group-level predictors, which differentiates it from classical regression. For example, we could model the varying intercepts and slopes as linear regressions on the group-level predictors $U_j$:

$$
\begin{equation}
\begin{aligned}
y_i &= \alpha_{j[i]} + \beta_{j[i]}X_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)\\
\alpha_j &= a_0 + b_0U_j + \eta_{\alpha_j}, \quad \eta_{\alpha_j} \sim \mathcal{N}(0, \sigma_{\alpha}^2)\\
\beta^{(l)}_j &= a^{(l)} + b^{(l)}U_j + \eta_{\beta^{(l)}_j}, \quad \eta_{\beta^{(l)}_j} \sim \mathcal{N}(0, \sigma_{\beta^{(l)}}^2), \quad l=1,...,k
\end{aligned} \label{slope-regr}
\end{equation} 
$$

Alternatively, we can treat the varying coefficients as random variables drawn from common distributions without group-level predictors:

$$
\begin{equation}
\alpha_j \sim \mathcal{N}(\mu_{\alpha}, \sigma_{\alpha}^2), \quad \beta^{(l)}_{j} \sim \mathcal{N}(\mu_{\beta^{(l)}}, \sigma_{\beta^{(l)}}^2), \quad l = 1,...,k \label{random-coef-distr}
\end{equation}
$$

In both cases, we capture variation between groups through the variance components, which is "the feature that distinguishes multilevel models from classical regression" {% cite gelmanMultilevel -l 1 %}. So even though the latter does not include group-level predictors, it's still considered a multilevel model, because we have specified a (probability) model for each of the coefficients. Generally, **a multilevel model is a regression in which the parameters are given a model--with parameters of its own that are also estimated from data.**

In \eqref{slope-regr}, we indicated independent (uncorrelated) normal distributions for each slope's (and the intercept's) error terms. If we also wanted to estimate the correlation between intercepts and slopes, we could specify a model like this:

$$
\begin{equation}
\begin{gathered}
y_i \sim \mathcal{N}(\alpha_{j[i]} + \beta_{j[i]}X_i, \sigma^2) \\
\begin{pmatrix}
\alpha_j \\
\beta_j
\end{pmatrix}
\sim \mathcal{N} \left(
\begin{pmatrix}
\mu_\alpha \\
\mu_\beta
\end{pmatrix},
\Sigma
\right),
\end{gathered}
\end{equation}
$$

where $\Sigma$ is the [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix). Here, we model all coefficients jointly as a multivariate normal distribution allowing for correlation between them. In the simple case when $X_i$ only contains one predictor (and hence $\beta_j$ reduces to a scalar), the covariance matrix is simply

$$
\Sigma = \begin{pmatrix}
\sigma_\alpha^2 & \rho \sigma_\alpha \sigma_\beta \\
\rho \sigma_\alpha \sigma_\beta & \sigma_\beta^2
\end{pmatrix},
$$

where $\rho$ is the correlation between the slopes and the intercepts (also called a correlation parameter). If we fix $\rho = 0$, it reduces to \eqref{random-coef-distr}. The covariance matrix can be naturally extended to higher dimensions.

Similarly, if we want to also include group-level predictors $U_j$ as in (???), we can expand the model above like this:

$$
\begin{equation}
\begin{gathered}
y_i \sim \mathcal{N}(\alpha_{j[i]} + \beta_{j[i]}X_i, \sigma^2), \\
\begin{pmatrix}
\alpha_j \\
\beta_j
\end{pmatrix}
\sim \mathcal{N} \left(
\begin{pmatrix}
a_0 + b_0 U_j \\
a_1 + b_1 U_j
\end{pmatrix},
\Sigma
\right).
\end{gathered}
\end{equation}
$$

Allowing correlation between coefficients makes for a much more flexible model, but modelling the correlations when the number of varying coefficients per group is greater than 2 can be a challenge. However, with modern statistical software is shouldn't be too much of a hassle (more on this later).

## Intermission: Shrinkage and pooling
Assigning a probability distribution to the parameters $\alpha_j$ and $\beta_j$ has the effect of pulling their estimates towards the overall means $\mu_\alpha$ and $\mu_\beta$ (this is true whether the mean is itself modelled as a regression or not). In that sense, we can say that the parameter estimates are *shrunk* or *partially pooled* towards their overall mean; hence, we call this effect *shrinkage* or *partial pooling*. In contrast, *complete pooling* occurs when we restrict the $\alpha_j$'s and $\beta_j$'s to be fixed across groups (such as (??)), and *no pooling* occurs if we fit intercepts separately for each group (such as (??))

The amount of shrinkage/pooling depends on the variances and the number of observations in each group. If there are few observations in a group, it won't carry much information and estimates will be shrunk closer to the overall mean, meaning more pooling. Similarly if the variance is small; we get more pooling as we wouldn't expect the true values of the parameters to differ much across group.

## Fitting multilevel models (with code examples)
There are many methods and software tools available for fitting multilevel models. I can't tell you if any are better than others, but here are some examples of programming languages and packages you can use:
- R (`lme4`)
- python (`statsmodels`, `pymc`)
- Julia (`MixedModels.jl`, `Turing.jl`)
Some of these use frequentist methods (`lme4`, `statsmodels`, `MixedModels.jl`), while other use [Bayesian methods]({{site.baseurl}}/blog/bayesian-inference) (`pymc`, `Turing.jl`).


Below, I will fit the same model using frequentist methods in R, and using Bayesian methods in Julia[^2], but the syntax is not too different among the various programming languages, so it shouldn't be too hard to translate it from one language to another.

We will start will a simple dataset that I scraped from the IFSC results website (you can find the scraper [here](https://maja-burrack.github.io/ifsc-results-scraper)). The dataset contains results from world cups in the boulder (a specific kind of climbing) discipline for 2025. I have only kept the results of the athletes who made it to the final. This is the data we will work with:

<table class="datatable">
  <thead>
    <tr>
      <th>event_id</th>
      <th>event_name</th>
      <th>dcat</th>
      <th>athlete_id</th>
      <th>athlete_name</th>
      <th>athlete_country</th>
      <th>comp_id</th>
      <th>gender</th>
      <th>score_quali</th>
      <th>score_semi</th>
      <th>score_final</th>

    </tr>
  </thead>
  <tbody>
    {% for row in site.data.processed_ifsc_boulder_results_2025 %}
    <tr>
      <td>{{ row.event_id }}</td>
      <td>{{ row.event_name }}</td>
      <td>{{ row.dcat }}</td>
      <td>{{ row.athlete_id }}</td>
      <td>{{ row.athlete_name }}</td>
      <td>{{ row.athlete_country }}</td>
      <td>{{ row.comp_id }}</td>
      <td>{{ row.gender }}</td>
      <td>{{ row.score_quali }}</td>
      <td>{{ row.score_semi }}</td>
      <td>{{ row.score_final }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>

For our small example, we want to try and predict the scores of the final. If I were a gambler, maybe the model could tell me whom to put my money on! 

Let's specify a model like this:

$$
\mathrm{score\_final}_i = \alpha + \beta_1 \mathrm{score\_quali}_i + \beta_2 \mathrm{score\_semi}_i + u_{j[i]} + v_{k[i]} + \epsilon_i,
$$

with

$$
u_{j} \sim \mathcal{N}(0, \sigma^2_{\mathrm{comp}}), \quad v_{k} \sim \mathcal{N}(0, \sigma^2_{\mathrm{athlete}}), \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2).
$$

This model has no varying slopes (it does have slopes, namely $\beta_1$ and $\beta_2$, but none of these vary by groups), but it does have varying intercepts for two groups: the athletes, and the competitions. We could also have specified varying slopes, but since we don't have very many observations (only 88!), we will keep it simple. If we introduce too many parameters, we won't be able to fit the model reliably. 

### Multilevel modelling in R
Using R's `lme4`, the syntax for specifying the model is this:
```R
score_final ~ 1 + score_quali + score_semi + (1 | comp_id) + (1 | athlete_id)
```
If, for example, we also wanted the slope $\beta_2$ of `score_semi` to vary for each athlete, we would change the last term to `(1 + score_semi | comp_id)`. 

Fitting the model yields:
```R
model <- lmer(
    score_final ~ 1 + score_semi + score_quali + (1 | event_name:gender) + (1 | athlete_name),
    data = data
)

summary(model)
```

<div class="scrollable-code">
<pre class="highlight"><code class="language-plaintext">Linear mixed model fit by REML ['lmerMod']
Formula: score_final ~ 1 + score_semi + score_quali + (1 | comp_id) +
    (1 | athlete_id)
   Data: data

REML criterion at convergence: 765.7

Scaled residuals:
     Min       1Q   Median       3Q      Max
-1.54204 -0.63172 -0.06045  0.66171  1.68100

Random effects:
 Groups     Name        Variance Std.Dev.
 athlete_id (Intercept) 187.1    13.68
 comp_id    (Intercept) 116.1    10.78
 Residual               190.0    13.78
Number of obs: 88, groups:  athlete_id, 37; comp_id, 11

Fixed effects:
            Estimate Std. Error t value
(Intercept)   8.2864    15.1752   0.546
score_semi    0.1727     0.1523   1.134
score_quali   0.3618     0.1335   2.711

Correlation of Fixed Effects:
            (Intr) scr_sm
score_semi  -0.441       
score_quali -0.733 -0.222
</code></pre>
</div>

We can read off of the summary that

$$
\alpha = 8.28, \quad \beta_1 = 0.17, \quad \beta_2 = 0.36.
$$

These are also referred to as fixed effects, although I find the terminology inconsistent across different resources. We can also read off the variances for the varying intercepts:

$$
\sigma^2_{\mathrm{comp}} = 116.1, \quad \sigma^2_{\mathrm{athlete}} = 116.1, \quad \sigma^2 = 190.0.
$$

We can get the varying coefficients (the $u_{j[i]}$'s and $v_{k[i]}$'s) by calling `ranef(model)`. The result is this:

```R
ranef(model)
```
<div class="scrollable-code">
<pre class="highlight"><code class="language-plaintext">$athlete_name
                  (Intercept)
AKHTAR Dayan      -11.0442441
AMAGASA Sohta       3.5316370
ANRAKU Sorato      15.3760783
AVEZOU Sam          3.3337337
AVEZOU Zélia        1.8862498
BERTONE Oriane      4.3463564
DUFFY Colin        -5.1786969
FUJIWAKI Yuji      -8.6469311
GARNBRET Janja     16.8487704
ITO Futaba         -9.2330032
JENFT Paul         -7.6332626
LEE Dohyun          1.8418759
MACKENZIE Oceania -12.1408701
Matsufuji Anon     -1.9040783
MATSUFUJI Anon      5.8048444
MCNEICE Erin        8.1260469
MEIGNAN Naïlé       0.2814486
MILNE Maximillian   6.4659660
MORONI Camilla     -1.0052101
NAKAMURA Mao        3.1474311
NARASAKI Meichi     3.8596111
NARASAKI Tomoa      3.7385412
NONAKA Miho         3.6976926
PAN Yufei          10.1213726
PEHARC Anze        -6.4021685
PRIHED Oren       -12.3083132
RICHARD Samuel      9.5467425
ROBERTS Toby        7.9891599
SANDERS Annie       1.3604972
SCHALCK Mejdi      11.1938663
SEKIKAWA Melody   -11.0982683
SEO Chaehyun       -6.6303504
SUGIMOTO Rei       -7.4438652
TESIO Giorgia     -16.1685495
UZNIK Nicolai     -11.1415945
VAN DUYSEN Hannes   5.4814844

$`event_name:gender`
                                          (Intercept)
IFSC World Cup Bern 2025:female              8.461075
IFSC World Cup Bern 2025:male               -1.842423
IFSC World Cup Innsbruck 2025:female        -4.177588
IFSC World Cup Innsbruck 2025:male          -4.351309
IFSC World Cup Keqiao 2025:female           -7.667442
IFSC World Cup Keqiao 2025:male              5.912214
IFSC World Cup Prague 2025:male              6.404660
IFSC World Cup Salt Lake City 2025:female   -1.454375
IFSC World Cup Salt Lake City 2025:male     -1.284810

with conditional variances for "athlete_name" "event_name:gender"
</code></pre>
</div>

We see very high intercepts for Janja Garnbret and Sorato Anraku, which is exactly what I excepted, because I watch climbing competitions fanatically and these two, in particular, bring home medals all the time (Janja is not human. This is a known fact).

Just for fun, I kept the world cup in Curitiba out of the sample, so we could review the predictions on unseen data with a new level:

```R
test_data$pred <- predict(model, test_data, allow.new.levels = TRUE)
test_data$residual <- round(test_data$score_final - test_data$pred, 1)

print(test_data[, c("event_name", "gender", "athlete_name", "score_final", "pred", "residual")])
```

```
   event_name                   gender athlete_name   score_final  pred residual
   <chr>                        <chr>  <chr>                <dbl> <dbl>    <dbl>
 1 IFSC World Cup Curitiba 2025 male   ANRAKU Sorato         69.7  86.0    -16.3
 2 IFSC World Cup Curitiba 2025 male   SCHALCK Mejdi         58.9  57.7      1.2
 3 IFSC World Cup Curitiba 2025 male   NARASAKI Tomoa        39    56.6    -17.6
 4 IFSC World Cup Curitiba 2025 male   AMAGASA Sohta         29.5  59.9    -30.4
 5 IFSC World Cup Curitiba 2025 male   FUJIWAKI Yuji         19.6  45.6    -26
 6 IFSC World Cup Curitiba 2025 male   PEHARC Anze           19.3  43.8    -24.5
 7 IFSC World Cup Curitiba 2025 male   JENFT Paul            19.2  48.3    -29.1
 8 IFSC World Cup Curitiba 2025 male   POSCH Jan-Luca         9.3  50.6    -41.3
 9 IFSC World Cup Curitiba 2025 female MEIGNAN Naïlé         99.6  61.8     37.8
10 IFSC World Cup Curitiba 2025 female BERTONE Oriane        99.5  72.1     27.4
11 IFSC World Cup Curitiba 2025 female MORONI Camilla        83.8  60.6     23.2
12 IFSC World Cup Curitiba 2025 female NAKAMURA Mao          69.7  64.7      5
13 IFSC World Cup Curitiba 2025 female SEKIKAWA Melo…        69.5  51.9     17.6
14 IFSC World Cup Curitiba 2025 female ITO Futaba            69.4  52.1     17.3
15 IFSC World Cup Curitiba 2025 female MATSUFUJI Anon        49.5  54.9     -5.4
16 IFSC World Cup Curitiba 2025 female SANDERS Nekaia        34.8  48.5    -13.7
```

We are not spot on, but at least the ordering for the men is correct! For the women, it's not too bad either - I think we all expected Oriane to take gold, but Naïlé did seriously well, too. 

### Bayesian Multilevel Modelling in Julia
Let's also fit the model above using Bayesian methods. We will leave R behind and use Julia instead for this.
For simplicity, we will stick to the same model specification as above.

Using `Turing.jl` to define out Bayesian multilevel model, the syntax is very different from `lme4` but much closer to the mathematical formulas above (we can even use greek letters in Julia):

```julia
using Turing

@model function bayesian_multilevel_model(data)
    N = length(data.score_final)
    Ncomp = length(levels(data.comp_id))
    Nath = length(levels(data.athlete_id))

    # Hyperpriors
    σ ~ truncated(Normal(0, 50), 0, Inf)
    σ_comp ~ truncated(Normal(0, 50), 0, Inf)
    σ_ath ~ truncated(Normal(0, 50), 0, Inf)
    
    # Fixed effects
    α ~ Normal(0, 100)
    β_semi ~ Normal(0, 10)
    β_quali ~ Normal(0, 10)
    
    # Varying intercepts for competitions and athletes
    comp_eff ~ filldist(Normal(0, σ_comp), Ncomp)
    ath_eff ~ filldist(Normal(0, σ_ath), Nath)

    # Likelihood
    for i in 1:N
        μ = α +
            β_semi * data.score_semi[i] +
            β_quali * data.score_quali[i] +
            comp_eff[data.comp_idx[i]] + # I have created indexes for comp_id and athlete_id previously
            ath_eff[data.athlete_idx[i]]
        
        data.score_final[i] ~ Normal(μ, σ)
    end
end
```

We fit the model like this (using the NUTS sampler, which I mention at the end of my post on the [Metropolis-Hastings Algorithm]({{site.baseurl}}/blog/metropolis-hastings-algorithm)):

```julia
model = bayesian_multilevel_model(data)
chain = sample(model, NUTS(), 4_000)
```

and can view the summary statistics by calling `describe(chain)`. However, the summary statistics are very different from that of the R model, and that's because it's a Bayesian model. We don't obtain a single estimate for each parameter; rather, we obtain a whole probability distribution.

Sometimes this is exactly what we want, but it makes comparing this model to the R model non-trivial. We are gonna do it anyways by simply taking the means of the probability distributions as our parameter estimates. Using the same notation as above, the parameter estimates from this model are:

$$
\begin{gathered}
    \alpha = 8.81, \quad \beta_1 = 0.41, \quad  \beta_2 = 0.11, \\
    \sigma^2_{\mathrm{comp}} = 74.23, \quad \sigma^2_{\mathrm{athlete}} = 124.74, \quad \sigma^2 = 261.45
\end{gathered}
$$

The variances are a bit different from the R estimates, but that could easily be due to my choice of priors. 

Let's also try and predict the final scores of the boulder world cup in Brazil like above. This is a bit involved, as there isn't (as far as I can tell as the time of writing) a suitable `predict` function implemented that would do the hard work for us. Therefore, I have had to write a custom `predict_score_final` function for this specific model. If you are interested, you can find all the Julia code [here](https://github.com/maja-burrack/maja-burrack.github.io/blob/cf149c358412838676b0aefa301892b97304476f/_includes/code/multilevel-model.jl). Here are the predictions:

<div class="scrollable-code">
<pre class="highlight"><code class="language-plaintext"> Row │ event_name                    gender   athlete_name     score_final  pred     residual 
     │ String                        String7  String31         Float64?     Float64  Float64
─────┼────────────────────────────────────────────────────────────────────────────────────────
   1 │ IFSC World Cup Curitiba 2025  male     ANRAKU Sorato           69.7     84.4     -14.7
   2 │ IFSC World Cup Curitiba 2025  male     SCHALCK Mejdi           58.9     56.4       2.5
   3 │ IFSC World Cup Curitiba 2025  male     NARASAKI Tomoa          39.0     56.6     -17.6
   4 │ IFSC World Cup Curitiba 2025  male     AMAGASA Sohta           29.5     59.7     -30.2
   5 │ IFSC World Cup Curitiba 2025  male     FUJIWAKI Yuji           19.6     47.1     -27.5
   6 │ IFSC World Cup Curitiba 2025  male     PEHARC Anze             19.3     45.1     -25.8
   7 │ IFSC World Cup Curitiba 2025  male     JENFT Paul              19.2     49.4     -30.2
   8 │ IFSC World Cup Curitiba 2025  male     POSCH Jan-Luca           9.3     51.0     -41.7
   9 │ IFSC World Cup Curitiba 2025  female   MEIGNAN Naïlé           99.6     61.4      38.2
  10 │ IFSC World Cup Curitiba 2025  female   BERTONE Oriane          99.5     71.0      28.5
  11 │ IFSC World Cup Curitiba 2025  female   MORONI Camilla          83.8     60.4      23.4
  12 │ IFSC World Cup Curitiba 2025  female   NAKAMURA Mao            69.7     63.8       5.9
  13 │ IFSC World Cup Curitiba 2025  female   SEKIKAWA Melody         69.5     53.6      15.9
  14 │ IFSC World Cup Curitiba 2025  female   ITO Futaba              69.4     53.0      16.4
  15 │ IFSC World Cup Curitiba 2025  female   MATSUFUJI Anon          49.5     50.9      -1.4
  16 │ IFSC World Cup Curitiba 2025  female   SANDERS Nekaia          34.8     47.7     -12.9
</code></pre>
</div>

The predictions are very similar to the predictions we obtained from the R model, which is expected. I specified some pretty weak priors for the Bayesian model in the hopes that it would yield similar results.

A lot more could be said about this Bayesian model, but that's beyond the scope of this post.

## Endnote
I only read up on multilevel models because I had to deal with one at work, but I am pleased to realise how closely it's tied with Bayesian statistics, which I have been spending quite some time reading up on recently.

Most of what I know about multilevel models, I learned from {% cite gelmanMultilevel %}. I have only read parts of it, but I highly recommend it.

---

## References
{:.no_toc}
{% bibliography --cited %}

## Footnotes
{:.no_toc}
[^1]: The groups are basically treated completely separately. I've built models like this before for work. Once, I estimated the price elasticities for *each* product in an assortment like this. I had a lot of data available, so the results were pretty good. But it might have made sense to treat some of the products as part of a larger group (such as a product category) and model in away that allows the sharing of information across products. That's basically what multilevel modelling is.
[^2]: I don't even know Julia, but I recently attended a Julia meetup and everyone's excitement about the language rubbed off on me. I had to try it. (Actually, I don't know R very well, either. Everywhere I have worked have been python houses, but it's nice exploring other options).