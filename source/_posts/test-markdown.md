---
title: test_markdown
date: 2021-12-02 20:03:22
tags: [Machine Learning, Test]
categories: Algorithm
mathjax: true
---

# Chapter 5. Machine Learning

# The Perceptron Algorithm

<!--more-->

The problem of fitting a half-space or a linear separator consists of $n$ labeled examples $x_1,\cdots,x_n$ in $d$-dimensional space. Each example has label $+1$ or $-1$. The task is to find a $d$-dimensional vector $w$, if one exists and a threshold $t$ s.t.
$$
w\cdot x_i>t\text{ for each }x_i\text{ labelled }+1\\
w\cdot x_i<t\text{ for each }x_i\text{ labelled }-1
$$
A vector-threshold pair, $(w, t)$, satisfying the inequalities is called a "linear separator".

Add an extra coordinate to each $x_i$ and $w$, writing $\hat{x}_i=(x_i,1)$ and $\hat{w}=(w,-t)$. Suppose $l_i$ is the $\pm 1$ label on $x_i$. Then, the above inequalities can be rewritten as
$$
(\hat{w}\cdot \hat{x}_i)l_i>1
$$

#### The Perceptron Algorithm

$w\leftarrow 0$

while there exists $x_i$ with $x_il_i\cdot w\le 0$, update $w\leftarrow w+x_il_i$

---

The intuition is that correcting $w$ by adding $x_il_i$ cause the new $(w\cdot x_i)l_i$ to be higher by $x_i\cdot x_il_i^2=|x_i|^2$. If a weight vector $w^*$ satisfies $(w\cdot x_i)l_i>0$ for $1\le i\le n$, the minimum distance of any example $x_i$ to the linear separator $w^*\cdot x=0$ is called the *margin* of the separator. Scale $w^*$ so that $(w^*\cdot x_i)l_i\ge 1$ for all $i$. Then the margin of the separator is at least $1/|w^*|$. If all points lie inside a ball of radius $r$, then $r|w^*|$ is the ratio of the radius of the ball to the margin. Theorem below shows that the number of update steps of the Perceptron Algorithm is at most
the square of this quantity.

#### Theorem 5.1

If there is a $w^*$ satisfying $(w^*\cdot x_i)l_i\ge 1$ for all $i$, then the Perceptron Algorithm finds a solution $w$ with $(w\cdot x_i)l_i>0$ for all $i$ in at most $r^2|w^*|^2$ updates where $r=\max_i|x_i|$.

**Proof** Let $w^*$ satisfy the "if" condition of the theorem. Each update increases $w^Tw^*$ by at least one.
$$
(w+x_il_i)^Tw^*=w^Tw^*+x_i^Tl_iw^*\ge w^Tw^*+1
$$
On each update, $|w|^2$ increases by at most $r^2$.
$$
(w+x_il_i)^T(w+x_il_i)=|w|^2+2x_i^Tl_iw+|x_il_i|^2\le |w|^2+|x_i|^2\le |w|^2+r^2
$$
since $x_i^Tl_iw\le 0$.

If the Perceptron Algorithm makes $m$ updates, then $w^Tw^*\ge m$, and $|w|^2\le mr^2$. Then
$$
m\le |w||w^*|\le\sqrt{m}r|w^*|\implies m\le r^2|w^*|^2
$$
as desired.

# Kernel Functions and Non-linearly Separable Data

If the data is not linearly separable, then perhaps one can map the data to a higher dimensional space where it is linearly separable.

If a function $\varphi$ maps the data to another space, one can run the Perceptron Algorithm in the new space. The weight vector will be a linear function $\sum_{i=1}^nc_i\varphi(x_i)$ of the new input data. To determine if a pattern $x_j$ is correctly classified compute
$$
w^T\varphi(x_j)=\sum_{i=1}^nc_i\varphi(x_i)^T\varphi(x_j)
$$
We do not need to explicitly compute $\varphi$ if we have a function
$$
k(x_i,x_j)=\varphi(x_i)^T\varphi(x_j)
$$
called a *kernel function* and *kernel matrix* $K$ is defined as $k_{ij}=\varphi(x_i)^T\varphi(x_j)$.

# Generalizing to New Data

#### Formalizing the problem

To formalize the learning problem, assume there is some probability distribution $D$ over the instance space $X$, such that

1. our training set $S$ consists of points drawn independently at random from $D$.
2. our objective is to predict well on new points that are also drawn from $D$.

---

Let $c^*$, called the *target concept*, denote the subset of $X$ corresponding to the positive class for a desired binary classification. Our goal is to produce a set $h\subseteq X$, called our *hypothesis*. The *true error* of $h$, $err_D(h)$, is the probability it incorrectly classifies a data point drawn at random from $D$. The *training error* of $h$, $err_S(h)$, is the fraction of points in $S$ on which $h$ and $c^*$ disagree.

An hypothesis class $\mathcal{H}$ over $X$ is a collection of subsets of $X$. Given an hypothesis class $\mathcal{H}$ and training set $S$, we aim to find an hypothesis in $\mathcal{H}$ that closely agrees with $c^*$ over $S$.

## Overfitting and Uniform Convergence

#### Theorem 5.4

Let $\mathcal{H}$ be an hypothesis class and let $\epsilon$ and $\delta$ be greater than zero. If a training set $S$ of size
$$
n\ge \frac{1}{\epsilon}(\ln|\mathcal{H}|+\ln(1/\delta))
$$
is drawn from distribution $D$, then with probability greater than or equal to $1-\delta$, every $h\in \mathcal{H}$ with training error zero has true error less than $\epsilon$.

**Proof** Let $h_1,h_2,\cdots$ be the hypotheses in $\mathcal{H}$ with true error greater than or equal to $\epsilon$. Consider drawing the sample $S$ of size $n$ and let $A_i$ be the event that $h_i$ has zero training error. Since every $h_i$ has true error greater than or equal to $\epsilon$
$$
\Pr[A_i]\le (1-\epsilon)^n
$$
By the union bound over all $i$, the probability that any of these $h_i$ is consistent with $S$ is given by
$$
\Pr\left[\bigcup_iA_i\right]\le |\mathcal{H}|(1-\epsilon)^n
$$
Using the fact that $1-\epsilon\le e^{-\epsilon}$ and replacing $n$ by the sample size bound from the theorem statement, this is at most $\mathcal{H}e^{-\ln|\mathcal{H}|-\ln(1-\delta)}=\delta$ as desired.

# VC-Dimension

## Definitions and Key Theorems

#### Definition 5.1

A set system $(X, \mathcal{H})$ consists of a set $X$ and a class $\mathcal{H}$ of subsets of $X$.

---

$\mathcal{H}$ is the class of potential hypothesis, where a hypothesis $h$ is a subset of $X$.

#### Definition 5.2

A set system $(X, \mathcal{H})$ shatters a set $A$ if each subset of $A$ can be expressed as $A\cap H$ for some $h$ in $\mathcal{H}$.

#### Definition 5.3

The **VC-dimension** of $\mathcal{H}$ is the size of the largest set shattered by $\mathcal{H}$.

## VD-Dimension of Some Set System

#### Intervals of the reals

Intervals on the real line can shatter any set of two points but no set of three points since the subset of the first and last points cannot be isolated. Thus, the VC-dimension of intervals is two.

#### Pairs of intervals of the reals

Consider the family of pairs of intervals, where a pair of intervals is viewed as the set of points that are in at least one of the intervals. There exists a set of size four that can be shattered but no set of size five since the subset of first, third, and last point cannot be isolated. Thus, the VC-dimension of pairs of intervals is four.

#### Convex polygons

For any positive integer $n$, place $n$ points on the unit circle. Any subset of the points are the vertices of a convex polygon. Clearly that polygon does not contain any of the points not in the subset. This shows that convex polygons can shatter arbitrarily large sets, so the VC-dimension is infinite.

#### Halfspace in $d$-dimensions

The VC-dimension of halfspaces in $d$-dimensions is $d+1$.

There exists a set of size $d + 1$ that can be shattered by halfspaces. Select the $d$ unit coordinate vectors plus the origin to be the $d+1$ points. Suppose $A$ is any subset of these $d+1$ points. Without loss of generality assume that the origin is in $A$. Take a 0-1 vector $w$ which has $1$'s precisely in the coordinates corresponding to vectors not in $A$. Clearly $A$ lies in the half-space $w^Tx\le 0$ and the complement of $A$ lies in the complementary halfspace.

We now show that no set of $d + 2$ points in $d$-dimensions can be shattered by halfspaces. This is done by proving that any set of $d + 2$ points can be partitioned into two disjoint subsets $A$ and $B$ whose convex hulls intersect. This establishes the claim since any linear separator with $A$ on one side must have its entire convex hull on that side, so it is not possible to have a linear separator with $A$ on one side and $B$ on the other.

#### Theorem 5.9 (Radon)

Any set $S\subseteq R^d$ with $|S|\ge d+2$, can be partitioned into two disjoint subsets $A$ and $B$ such that $convex(A)\cap convex(B)\neq \empty$.

**Proof** Assume $|S|=d+2$. Form a $d\times (d+2)$ matrix $A$ with one column for each point of $S$. Add an extra row of all 1's to construct a $(d+1)\times (d+2)$ matrix $B$. Say $x=(x_1,\cdots,x_{d+2})$ is a non-zero vector with $Bx=0$. Reorder the columns so that $x_1,\cdots,x_s\ge 0$ and $x_{s+1},\cdots,x_{d+2}<0$. Normalize $x$ so $\sum_{i=1}^s|x_i|=1$. Let $a_i$ be the $i^{th}$ column of $A$. Then, $\sum_{i=1}^s|x_i|a_i=\sum_{i=s+1}^{d+2}|x_i|a_i$ and $\sum_{i=1}^s|x_i|=\sum_{i=s+1}^{d+2}|x_i|$. Since $\sum_{i=1}^s|x_i|=\sum_{i=s+1}^{d+2}|x_i|=1$, each side of $\sum_{i=1}^s|x_i|a_i=\sum_{i=s+1}^{d+2}|x_i|a_i$ is a convex combination of columns of $A$, which proves the theorem.

## Shatter Function for Set Systems of Bounded VC-Dimension

For a set system $(X,\mathcal{H})$, the shatter function $\pi_{\mathcal{H}}(n)$ is the maximum number of subsets of any set $A$ of size $n$ that can be expressed as $A\cap h$ for $h$ in $\mathcal{H}$. The function $\pi_{\mathcal{H}}(n)$ equals $2^n$ for $n$ less than or equal to the VC-dimension of $\mathcal{H}$. Define
$$
\binom{n}{\le d}=\binom{n}{0}+\cdots+\binom{n}{d}\le n^d+1
$$
The inequality holds because to choose between $1$ and $d$ elements out of $n$, for each position there are $n$ possible items if we allow duplicates. The $1$ is for $\binom{n}{0}$.

#### Lemma 5.10 (Sauer)

For any set system $(X,\mathcal{H})$ of VC-dimension at most $d$, $\pi_{\mathcal{H}}(n)\le \binom{n}{\le d}$ for all $n$.

## VC-Dimension of Combinations of Concepts

Let $(X,\mathcal{H}_1)$ and $(X,\mathcal{H}_2)$ be two set systems. Define intersection system $(X,\mathcal{H}_1\cap \mathcal{H}_2)$, where $\mathcal{H}_1\cap \mathcal{H}_2=\{h_1\cap h_2|h_1\in \mathcal{H}_1,h_2\in\mathcal{H}_2\}$.

#### Lemma 5.11

Suppose $(X,\mathcal{H}_1)$ and $(X,\mathcal{H}_2)$ are two set systems on the same set $X$. Then
$$
\pi_{\mathcal{H}_1\cap \mathcal{H}_2}(n)\le \pi_{\mathcal{H}_1}(n)\pi_{\mathcal{H}_2}(n)
$$
**Proof** Let $A\subset X$ and $\mathcal{S}=\{A\cap h|h\in \mathcal{H}_1\cap\mathcal{H}_2\}$. Let $h=h_1\cap h_2$. Then $A\cap h=(A\cap h_1)\cap (A\cap h_2)$. Therefore, $|S|\le |\{A\cap h_1|h_1\in \mathcal{H}_1\}|\cdot |\{A\cap h_2|h_2\in \mathcal{H}_2\}|$, as desired.

## The Key Theorem

#### Theorem 5.14

Let $(X,\mathcal{H})$ be a set system, $D$ a probability distribution over $X$, and let $n$ be an integer satisfying
$$
n\ge \frac{2}{\epsilon}\left[\log_22\pi_{\mathcal{H}}(2n)+\log_2\frac{1}{\delta}\right]
$$
Let $S_1$ consists of $n$ points drawn from $D$. With probability greater than or equal to $1-\delta$, every set in $\mathcal{H}$ of probability mass greater than $\epsilon$ intersects $S_1$.

**Proof** Let $A$ be the event that there exists a set $h$ in $\mathcal{H}$ of probability mass greater than or equal to $\epsilon$ that is disjoint from $S_1$. Draw a second set $S_2$ of $n$ points from $D$. Let $B$ be the event that there exists $h$ in $\mathcal{H}$ that is disjoint from $S_1$ but that contains at least $\frac{\epsilon}{2}n$ points in $S_2$. 

By Chebyshev, $\Pr[B|A]\ge\frac{1}{2}$. This means that
$$
\Pr[B]\ge\Pr[A,B]=\Pr[B|A]\Pr[A]\ge\frac{1}{2}\Pr[A]
$$
Therefore, it suffices to prove that $\Pr[B]\le\frac{\delta}{2}$. Consider a second way of picking $S_1$ and $S_2$. Draw a random set $S_3$ of $2n$ points from $D$, and then randomly partition $S_3$ into two equal pieces $S_1$ and $S_2$. 

Consider the point in time after $S_3$ has been drawn but before it has been randomly partitioned. $\mathcal{H}$ has at most $\pi_{\mathcal{H}}(2n)$ distinct intersections with $S_3$. To prove that $\Pr[B]\le\frac{\delta}{2}$, it is sufficient to prove that for any $h'\subseteq S_3$, the probability that $|S_1\cap h'|=0$ but $|S_2\cap h'|\ge\frac{\epsilon}{2}n$ is at most $\frac{\delta}{2\pi_{\mathcal{H}}(2n)}$.

Note that if $h'$ contains fewer than $\frac{\epsilon}{2}n$ points, it is impossible to have $S_2\cap h'\ge\frac{\epsilon}{2}n$. For $h'$ larger than $\frac{\epsilon}{2}n$, the probability that none of the points in $h'$ fall into $S_1$ is at most $(\frac{1}{2})^{\epsilon n/2}$ (negative correlation). Plugging in our bound on $n$ get the desired result.

# VC-dimension and Machine Learning

We have a target concept $c^*$ and a set of hypotheses $\mathcal{H}$. Let $\mathcal{H}'=\{h\Delta c^*|h\in \mathcal{H}\}$ be the collection of error regions of hypotheses in $\mathcal{H}$, where $\Delta$ refers to symmetry difference. Note that $\mathcal{H}'$ and $\mathcal{H}$ have the same VC-dimension and shatter function.

#### Theorem 5.15 (sample bound)

For any class $\mathcal{H}$ and distribution $D$, if a training sample $S$ is drawn from $D$ of size
$$
n\ge\frac{2}{\epsilon}\left[\log(2\pi_{\mathcal{H}}(2n))+\log\frac{1}{\delta}\right]
$$
then with probability greater than or equal to $1-\delta$, every $h\in\mathcal{H}$ with true error $err_D(h)\ge\epsilon$ has $err_S(h)>0$.

**Proof** The proof follows from Theorem 5.14 applied to $\mathcal{H}'$.

# Online Learning

