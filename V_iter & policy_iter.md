# Policy Iteration

## 1. Initialization
$V(s) \in \mathbb{R}$ and $\pi(s) \in \mathcal{A}(s)$ arbitrarily for all $s \in \mathcal{S}$.
## 2. Policy Evaluation
**Loop:**
- $\Delta \leftarrow 0$
- **Loop for each** $s \in \mathcal{S}$:
  - $v \leftarrow V(s)$
  - $$
    V(s) = \sum_{s',r} p(s', r \mid s, \pi(s)) \left[ r + \gamma V(s') \right]
    $$
  - $\Delta \leftarrow \max(\Delta, |v - V(s)|)$

**until** $\Delta < \theta$ (a small positive number determining the accuracy of estimation)
## 3. Policy Improvement
_policy-stable_ $\leftarrow$ true

**For each** $s \in \mathcal{S}$:
- _old-action_ $\leftarrow \pi(s)$
- $$
  \pi(s) = \arg\max_{a} \sum_{s',r} p(s',r \mid s, a) \left[ r + \gamma V(s') \right]
  $$
- **If** _old-action_ $\neq \pi(s)$, **then** _policy-stable_ $\leftarrow$ false **If** _policy-stable_, **then** stop and return $V \approx v_{*}$ and $\pi \approx \pi_{*}$; **else** go to 2.
