# Policy Gradient

---
Guanya的hw2.pdf：
[[hw2.pdf]]
## REINFORCE
首先是，我们为什么要使用PG，一个逻辑上的回答就是，我们想得到的就是$\pi_\theta$ ，所以就直接求这个policy就可以了，不需要使用Value Iteration来间接求了。
###### 1. 定义式
所以首先任务就是**找到一个最优策略$\theta^*$** 使得在交互过程中，期望的回报是最大的：
$$
J(\theta) = \mathbb{E}_{\tau \sim p_{\theta}(\tau)} \left[ r(\tau) \right]\tag{1}
$$
其中$\tau$ 表示一条完整的轨迹，长度就是$T$：$\tau = (s_0, a_0, s_1, a_1, \dots, s_{T-1}, a_{T-1}, s_T)$ 
$$
p_{\theta}(\tau) = p(s_0) \prod_{t=0}^{T-1} \left[ \pi_{\theta}(a_t \mid s_t) p(s_{t+1} \mid s_t, a_t) \right]\tag{2}
$$
$p_{\theta}(\tau)$ 表示的是在策略$\pi_\theta$下，产生轨迹$\tau$的概率分布，这点其实也很好理解，因为$p_{\theta}(\tau)$ 就是每一个state概率的连乘，对于一个任务，他达到目标的state的过程会有非常多的可能，但是最后求的是对每条trajectory($\tau$)的积分，或者说是期望，所以其实是很多很多个trajectory的返回值**加权求和**的感觉，拆成离散形式就是求期望了。

###### 2. 数学表达
根据定义
$$ J(\theta) = \int p_{\theta}(\tau) r(\tau) \, d\tau \tag{3}$$
注意，这里面的$\tau$是有点抽象的，他指的是一整条轨迹，可以思考积分的定义，就是遍历所有定义域内的自变量，然后把函数值加起来，现在这个积分就表现了这个定义，遍历所有的轨迹，每个轨迹都有他的概率和这条轨迹的**总奖励**，这样就能构成$J(\theta)$了。

然后我们想要求$\nabla_{\theta} J(\theta)$直接对上面的式子求梯度，可以得到：
$$
\nabla_{\theta} J(\theta) = \nabla_{\theta} \int p_{\theta}(\tau) r(\tau) d\tau = \int \nabla_{\theta} \left[ p_{\theta}(\tau) r(\tau) \right] d\tau   \tag{4}
$$
这里**梯度和积分符号可以交换**是因为这里的积分本质上是对轨迹 $\tau$进行求和（或积分），只要积分上下限不依赖于 $\theta$，那么可以交换求导和积分顺序。

###### 3. 运算
- 这里首先有一个 log-derivative trick：
	- 证明方法：我们从概率密度函数 $p_θ(τ)$出发，目标是证明下面这个等式：$$\nabla_{\theta}p_θ(τ)=p_\theta(\tau)\nabla_{\theta}\log(p_θ​(τ))    \tag{5}$$ 根据链式法则，对 $\log p_\theta(\tau)$ 关于 $\theta$ 求梯度：
$$\nabla_\theta \log p_\theta(\tau) = \frac{1}{p_\theta(\tau)} \nabla_\theta p_\theta(\tau)   \tag{6}$$
	   将上式两边同时乘以 $p_\theta(\tau)$：
$$p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) = \nabla_\theta p_\theta(\tau)     \tag{8}$$
- $r(\tau)$是一个和$\theta$没有关系的东西（因为奖励并不因为策略而变，他是有了固定的state+action之后就会得到的东西，只依赖于s和a），所以可以把他看作一个常数，可以提到$\nabla_\theta$前面去，所以现在就变成了：$\int r(\tau) \nabla_{\theta}  p_{\theta}(\tau)  d\tau$，在这个上面用log derivative trick就可以得到
- $$∇_θ​J(θ)=∫p_θ​(τ)r(τ)∇_θ\log(p_θ​(τ))dτ \tag{9}$$
- 写成期望：$$∇_θ​J(θ)=\mathbb{E}_{τ∼p_θ​(τ)}​[r(τ)∇_θ\log(p_θ​(τ))] \tag{10}$$
- 分解$p_\theta(\tau)$

根据公式(2)，可得
$$\log p_θ​(τ)=\log p(s_0​)+\sum_{t=0}^{T−1​}[\log π_θ​(a_t​∣s_t​)+\log p(s_{t+1}​∣s_t​,a_t​)] \tag{11}$$
对它求梯度时，环境动态那部分 $\log p(s_{t+1} \mid s_t, a_t)$和$\log p(s_0)$不会影响 $\theta$，所以直接去掉就行了，因为求完梯度也是0。
所以$$∇_θ​\log p_θ​(τ)=\sum_{t=0}^{T−1​}​∇_θ\log π_θ​(a_t​∣s_t​)  \tag{12}$$
于是

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\Bigl[r(\tau)\,\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\Bigr]  \tag{13}$$
如果将总回报 $r(\tau)$ 写成时间步上的加和（或折扣加和），即
$$r(\tau) = \sum_{t=0}^{T-1} \gamma^t  r(s_t, a_t)$$
$$∇_θ​J(θ)=\mathbb{E}_{\tau \sim p_\theta(\tau)}\sum_{t=0}^{T-1}\Bigl[ ∇_θ​\logπ_θ​(a_t​∣s_t​)\sum_{k=0}^{T-1}γ^kr(s_k​,a_k​) \Bigr]$$
ps:在具体代码实现中，这个$t=0$到$T-1$的$∇_θ​\logπ_θ​(a_t​∣s_t​)$值其实是通过神经网络同时预测出来的一个序列，没有时间先后顺序。具体实现的时候，一般都会把所有的trajectory合并在一起

～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
刚刚突然想到
$$∇_θ​J(θ)=\mathbb{E}_{τ∼p_θ​(τ)}​[r(τ)\sum_{t=0}^{T−1}​∇_θ\logπ_θ​(a_t​∣s_t​)]$$
- 这个里面为什么在实现的时候用的是逐项相乘，当然是了！！！你要注意这个$r(\tau)$是在$\sum$外面的.....也就是说如果我们正常展开这部分：$$r(τ)\sum_{t=0}^{T−1}​\left[∇_θ\logπ_θ​(a_t​∣s_t​)\right]$$那就是是每一个时间步的$∇_θ\logπ_θ​(a_t​∣s_t​)$去乘以一个$r(\tau)$再求和，但是在普通的REINFORCE算法中，这个$r(\tau)$使用的是整条轨迹的返回值，所以这样的结果和先算$\sum_{t=0}^{T−1}​∇_θ\logπ_θ​(a_t​∣s_t​)$再去乘$r(\tau)$是没有区别的（**乘法分配律**），这也就是为什么guanya的pdf这里面写的是公式(a)，这只是在REINFORCE的特定情况下成立的：
$$\frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta} (a_{it} | s_{it}) \right) \left( \sum_{t=0}^{T-1} r(s_{it}, a_{it}) \right) \tag{a}$$
  因为这里用的还是最基础的REINFORCE算法，他没有用reward to go，也就是说每一个时间步乘的都是一样的$r(\tau)$，但是reward to go的公式是(b)，因为每个时间步乘的$r$是去除了以前时间步的$r$：
$$\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta} (a_{it} | s_{it}) \left( \sum_{t' = t}^{T-1} r(s_{it'}, a_{it'}) \right) \tag{b}$$
- 所以代码中使用的就是逐项相乘
```python
def update(self, observations, actions, 
           advantages, q_values=None):
           # batchsize is the total timesteps of each traj
	observations = ptu.from_numpy(observations) # shape (batchsize, state_dim)
	actions = ptu.from_numpy(actions) # shape (batch_size,)
	advantages = ptu.from_numpy(advantages)
	self.optimizer.zero_grad()
	action_dist = self.forward(observations) #shape (batchsize, action_dim) 
	log_prob = action_dist.log_prob(actions) #shape (action_dim,)
	# if not self.discrete:
	#    log_prob = log_prob.sum(1)
	policy_loss = -(log_prob*advantages).mean() # 这里就是逐项相乘然后求平均
	self.optimizer.zero_grad()
	policy_loss.backward()
	self.optimizer.step()
	train_log = {
	'Training Loss': ptu.to_numpy(policy_loss),
	}
	return train_log

```
然后还有就是明明公式中写的是对每条trajectory求平均，但是在代码中却是对每个时间步（traj数量 x 轨迹长度$T$）求平均(因为obs和actions这些都是把每条traj的数据展平成一维了)。这个是因为在梯度更新的时候这个多除以的轨迹长度T其实会被学习率吸收。

- 常数因子可以被学习率吸收的解释：
指的是在梯度下降/上升过程中，如果你把梯度乘以一个常数 ( $c$ )，那么只要同时把学习率 ($\alpha$) 除以相同的常数 ($c$ )，就能得到和原来相同的更新效果。假设有一个简单的梯度更新规则（以梯度下降为例）： $$ \theta \leftarrow \theta - \alpha \nabla_{\theta} L(\theta) $$ 如果我们把梯度 ($\nabla_{\theta} L(\theta)$) 换成 ($c \cdot \nabla_{\theta} L(\theta)$)，则更新公式变为： $$ \theta \leftarrow \theta - \alpha c \nabla_{\theta} L(\theta) $$ 你可以通过把学习率从($\alpha$) 改为 ( $\alpha'$ = $\frac{\alpha}{c}$)，让更新公式变为： $$ \theta \leftarrow \theta - \alpha' \nabla_{\theta} L(\theta) $$ 其中： $$ \alpha' = \frac{\alpha}{c} $$ 这就意味着乘以常数($c$) 的梯度和不乘常数的梯度在本质上是一样的，只要你也相应地调整学习率。它不会改变更新的方向，也不会改变最终的收敛点。
https://chatgpt.com/c/67bfa00d-b084-8010-9682-37915ecafaf2
～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～
## Reward-to-go
## Discounting
## Baseline


