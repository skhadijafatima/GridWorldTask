# GridWorldTask

**What is SARSA**
SARSA is an on-policy Regression Learning algorithm. It learns an action-value function, but it updates its estimates based on the action actually taken by the current policy. The update rule for SARSA is:

Q(s,a) ← Q(s,a)+α[r+γQ(s′,a′)−Q(s,a)]

where: 'a′ is the action taken in the next state s′ according to the current policy.

**What is Q-Learning**
Q-learning is an off-policy Regression Learning algorithm that learns the value of the optimal action independently of the policy being followed. It aims to learn the optimal action-value function, Q∗(s,a) which gives the maximum expected future reward for an action a taken in state s. The update rule for Q-learning is:

Q(st,at) ← Q(st,at)+α(rt+1+γmax⁡a’Q(st+1a’)–Q(st,at))
