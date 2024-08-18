# GridWorldTask

**What is SARSA**

SARSA, which expands to State, Action, Reward, State, Action, is an on-policy value-based approach. As a form of value iteration, we need a value update rule.

For SARSA, we show this in equation:

   \begin{equation*} Q(s_t,a_t) = Q(a_t,a_t)+ \alpha ( r_t + \gamma (Q(s_{t+1},a_{t+1}) - Q(s_{t},a_{t})) ) \end{equation*}

The Q-value update rule is what distinguishes SARSA from Q-learning. In SARSA we see that the time difference value is calculated using the current state-action combo and the next state-action combo. This means we need to know the next action our policy takes in order to perform an update step.This makes SARSA an on-policy algorithm as it is updated based on the current choices of our policy.

**What is Q-Learning**

Q-learning differs from SARSA in its update rule by assuming the use of the optimal policy. The use of the \max_{a} function over the available actions makes the Q-learning algorithm an off-policy approach. This is because the policy we are updating differs in behavior from the policy we use to explore the world, which uses an exploration parameter \epsilon to choose between the best-identified action and a random choice of action:

   \begin{equation*} Q(S,A) = Q(S_t,A_t)+ \alpha ( r_t + \gamma (\max_{a}Q(S_{t+1},a) - Q(S_{t},A_{t})) ) \end{equation*} 
