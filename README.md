# Randomised Mini-Max

A game is defined by states $\mathcal{S}$, actions $\mathcal{A}$ and transition function 
$\delta : \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{S} \; \cup \; \{\tilde{s}
\}$ where $\tilde{s} \notin \mathcal{S}$ denotes the illegal state.

A state $s \in \mathcal{S}$ is either non-terminal or is terminal with one of three 
possible outcomes:
1. Player 1 wins,
2. Player 2 wins,
3. or a tie

Let the player of a given state $s$ be denoted by $P(s)$

## Tree exploration
We will assume for simplicity that only legal moves need be considered.

Each edge $(s, a) \in \mathcal{S} \times \mathcal{A}$ stores two values:
1. The "value" $V(s, a) \in [-1, 1]$,
2. and the visit count $N(s, a) \in \mathbb{N}$

Starting from the root $s_0$, we traverse down the tree, selecting at timestep $t$ the 
edge $(s_t, a_t)$ which maximises:
$$
Q(s_t, a_t) + \alpha \frac{\log\left({N(s_{t-1}, a_{t-1})}\right)}{1 + N(s_t, a_t)}
$$
where
$Q(s_t, a_t) = V(s_t, a_t)$ if $P(s_t) = 1$, otherwise $Q(s_t, a_t) = -V(s_t, a_t)$, 
and $\alpha > 0$ is a hyperparameter controlling exploration vs. exploitation.

We continue the above traversal until we reach a leaf $s_L$ at time $L$. We add $s_L$ 
to the tree, setting
$$
N(s_L, a) = 0
$$
$$
\begin{equation}
V(s_L, a) =
\left\{\begin{array}{lr}
1 & \textrm{Player 1 wins} \\
-1 & \textrm{Player 2 wins} \\
0 & \textrm{It's a tie} \\
q_\theta(s_L)_a & P(s_L) = 1 \\
-q_\theta(s_L)_a & \textrm{otherwise}
\end{array}\right.
\end{equation}
$$
Where $q_\theta : \mathcal{S} \rightarrow \mathbb{R}^{|\mathcal{A}|}$ 
is a learnable neural network.

Once $s_L$ has been initialised, we backtrack up the tree setting, for all $t < L$:
$$
N(s_t, a_t) \leftarrow N(s_t, a_t) + 1
$$
$$
V(s_t, a_t) \leftarrow
\left\{\begin{array}{lr}
\gamma \max_b(V(s_{t + 1}, b)) & P(s_t) = 1 \\
\gamma \min_b(V(s_{t + 1}, b)) & \textrm{otherwise}
\end{array}\right.
$$

After $n$ simulations have been run, an action $a$ is chosen with probability
$$
P(a | s_0) = \frac{N(s_0, a)^{1/\tau}}{\sum_b N(s_0, a)^{1 / \tau}}
$$
where $\tau > 0$ is a hyperparameter controlling temperature.

## Training
We simultaneously run two independent threads.

In the data collection thread, the most recent version of the model is repeatedly 
loaded and used to generate game histories via self-play.

A game history is a set $\{(s_t, V(s_t, a))\}_{t=1, a \in \mathcal{A}}^{t=T}$.

In the training thread, The model is continuously trained based on
$$
\mathcal{L}(s) = \frac{1}{|\mathcal{A}|}\sum_a ||q_\theta(s, a) - Q(s, a)||_2^2
$$