# Masterproef

....

## DQN

In Deep-Q Networks (DQN) a Q-function is used rather than a Q-table. This function will be estimated with a neural network, namely a DQN. The DQN will learn the optimal weights such that it can output the Optimal Q Values. The DQN can be anything from a simple neural network to a CNN or RNN when working with images or text. 

### Experience replay

The experience replay used in a DQN is a buffer that saves the agent's previous experiences at each time step in a dataset. During learning, Q-learning updates are applied on samples of experience drawn uniformly from the pool of stored samples. (http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) This improves the use of previous experience, by learning with it multiple times. Q-learning updates are incremental and do not converge quickly, so multiple passes with the same data is beneficial. (https://datascience.stackexchange.com/questions/20535/what-is-experience-replay-and-what-are-its-benefits)

## Double DQN

The Q-learning algorithm is known to overestimate action values under certain conditions. (https://arxiv.org/abs/1509.06461). To combat this, Double DQN was proposed as a solution. (https://arxiv.org/abs/1509.06461). Double DQN shows to not only improve the value estimates but also leads to much higher scores on several games such as Atari. (https://arxiv.org/abs/1509.06461).

In standard Q-learning and DQN, the max operator uses the same values both to select and to evaluate an action. This makes it more likely to select overestimated values, resulting in overoptimistic value estimates. To prevent this we can decouple the selection from the evaluation. This is the idea behind Double Q-learning. (https://arxiv.org/abs/1509.06461). This means that we will arbitrarily choose one of the two value functions to be updated with experience. We then become two sets of weights, Theta and Theta'. One set of weights will determine the greedy policy and the other will determine its value. Using this algorithm, we still select the action due to the online weights. We are still estimating the value of the greedy policy according to the current values, as defined by ThetaT. However, the second set of weights is used to fairly evaluate the value of this policy. The weights get updated symmetrically by switching the roles of Theta and Theta'.

In the DQN structure, there already is a candidate present for the second value function, namely the target network. Thus, the online network will evaluate the greedy policy while the target network will estimate its value.

## Prioritized experience replay

In previous sections we explained experience replay briefly. Experience replay allows online reinforcement agents to remember and use experiences from the past. These experiences were uniformly sampled from the replay memory. With prioritized experience replay we can replay important transitions more frequently and therefore learn more efficiently. (https://arxiv.org/abs/1511.05952). More specifically, prioritized experience replay more frequently replays transitions with high expected learning progress, measured by the magnitude of their temporal-difference (TD) error. Because of the prioritization we can experience a loss of diversity, which is alleviated with stochastic prioritization, and introduce bias, which is corrected with importance sampling. 

## Dueling DQN

![Dueling architecture](dueling-architecture.png)

A further improvement is the implementation of a dueling network architecture. In this architecture, a separate estimation is made for the value and advantage function. These two estimates are then combined to produce a single output Q function. The output of the network is a set of Q values, one for each action. The Q value is calculated using following equation.
$$
\begin{aligned} &Q(s, a ; \theta, \alpha, \beta)=V(s ; \theta \beta)+ \\
&\left(A(s, a ; \theta. \alpha)-\frac{1}{|\mathcal{A}|} \sum_{a^{\prime}} A\left(s, a^{\prime} ; \theta, \alpha\right)\right)
\end{aligned}
$$
Because of this improvement, the architecture can learn which states are (or are not) valuable, without having to learn the effect of each action for each state. 

## Parameter noise

A last improvement we will discuss is the addition of parameter noise to the network architecture. Parameter noise or parameter space noise is noise that will improve the exploratory nature of the network. Through the noise, more robust models can be trained. In the literature, the parameter noise is implemented as gaussian noise. 

## Advantage Actor Critic

Advantage actor critic (A2C) is an on-policy algorithm. This means that the algorithm updates its Q-values using the Q-value of the next state s' and the current policy's action a'. This in contrast with the off-policy DQN which updates its Q-values using the Q-value of the next state s' and the greedy action a''. 

This means that off-policy estimates the return for state-action pairs assuming a greedy policy were followed even though in fact it is not following a greedy policy while on-policy will assume that the current policy continues to be followed.

## Experiments

We look to understand the performance of DRL models in a classification problem, more specifically a network intrusion problem. We analyse the generalisation of these models.

We train classification models on the datasets NSL-KDD, UNSW-NB15, CICDDOS2019, CICDOS2017 and CICIDS2017. Firstly we train the models on NSL-KDD and UNSW-NB15 to ensure the validity of our models. We then train these models on the other three datasets to evaluate the robustness and the generalization of the models. We will use singular datasets, but also a combination of multiple datasets to see the influence on the metrics of the models. 

The models we will use are models based on the theory we explained in previous sections. These models are trained and implemented:

1. Vanilla DQN
2. Double DQN
3. Dueling DQN
4. Double Dueling DQN (DDDQN)
5. DDDQN with prioritized experience replay
6. DDDQN with parameter noise
7. DDDQN with both prioritized experience replay and parameter noise
8. Advantage actor critic (A2C)

For datasets NSL-KDD and UNSW-NB15 we trained the models with and without one-hot encoding. Some experiments were also conducted while reducing the amount of features that were trained on. 

We train each model four times and report the accuracy, precision, recall (TPR), FPR, and f1 score of each model aswell as an average per architecture. 

The A2C model is trained to compare the performance of the off-policy DQN models to a on-policy RL model.

The hyperparameters have been tuned using both manual tuning and grid search. The hyperparameters were tuned on the vanilla DQN model and reused for all other models. For the A2C model, we use the default parameters from stable baselines 2.

## Results

### Accuracy




### Generalization

