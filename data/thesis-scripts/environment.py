class IdsEnv(gym.Env):
    def __init__(self, images_per_episode=1, dataset=(x_train, y_train), random=True):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = 
            gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(LENGTH,))
        self.images_per_episode = images_per_episode
        self.step_count = 0

        self.x, self.y = dataset
        self.random = random
        self.dataset_idx = 0
    
    def step(self, action):
        done = False
        reward = int(action == self.expected_action)
        current_label = self.expected_action
        obs = self._next_obs()

        self.step_count += 1
        if self.step_count >= self.images_per_episode:
            done = True

        return obs, reward, done, {'label': current_label}

    def _next_obs(self):
        if self.random:
            next_obs_idx = random.randint(0, len(self.x) - 1)
            self.expected_action = int(self.y.iloc[next_obs_idx,:])
            obs = self.x.iloc[next_obs_idx,:]

        else:
            obs = self.x.iloc[self.dataset_idx]
            self.expected_action = int(self.y.iloc[self.dataset_idx])

            self.dataset_idx += 1
            if self.dataset_idx >= len(self.x):
                raise StopIteration()
        return obs
    
    def reset(self):
        self.step_count = 0

        obs = self._next_obs()
        return obs