def ids(params, tp):
    
    env = IdsEnv(images_per_episode=1)
    if tp == 'DQN':
        model = deepq.DQN(
            CustomPolicy, 
            env, 
            policy_kwargs=dict(dueling=params[0], layers=[128,128]),
            double_q=params[1],
            verbose=1, 
            learning_rate=0.00025,
            buffer_size=1000000,
            exploration_fraction=0.1,
            exploration_final_eps=0.1,
            train_freq=4,
            learning_starts=5000,
            target_network_update_freq=10000,
            gamma=1.0,
            param_noise=params[3],
            prioritized_replay=params[2],
            prioritized_replay_alpha=0.6,
            batch_size=32,
        )
    if tp == 'A2C':
        model = A2C(
            MlpPolicy, 
            env, 
            verbose=1,
        )
    model.learn(
        total_timesteps=int(1.0e5),
        log_interval=int(1.0e4),
    )

    env.close()
    
    return model