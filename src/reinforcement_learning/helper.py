def predict_data_env(env, data, agent):
    env_predict = env(data)
    obs, _ = env_predict.reset(seed=42)
    done = False

    while not done:
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_predict.step(action)
        env_predict.render()  # pokaże aktualną wartość portfela
    return env_predict


# def transform_data(data):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     # Scaling dataset
#     scaled_train = scaler.fit_transform(data)
