# def predict_data_env(env, data, agent):
#     env_predict = env(data)
#     obs, _ = env_predict.reset(seed=42)
#     done = False
#     lista = []
#     while not done:
#         action, _states = agent.predict(obs, deterministic=False)
#         obs, reward, done, _, info = env_predict.step(action)
#         lista.append(env_predict.render())  # pokaże aktualną wartość portfela
#     return env_predict, lista


def predict_data_env(env, data, agent):
    env_predict = env(data)
    obs, _ = env_predict.reset(seed=42)
    done = False
    lista = []
    while not done:
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_predict.step(action)
        lista.append(env_predict.render())  # pokaże aktualną wartość portfela
    return env_predict, lista


def sum_rewards_from_log_list(log_list):
    """
    Sumuje wartości 'reward' z listy słowników zawierających logi ze środowiska.

    Args:
        log_list (list of dict): Lista słowników, gdzie każdy słownik reprezentuje
                                  log z jednego kroku i zawiera klucz 'reward'.

    Returns:
        float: Suma wszystkich wartości 'reward' z listy.
    """
    total_reward = 0.0
    for entry in log_list:
        if "reward" in entry:
            total_reward += float(
                entry["reward"]
            )  # Konwertuj na float na wypadek np.float64
        else:
            print(f"Ostrzeżenie: Element listy nie zawiera klucza 'reward': {entry}")
    return total_reward
