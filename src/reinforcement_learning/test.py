from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


from trading_envs import SimpleTradingEnv
from logger import TrainingLoggerCallback
from data_and_plots import generate_sine_wave


prices = generate_sine_wave()
# Stwórz instancję środowiska
env_agent = SimpleTradingEnv(prices)
env_agent = Monitor(env_agent)

# Sprawdź zgodność środowiska ze Stable-Baselines3
check_env(env_agent)

logger_callback = TrainingLoggerCallback()
# Tworzenie modelu PPO
model_agent = PPO(
    "MlpPolicy",
    env_agent,
    # tensorboard_log="./ppo_trading_tensorboard/",  # <- folder z logami
    policy_kwargs=dict(net_arch=[64, 64, 64]),
    # learning_rate=5e-4,        # domyślnie 3e-4
    # n_steps=512,                # liczba kroków zanim agent zrobi aktualizację
    # batch_size=32,               # batch size do uczenia
    # n_epochs=10,                 # ile razy przebiega po danych przy każdej aktualizacji
    # gamma=0.50,                  # współczynnik dyskontujący nagrody
    # gae_lambda=0.95,             # współczynnik GAE
    # clip_range=0.2,              # zakres klipu PPO (stabilizacja treningu)
    # ent_coef=0.01,                # zachęta do eksploracji
    verbose=1,
)

# Trening modelu
model_agent.learn(total_timesteps=50_000, callback=logger_callback)

# Zapisanie modelu
# model_agent.save("ppo_trading_agent")
