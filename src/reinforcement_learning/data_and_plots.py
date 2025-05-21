import numpy as np
import matplotlib.pyplot as plt


def generate_sine_wave(length=400, noise_std=0.02, seed=None):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * 2 * np.pi, length)
    prices = 10 + 5 * np.sin(t) + rng.normal(0, noise_std, size=length)
    return prices


def plot_sin(data):
    plt.figure(figsize=(10, 4))
    plt.plot(data, label="Cena (sinusoida)")
    plt.title("Wygenerowana sinusoida cen w SimpleTradingEnv")
    plt.xlabel("Krok czasowy")
    plt.ylabel("Cena")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def training_plot(logger_callback):
    # logs = logger_callback.logs
    logs = logger_callback.logs

    plt.figure(figsize=(14, 6))

    # plt.subplot(1, 2, 1)
    plt.plot(logs["timesteps"], logs["value_loss"], label="Value Loss")
    plt.plot(logs["timesteps"], logs["policy_loss"], label="Policy Gradient Loss")
    plt.plot(logs["timesteps"], logs["entropy_loss"], label="Entropy Loss")
    plt.xlabel("Timesteps")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Straty w trakcie treningu")

    plt.tight_layout()
    plt.show()


def results_plot(env_predict, prices):
    # Zakładam, że środowisko jest już przetestowane i mamy historyczne dane
    actions = env_predict.actions_history

    # Zbierz indeksy kupna i sprzedaży
    buy_points = [i for i, a in enumerate(actions) if a == 1]  # zakładamy: 1 = kup
    sell_points = [
        i for i, a in enumerate(actions) if a == 2
    ]  # zakładamy: 2 = sprzedaj

    # Tworzenie wykresu
    plt.figure(figsize=(14, 6))
    plt.plot(prices, label="Cena (sinusoida)", color="blue")
    plt.scatter(
        buy_points,
        [prices[i] for i in buy_points],
        color="green",
        label="Kupno",
        marker="^",
    )
    plt.scatter(
        sell_points,
        [prices[i] for i in sell_points],
        color="red",
        label="Sprzedaż",
        marker="v",
    )

    plt.title("Wykres ceny + decyzje agenta")
    plt.xlabel("Czas")
    plt.ylabel("Cena")
    plt.legend()
    plt.grid(True)
    plt.show()
