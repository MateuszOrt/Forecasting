import numpy as np
import matplotlib.pyplot as plt


def generate_sine_wave(length=400, noise_std=0.02, period=4, seed=None):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, period * 2 * np.pi, length)
    prices = 5 + 3 * np.sin(t) + rng.normal(0, noise_std, size=length)
    return prices


def plot(data):
    plt.figure(figsize=(10, 4))
    plt.plot(data, label="Cena")
    plt.title("Wykres cen")
    plt.xlabel("Krok czasowy")
    plt.ylabel("Cena")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def training_plot(logger_callback):
    logs = logger_callback.logs

    plt.figure(figsize=(14, 6))

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
    actions = env_predict.actions_history

    buy_points = [i for i, a in enumerate(actions) if a == 1]
    sell_points = [i for i, a in enumerate(actions) if a == 2]

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


def results_plot_continuous(env_predict, prices, threshold=0.001, size_scale=200):
    actions = env_predict.actions_history

    buy_points = [(i, a) for i, a in enumerate(actions) if a > threshold]
    sell_points = [(i, a) for i, a in enumerate(actions) if a < -threshold]

    buy_indices = [i for i, a in buy_points]
    buy_sizes = [abs(a) * size_scale for i, a in buy_points]

    sell_indices = [i for i, a in sell_points]
    sell_sizes = [abs(a) * size_scale for i, a in sell_points]

    plt.figure(figsize=(14, 6))
    plt.plot(prices, label="Cena (sinusoida)", color="blue")

    plt.scatter(
        buy_indices,
        [prices[i] for i in buy_indices],
        color="green",
        label="Kupno",
        marker="^",
        s=buy_sizes,  # s = wielkość markera
        alpha=0.7,
    )
    plt.scatter(
        sell_indices,
        [prices[i] for i in sell_indices],
        color="red",
        label="Sprzedaż",
        marker="v",
        s=sell_sizes,
        alpha=0.7,
    )

    plt.title("Wykres ceny + decyzje agenta (ciągłe akcje, wielkość ^ siła decyzji)")
    plt.xlabel("Czas")
    plt.ylabel("Cena")
    plt.legend()
    plt.grid(True)
    plt.show()


def get_shares_change_list(log_list):
    """
    Tworzy listę zmian w ilości akcji ('shares') z listy logów.

    Args:
        log_list (list of dict): Lista słowników, gdzie każdy słownik reprezentuje
                                  log z jednego kroku i zawiera klucz 'shares'.

    Returns:
        list: Lista liczb całkowitych reprezentujących zmianę ilości akcji
              w każdym kroku. Pierwszy element będzie równy ilości akcji z pierwszego kroku.
    """
    shares_changes = []

    if not log_list:
        return shares_changes

    previous_shares = 0
    for entry in log_list:
        if "shares" in entry:
            current_shares = int(entry["shares"])  # Konwertuj na int, jeśli np.float64
            change = current_shares - previous_shares
            shares_changes.append(change)
            previous_shares = current_shares
        else:
            print(f"Ostrzeżenie: Element listy nie zawiera klucza 'shares': {entry}")
            shares_changes.append(0)

    return shares_changes


def plot_shares_changes_scaled(log_list, prices, size_scale=0.5):
    """
    Tworzy wykres ceny z wizualizacją zmian w ilości akcji (kupno/sprzedaż).

    Args:
        log_list (list of dict): Lista słowników z logami środowiska,
                                  zawierająca klucz 'shares'.
        prices (list or np.array): Lista lub tablica cen do narysowania na wykresie.
        size_scale (int): Skala dla rozmiaru markera, im większa zmiana, tym większy marker.
    """
    shares_changes = get_shares_change_list(log_list)

    buy_indices = []
    buy_sizes = []
    sell_indices = []
    sell_sizes = []

    num_steps = min(len(shares_changes), len(prices))

    for i in range(num_steps):
        change = shares_changes[i]

        if change > 0:  # Kupno (ilość akcji wzrosła)
            buy_indices.append(i)
            buy_sizes.append(abs(change) * size_scale)
        elif change < 0:  # Sprzedaż (ilość akcji zmalała)
            sell_indices.append(i)
            sell_sizes.append(abs(change) * size_scale)

    plt.figure(figsize=(14, 6))
    plt.plot(prices[:num_steps], label="Cena (sinusoida)", color="blue")

    plt.scatter(
        buy_indices,
        [prices[i] for i in buy_indices],
        color="green",
        label="Kupno",
        marker="^",
        s=buy_sizes,
        alpha=0.7,
    )
    plt.scatter(
        sell_indices,
        [prices[i] for i in sell_indices],
        color="red",
        label="Sprzedaż",
        marker="v",
        s=sell_sizes,
        alpha=0.7,
    )

    plt.title("Wykres ceny + zmiany ilości akcji w portfelu")
    plt.xlabel("Czas")
    plt.ylabel("Cena / Ilość akcji")  # Zmieniono etykietę Y
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()


def plot_shares_changes_fixed_size(log_list, prices, marker_size=100):
    """
    Tworzy wykres ceny z wizualizacją zmian w ilości akcji (kupno/sprzedaż).
    Wielkość wskaźników jest stała.

    Args:
        log_list (list of dict): Lista słowników z logami środowiska,
                                  zawierająca klucz 'shares'.
        prices (list or np.array): Lista lub tablica cen do narysowania na wykresie.
        marker_size (int): Stała wielkość markera dla wszystkich wskaźników kupna/sprzedaży.
    """
    shares_changes = get_shares_change_list(log_list)

    buy_indices = []
    sell_indices = []

    # Iterujemy przez zmiany w akcjach
    num_steps = min(len(shares_changes), len(prices))

    for i in range(num_steps):
        change = shares_changes[i]

        if change > 0:  # Kupno (ilość akcji wzrosła)
            buy_indices.append(i)
        elif change < 0:  # Sprzedaż (ilość akcji zmalała)
            sell_indices.append(i)

    plt.figure(figsize=(14, 6))
    plt.plot(prices[:num_steps], label="Cena (sinusoida)", color="blue")

    plt.scatter(
        buy_indices,
        [prices[i] for i in buy_indices],
        color="green",
        label="Kupno (zmiana akcji)",
        marker="^",
        s=marker_size,  # Stały rozmiar markera
        alpha=0.7,
    )
    plt.scatter(
        sell_indices,
        [prices[i] for i in sell_indices],
        color="red",
        label="Sprzedaż (zmiana akcji)",
        marker="v",
        s=marker_size,  # Stały rozmiar markera
        alpha=0.7,
    )

    plt.title(
        "Wykres ceny + zmiany ilości akcji w portfelu (stała wielkość wskaźników)"
    )
    plt.xlabel("Czas")
    plt.ylabel("Cena / Ilość akcji")
    plt.legend(loc="upper left")  # Ustawienie legendy
    plt.grid(True)
    plt.show()
