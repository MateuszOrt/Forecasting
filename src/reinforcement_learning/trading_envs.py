import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SimpleTradingEnv(gym.Env):
    """Proste środowisko handlu"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, prices, initial_cash=100):
        super(SimpleTradingEnv, self).__init__()

        self.prices = prices
        self.max_steps = len(prices)
        self.initial_cash = initial_cash

        # Akcje: 0 - nic, 1 - kup, 2 - sprzedaj
        self.action_space = spaces.Discrete(3)
        # Obserwacja: [cena, gotówka, liczba akcji]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.actions_history = []
        self.reset()

    @property
    def current_price(self):
        return self.prices[min(self.step_count, self.max_steps - 1)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.cash = self.initial_cash
        self.shares = 0

        self.actions_history = []
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.current_price, self.cash, self.shares], dtype=np.float32)

    def _get_portfolio_value(self):
        return self.cash + self.shares * self.current_price

    def step(self, action):
        done = False
        reward = 0
        # previous_portfolio_value = self._get_portfolio_value()
        mean_price = np.mean(
            self.prices[: self.step_count + 1]
        )  # Średnia cena dotychczas
        # reward = self._get_portfolio_value() - self.initial_cash
        if action == 1:  # Kup
            if self.cash >= self.current_price:
                self.cash -= self.current_price
                self.shares += 1
            if self.current_price < mean_price:
                reward += 0.5  # Dodatkowa nagroda za kupno poniżej średniej

        elif action == 2:  # Sprzedaj
            if self.shares > 0:
                self.cash += self.current_price
                self.shares -= 1
            if self.current_price > mean_price:
                reward += 0.5  # Dodatkowa nagroda za sprzedaż powyżej średniej

        self.actions_history.append(action)

        # reward = (self._get_portfolio_value() - self.initial_cash)/self.initial_cash
        # reward = self._get_portfolio_value() - self.initial_cash

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, False, {}

    def render(self):
        print(
            f"Krok: {self.step_count}, Cena: {self.current_price:.2f}, Gotówka: {self.cash:.2f}, Akcje: {self.shares}, Wartość portfela: {self._get_portfolio_value():.2f}"
        )

    def close(self):
        pass


class TradingEnv(gym.Env):
    """Proste środowisko handlu"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, prices, initial_cash=1000, short_window=5, long_window=20):
        super(TradingEnv, self).__init__()
        self.prices = prices
        self.max_steps = len(prices)
        self.initial_cash = initial_cash
        self.short_window = short_window  # Krótkoterminowa krocząca
        self.long_window = long_window  # Długoterminowa krocząca

        # Akcje: 0 - nic, 1 - kup, 2 - sprzedaj
        self.action_space = spaces.Discrete(3)
        # Obserwacja: [cena, gotówka, liczba akcji]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.actions_history = []
        self.reset()

    @property
    def current_price(self):
        return self.prices[min(self.step_count, self.max_steps - 1)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.portfolio_value = self.initial_cash
        self.actions_history = []
        return self._get_obs(), {}

    def _get_obs(self):
        """
        Zwraca obserwację: [cena, gotówka, akcje, short_ma, long_ma, trend]
        """
        if self.step_count >= self.short_window:
            short_ma = np.mean(
                self.prices[self.step_count - self.short_window : self.step_count]
            )
        else:
            short_ma = self.prices[0]  # Można użyć ceny początkowej lub 0

        if self.step_count >= self.long_window:
            long_ma = np.mean(
                self.prices[self.step_count - self.long_window : self.step_count]
            )
        else:
            long_ma = self.prices[0]  # Można użyć ceny początkowej lub 0

        # short_ma = np.mean(self.prices[:self.step_count + 1][-self.short_window:])
        # long_ma = np.mean(self.prices[:self.step_count + 1][-self.long_window:])

        # Oblicz trend (prosty przykład: wzrost/spadek)
        if self.step_count > 0 and self.step_count < len(self.prices):
            trend = (
                1
                if self.prices[self.step_count] > self.prices[self.step_count - 1]
                else -1
            )
        else:
            trend = 0  # Brak trendu na początku

        return np.array(
            [self.current_price, self.cash, self.shares, short_ma, long_ma, trend],
            dtype=np.float32,
        )

    def _get_portfolio_value(self):
        return self.cash + self.shares * self.current_price

    def step(self, action):
        done = False
        reward = 0
        previous_portfolio_value = self._get_portfolio_value()
        # reward = self._get_portfolio_value() - self.initial_cash
        if action == 1:  # Kup
            if self.cash >= self.current_price:
                self.cash -= self.current_price
                self.shares += 1

        elif action == 2:  # Sprzedaj
            if self.shares > 0:
                self.cash += self.current_price
                self.shares -= 1

        elif action == 0:  # Nic nie robimy
            pass

        self.actions_history.append(action)
        # reward = (self._get_portfolio_value() - self.initial_cash)/self.initial_cash
        # reward = self._get_portfolio_value() - previous_portfolio_value

        self.step_count += 1
        if self.step_count >= self.max_steps - 1:
            done = True

        reward = self._get_portfolio_value() - previous_portfolio_value

        return self._get_obs(), reward, done, False, {}

    def render(self):
        print(
            f"Krok: {self.step_count}, Cena: {self.current_price:.2f}, Gotówka: {self.cash:.2f}, Akcje: {self.shares}, Wartość portfela: {self._get_portfolio_value():.2f}"
        )

    def close(self):
        pass


class TradingEnvAdvanced(gym.Env):
    """Proste środowisko handlu"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, prices, initial_cash=1000, short_window=5, long_window=20):
        super(TradingEnvAdvanced, self).__init__()
        self.prices = prices
        self.max_steps = len(prices)
        self.initial_cash = initial_cash
        self.short_window = short_window  # Krótkoterminowa krocząca
        self.long_window = long_window  # Długoterminowa krocząca

        # Akcje: 0 - nic, 1 - kup, 2 - sprzedaj
        self.action_space = spaces.Discrete(3)
        # Obserwacja: [cena, gotówka, liczba akcji]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.actions_history = []
        self.reset()

    @property
    def current_price(self):
        return self.prices[min(self.step_count, self.max_steps - 1)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.portfolio_value = self.initial_cash
        self.actions_history = []
        return self._get_obs(), {}

    def _get_obs(self):
        """
        Zwraca obserwację: [cena, gotówka, akcje, short_ma, long_ma, trend]
        """
        if self.step_count >= self.short_window:
            short_ma = np.mean(
                self.prices[self.step_count - self.short_window : self.step_count]
            )
        else:
            short_ma = self.prices[0]  # Można użyć ceny początkowej lub 0

        if self.step_count >= self.long_window:
            long_ma = np.mean(
                self.prices[self.step_count - self.long_window : self.step_count]
            )
        else:
            long_ma = self.prices[0]  # Można użyć ceny początkowej lub 0

        # short_ma = np.mean(self.prices[:self.step_count + 1][-self.short_window:])
        # long_ma = np.mean(self.prices[:self.step_count + 1][-self.long_window:])

        # Oblicz trend (prosty przykład: wzrost/spadek)
        if self.step_count > 0 and self.step_count < len(self.prices):
            trend = (
                1
                if self.prices[self.step_count] > self.prices[self.step_count - 1]
                else -1
            )
        else:
            trend = 0  # Brak trendu na początku

        return np.array(
            [self.current_price, self.cash, self.shares, short_ma, long_ma, trend],
            dtype=np.float32,
        )

    def _get_portfolio_value(self):
        return self.cash + self.shares * self.current_price

    def step(self, action):
        done = False
        reward = 0
        previous_portfolio_value = self._get_portfolio_value()
        # reward = self._get_portfolio_value() - self.initial_cash
        if action == 1:  # Kup
            if self.cash >= self.current_price:
                self.cash -= self.current_price
                self.shares += 1

        elif action == 2:  # Sprzedaj
            if self.shares > 0:
                self.cash += self.current_price
                self.shares -= 1

        elif action == 0:  # Nic nie robimy
            pass

        price_diff = self.current_price - self.prices[self.step_count - 1]
        # warunki do nagorody
        if action == 1:  # kupił
            reward += price_diff  # jeśli cena rośnie, to na plus

        if action == 2:
            reward -= price_diff

        self.actions_history.append(action)
        # reward = (self._get_portfolio_value() - self.initial_cash)/self.initial_cash
        reward += self._get_portfolio_value() - previous_portfolio_value

        self.step_count += 1
        if self.step_count >= self.max_steps - 1:
            done = True

        reward = self._get_portfolio_value() - previous_portfolio_value

        return self._get_obs(), reward, done, False, {}

    def render(self):
        print(
            f"Krok: {self.step_count}, Cena: {self.current_price:.2f}, Gotówka: {self.cash:.2f}, Akcje: {self.shares}, Wartość portfela: {self._get_portfolio_value():.2f}"
        )

    def close(self):
        pass
