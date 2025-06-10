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
        return self.get_observation(), {}

    def get_observation(self):
        return np.array([self.current_price, self.cash, self.shares], dtype=np.float32)

    @property
    def portfolio_value(self):
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

        return self.get_observation(), reward, done, False, {}

    def render(self):
        print(
            f"Krok: {self.step_count}, Cena: {self.current_price:.2f}, Gotówka: {self.cash:.2f}, Akcje: {self.shares}, Wartość portfela: {self.portfolio_value:.2f}"
        )

    def close(self):
        pass


# class TradingEnv(gym.Env):
#     """Proste środowisko handlu"""

#     metadata = {"render_modes": ["human"]}

#     def __init__(self, prices, initial_cash=100, short_window=5, long_window=20, trend_window=5):
#         super(TradingEnv, self).__init__()
#         self.prices = prices
#         self.max_steps = len(prices)
#         self.initial_cash = initial_cash
#         self.short_window = short_window  # Krótkoterminowa krocząca
#         self.long_window = long_window  # Długoterminowa krocząca
#         self.trend_window = trend_window
#         # Akcje: 0 - nic, 1 - kup, 2 - sprzedaj
#         self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
#         # Obserwacja: [cena, gotówka, liczba akcji]
#         self.observation_space = spaces.Box(
#             low=0, high=np.inf, shape=(6,), dtype=np.float32
#         )
#         self.actions_history = []
#         self.last_reward=0
#         self.reset()

#     @property
#     def current_price(self):
#         return self.prices[min(self.step_count, self.max_steps - 1)]

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.step_count = 0
#         self.cash = self.initial_cash
#         self.shares = 0
#         self.portfolio_value = self.initial_cash
#         self.actions_history = []
#         return self._get_obs(), {}

#     def _get_obs(self):
#         """
#         Zwraca obserwację: [cena, gotówka, akcje, short_ma, long_ma, trend]
#         """
#         if self.step_count >= self.short_window:
#             short_ma = np.mean(
#                 self.prices[self.step_count - (self.short_window-1) : self.step_count+1]
#             )
#         else:
#             short_ma = self.prices[0]  # Można użyć ceny początkowej lub 0

#         if self.step_count >= self.long_window:
#             long_ma = np.mean(
#                 self.prices[self.step_count - (self.long_window-1) : self.step_count+1]
#             )
#         else:
#             long_ma = self.prices[0]  # Można użyć ceny początkowej lub 0

#         # short_ma = np.mean(self.prices[:self.step_count + 1][-self.short_window:])
#         # long_ma = np.mean(self.prices[:self.step_count + 1][-self.long_window:])

#         # Oblicz trend (prosty przykład: wzrost/spadek)
#         x = np.array([0, 1, 2, 3, 4])
#         if self.step_count >= self.trend_window:
#             y = self.prices[self.step_count - (self.trend_window-1) : self.step_count+1]
#             a, b = np.polyfit(x, y, deg=1)
#             trend=a
#         else:
#             trend=0

#         return np.array(
#             [self.current_price, self.cash, self.shares, short_ma, long_ma, trend],
#             dtype=np.float32,
#         )

#     def _get_portfolio_value(self):
#         return self.cash + self.shares * self.current_price

#     def step(self, action):
#         done = False
#         reward = 0
#         previous_portfolio_value = self._get_portfolio_value()
#         # reward = self._get_portfolio_value() - self.initial_cash
#         action_value = np.clip(action[0], -1, 1) # ograniczenie jeżeli wartości będę w góre np -2
#         effective_price = max(self.current_price, 1e-3)  # np. nie mniejsza niż 0.001
#         day_one_amount = self.initial_cash / self.prices[0]
#         day_one_buy=effective_price * day_one_amount

#         if action_value > 0:  # Kup
#             max_affordable = int(self.cash / effective_price)
#             shares_to_buy = int(action_value * max_affordable)
#             if shares_to_buy > 0:
#                 self.cash -= shares_to_buy * effective_price
#                 self.shares += shares_to_buy

#         elif action_value < 0:  # Sprzedaj
#             shares_to_sell = int(-action_value * self.shares)
#             if shares_to_sell > 0:
#                 self.cash += shares_to_sell * effective_price
#                 self.shares -= shares_to_sell

#         self.actions_history.append(action)
#         # reward = (self._get_portfolio_value() - self.initial_cash)/self.initial_cash
#         # reward = self._get_portfolio_value() - previous_portfolio_value

#         self.step_count += 1
#         if self.step_count >= self.max_steps - 1:
#             done = True
#         reward = float(self._get_portfolio_value() - day_one_buy)
#         # value = float(self._get_portfolio_value() - day_one_buy)
#         # if value>0:
#         #     reward=1
#         # elif value<0:
#         #     reward=-1
#         self.last_reward = reward

#         return self._get_obs(), reward, done, False, {}

#     def render(self):
#         print(
#             f"Krok: {self.step_count}, Cena: {self.current_price:.2f}, Gotówka: {self.cash:.2f}, Akcje: {self.shares}, Wartość portfela: {self._get_portfolio_value():.2f}, Reward: {self.last_reward:.4f}"
#         )

#     def close(self):
#         pass


class AdvancedTradingEnv(gym.Env):
    """Zaawansowane środowisko handlu"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self, prices, initial_cash=100, short_window=5, long_window=20, trend_window=5
    ):
        super(AdvancedTradingEnv, self).__init__()
        self.prices = prices
        self.max_steps = len(prices)
        self.initial_cash = initial_cash
        self.short_window = short_window  # Krótkoterminowa krocząca
        self.long_window = long_window  # Długoterminowa krocząca
        self.trend_window = trend_window
        self.profit_change = 0
        self.trend = 0
        self.total_profit_loss = 0
        # Akcje: 0 - nic, 1 - kup, 2 - sprzedaj
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Obserwacja: [cena, gotówka, liczba akcji]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        self.actions_history = []
        self.last_reward = 0
        self.reset()

    @property
    def current_price(self):
        # return self.prices[min(self.step_count, self.max_steps - 1)]
        return max(self.prices[min(self.step_count, self.max_steps - 1)], 1e-3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.cash = self.initial_cash
        self.shares = 0  # Ustalamy 0 akcji na start
        self.average_cost_per_share = 0  # Ustalamy 0 akcji na start
        self.cost_basis_shares = 0  # Ustalamy 0 akcji na start
        self.portfolio_value = self.initial_cash
        self.profit_change = 0
        self.trend = 0
        # self.total_profit_loss = 0
        self.actions_history = []
        return self.get_observation, {}

    def get_observation(self):
        """
        Zwraca obserwację: [cena, gotówka, akcje, short_ma, long_ma, trend]
        """
        if self.step_count >= self.short_window:
            short_ma = np.mean(
                self.prices[
                    self.step_count - (self.short_window - 1) : self.step_count + 1
                ]
            )
        else:
            short_ma = self.prices[0]  # Można użyć ceny początkowej lub 0

        if self.step_count >= self.long_window:
            long_ma = np.mean(
                self.prices[
                    self.step_count - (self.long_window - 1) : self.step_count + 1
                ]
            )
        else:
            long_ma = self.prices[0]  # Można użyć ceny początkowej lub 0

        # short_ma = np.mean(self.prices[:self.step_count + 1][-self.short_window:])
        # long_ma = np.mean(self.prices[:self.step_count + 1][-self.long_window:])

        # Oblicz trend (prosty przykład: wzrost/spadek)
        x = np.array([0, 1, 2, 3, 4])
        if self.step_count >= self.trend_window:
            y = self.prices[
                self.step_count - (self.trend_window - 1) : self.step_count + 1
            ]
            a, b = np.polyfit(x, y, deg=1)
            self.trend = a
        else:
            self.trend = 0

        profit_ratio = self.current_profit / self.portfolio_value

        return np.array(
            [
                self.current_price,
                self.cash,
                self.shares,
                short_ma,
                long_ma,
                self.trend,
                profit_ratio,
            ],
            dtype=np.float32,
        )

    @property
    def portfolio_value(self):
        return self.cash + self.shares * self.current_price

    @property
    def current_profit(self):
        market_value = self.shares * self.current_price
        invested_value = self.cost_basis_shares
        profit = market_value - invested_value
        return profit

    def step(self, action):
        done = False
        reward = 0
        # realized_profit=0
        # previous_profit = self.current_profit
        previous_portfolio_value = self.portfolio_value
        # reward = self._get_portfolio_value() - self.initial_cash
        # action_value = np.clip(action[0], -1, 1) # ograniczenie jeżeli wartości będę w góre np -2
        # effective_price = max(self.current_price, 1e-3)  # np. nie mniejsza niż 0.001

        if action > 0:  # Kup
            max_affordable = int(self.cash / self.current_price)
            shares_to_buy = int(action * max_affordable)
            if shares_to_buy > 0:
                # Update average cost per share
                new_total_shares = self.shares + shares_to_buy
                total_cost_so_far = self.average_cost_per_share * self.shares
                cost_of_current_purchase = shares_to_buy * self.current_price

                self.average_cost_per_share = (
                    total_cost_so_far + cost_of_current_purchase
                ) / new_total_shares
                self.cost_basis_shares = self.average_cost_per_share * new_total_shares

                # Update cash and shares
                self.shares = new_total_shares
                self.cash -= cost_of_current_purchase

        elif action < 0:  # Sprzedaj
            shares_to_sell = int(-action * self.shares)
            # if shares_to_sell == 0 and self.shares > 0 and action_value < 0:
            #     shares_to_sell = 1
            if shares_to_sell > 0:
                # self.cash += shares_to_sell * effective_price
                self.shares -= shares_to_sell
                if self.shares == 0:
                    self.average_cost_per_share = 0
                    self.cost_basis_shares = 0
                else:
                    self.cost_basis_shares = self.average_cost_per_share * self.shares
                self.cash += shares_to_sell * self.current_price
                # transaction_profit_loss = (effective_price - self.average_cost_per_share) * shares_to_sell
                # self.total_profit_loss += transaction_profit_loss

        self.actions_history.append(action)

        self.step_count += 1
        if self.step_count >= self.max_steps - 1:
            done = True

        reward = self.portfolio_value - previous_portfolio_value

        # if self.step_count > 1:
        #     self.profit_change=self.current_profit-previous_profit

        self.last_reward = reward

        return self.get_observation, reward, done, False, {}

    def render(self):
        log_string = (
            f"Krok: {self.step_count} | "
            f"Cena: {self.current_price:.2f} | "
            f"Gotówka: {self.cash:.2f} | "
            f"Akcje: {self.shares} | "
            f"Wartość portfela: {self.portfolio_value:.2f} | "
            f"Reward: {self.last_reward:.4f} | "
            f"Profit: {self.current_profit/self.portfolio_value:.4f} | "
            # f"Total profit: {self.total_profit_loss:.4f}"
        )
        print(log_string)
        return {
            "step": self.step_count,
            "price": self.current_price,
            "cash": self.cash,
            "shares": self.shares,
            "portfolio_value": self.portfolio_value,
            "reward": self.last_reward,
            "profit_ratio": self.current_profit / self.portfolio_value,
        }

    def close(self):
        pass
