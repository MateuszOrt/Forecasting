# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np


# class TradingEnv(gym.Env):
#     """Proste środowisko handlu"""

#     metadata = {"render_modes": ["human"]}

#     def __init__(self, prices, initial_cash=100):
#         super(TradingEnv, self).__init__()

#         self.prices = prices
#         self.max_steps = len(prices)
#         self.initial_cash = initial_cash

#         # Akcje: 0 - nic, 1 - kup, 2 - sprzedaj
#         self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
#         # Obserwacja: [cena, gotówka, liczba akcji]
#         self.observation_space = spaces.Box(
#             low=0, high=np.inf, shape=(3,), dtype=np.float32
#         )
#         self.actions_history = []
#         self.reset()

#     @property
#     def current_price(self):
#         return self.prices[min(self.step_count, self.max_steps - 1)]

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.step_count = 0
#         self.cash = self.initial_cash
#         self.shares = 0

#         self.actions_history = []
#         return self.get_observation(), {}

#     def get_observation(self):
#         return np.array([self.current_price, self.cash, self.shares], dtype=np.float32)

#     @property
#     def portfolio_value(self):
#         return self.cash + self.shares * self.current_price

#     def step(self, action):
#         done = False
#         reward = 0
#         # previous_portfolio_value = self._get_portfolio_value()
#         mean_price = np.mean(
#             self.prices[: self.step_count + 1]
#         )  # Średnia cena dotychczas
#         # reward = self._get_portfolio_value() - self.initial_cash
#         if action > 0:  # Kup
#             max_affordable = int(
#                 self.cash / (self.current_price + self.current_price * 0.01)
#             )
#             shares_to_buy = int(action * max_affordable)
#             if shares_to_buy > 0:
#                 # Update average cost per share
#                 new_total_shares = self.shares + shares_to_buy
#                 # total_cost_so_far = self.average_cost_per_share * self.shares
#                 cost_of_current_purchase = shares_to_buy * self.current_price

#                 # Opłata transakcyjna 1 %
#                 # cost_of_current_purchase += 0.01 * cost_of_current_purchase

#                 # self.average_cost_per_share = (
#                 #     total_cost_so_far + cost_of_current_purchase
#                 # ) / new_total_shares
#                 # self.cost_basis_shares = self.average_cost_per_share * new_total_shares

#                 # Update cash and shares
#                 self.shares = new_total_shares
#                 self.cash -= cost_of_current_purchase

#         elif action < 0:  # Sprzedaj
#             shares_to_sell = int(-action * self.shares)
#             # if shares_to_sell == 0 and self.shares > 0 and action_value < 0:
#             #     shares_to_sell = 1
#             if shares_to_sell > 0:
#                 # self.cash += shares_to_sell * effective_price
#                 self.shares -= shares_to_sell
#                 # if self.shares == 0:
#                 #     self.average_cost_per_share = 0
#                 #     self.cost_basis_shares = 0
#                 # else:
#                 #     self.cost_basis_shares = self.average_cost_per_share * self.shares
#                 self.cash += shares_to_sell * self.current_price
#                 # transaction_profit_loss = (effective_price - self.average_cost_per_share) * shares_to_sell
#                 # self.total_profit_loss += transaction_profit_loss

#         if self.current_price < mean_price and action > 0:
#             reward += 0.5

#         if self.current_price > mean_price and action < 0:
#             reward += 0.5

#         self.actions_history.append(action)

#         # reward = (self._get_portfolio_value() - self.initial_cash)/self.initial_cash
#         # reward = self._get_portfolio_value() - self.initial_cash

#         self.step_count += 1
#         if self.step_count >= self.max_steps:
#             done = True
#         self.last_reward = reward
#         return self.get_observation(), reward, done, False, {}

#     def render(self):
#         log_string = (
#             f"Krok: {self.step_count}|"
#             f"Cena: {self.current_price:.2f}|"
#             f"Gotówka: {self.cash:.2f}|"
#             f"Akcje: {self.shares}|"
#             f"Wartość portfela: {self.portfolio_value:.2f}|"
#             f"Reward: {self.last_reward:.4f}|"
#         )
#         print(log_string)

#     def close(self):
#         pass


# class NewTradingEnv(gym.Env):
#     """Proste środowisko handlu"""

#     metadata = {"render_modes": ["human"]}

#     def __init__(self, prices, initial_cash=100):
#         super(NewTradingEnv, self).__init__()

#         self.prices = prices
#         self.max_steps = len(prices)
#         self.initial_cash = initial_cash

#         # Akcje: 0 - nic, 1 - kup, 2 - sprzedaj
#         self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
#         # Obserwacja: [cena, gotówka, liczba akcji]
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
#         )
#         self.actions_history = []
#         self.reset()

#     @property
#     def current_price(self):
#         return self.prices[min(self.step_count, self.max_steps - 1)]

#     @property
#     def previous_price(self):
#         if self.prices[min(self.step_count - 1, self.max_steps - 2)]:
#             return self.prices[min(self.step_count - 1, self.max_steps - 2)]
#         else:
#             return self.prices[min(self.step_count, self.max_steps - 1)]

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.step_count = 0
#         self.cash = self.initial_cash
#         self.shares = 20
#         self.cost_basis_shares = 100
#         self.average_cost_per_share = 5
#         self.actions_history = []
#         return self.get_observation(), {}

#     def get_observation(self):
#         """
#         Zwraca obserwację: [cena, gotówka, akcje, short_ma, long_ma, trend]
#         """
#         short_window = 5
#         long_window = 20
#         trend_window = 5
#         if self.step_count >= short_window:
#             short_ma = np.mean(
#                 self.prices[self.step_count - (short_window - 1) : self.step_count + 1]
#             )
#         else:
#             short_ma = self.prices[0]  # Można użyć ceny początkowej lub 0

#         if self.step_count >= long_window:
#             long_ma = np.mean(
#                 self.prices[self.step_count - (long_window - 1) : self.step_count + 1]
#             )
#         else:
#             long_ma = self.prices[0]  # Można użyć ceny początkowej lub 0

#         # Oblicz trend (prosty przykład: wzrost/spadek)
#         x = np.array([0, 1, 2, 3, 4])
#         if self.step_count >= trend_window - 1:
#             y = self.prices[self.step_count - (trend_window - 1) : self.step_count + 1]
#             x = np.arange(len(y))  # dopasowanie długości
#             if len(x) == len(y):  # zabezpieczenie na wszelki wypadek
#                 a, b = np.polyfit(x, y, deg=1)
#                 trend = a
#             else:
#                 trend = 0
#         else:
#             trend = 0

#         profit_ratio = self.current_profit / self.portfolio_value
#         price_diff = 1 - (self.current_price / self.prices[self.step_count - 1])

#         return np.array(
#             [
#                 self.current_price,
#                 self.cash,
#                 self.shares,
#                 short_ma,
#                 long_ma,
#                 trend,
#                 profit_ratio,
#                 price_diff,
#             ],
#             dtype=np.float32,
#         )

#     @property
#     def portfolio_value(self):
#         return self.cash + self.shares * self.current_price

#     @property
#     def previous_portfolio_value(self):
#         return self.cash + self.shares * self.previous_price

#     @property
#     def current_profit(self):
#         market_value = self.shares * self.current_price
#         invested_value = self.cost_basis_shares
#         profit = market_value - invested_value
#         return profit

#     @property
#     def previous_profit(self):
#         market_value = self.shares * self.previous_price
#         invested_value = self.cost_basis_shares
#         profit = market_value - invested_value
#         return profit

#     def step(self, action):
#         done = False
#         reward = 0
#         # previous_price=self.prices[min(self.step_count-1, self.max_steps - 2)]
#         # previous_profit = self.current_profit(price=previous_price)
#         previous_shares = self.shares
#         mean_price = np.mean(self.prices[: self.step_count + 1])
#         if action > 0:  # Kup
#             max_affordable = int(
#                 self.cash / (self.current_price + self.current_price * 0.01)
#             )
#             shares_to_buy = int(action * max_affordable)
#             if shares_to_buy > 0:
#                 # Update average cost per share
#                 new_total_shares = self.shares + shares_to_buy
#                 total_cost_so_far = self.average_cost_per_share * self.shares
#                 cost_of_current_purchase = shares_to_buy * self.current_price

#                 # Opłata transakcyjna 1 %
#                 cost_of_current_purchase += 0.01 * cost_of_current_purchase

#                 self.average_cost_per_share = (
#                     total_cost_so_far + cost_of_current_purchase
#                 ) / new_total_shares
#                 self.cost_basis_shares = self.average_cost_per_share * new_total_shares

#                 # Update cash and shares
#                 self.shares = new_total_shares
#                 self.cash -= cost_of_current_purchase

#         elif action < 0:  # Sprzedaj
#             shares_to_sell = int(-action * self.shares)
#             if shares_to_sell > 0:
#                 self.shares -= shares_to_sell
#                 if self.shares == 0:
#                     self.average_cost_per_share = 0
#                     self.cost_basis_shares = 0
#                 else:
#                     self.cost_basis_shares = self.average_cost_per_share * self.shares
#                 self.cash += shares_to_sell * self.current_price

#         price_difference = (self.current_price / self.previous_price) - 1
#         # profit_difference = self.previous_profit - self.current_profit

#         if price_difference > 0 and action > 0:
#             reward += 0.5

#         if price_difference < 0 and action < 0:
#             reward += 0.5

#         # if self.current_price < mean_price and action > 0:
#         #             reward +=0.5

#         # if self.current_price > mean_price and action < 0:
#         #             reward +=0.5

#         #             if previous_shares > self.shares:
#         #                 reward += profit_difference
#         # elif price_difference > 0 and action < 0:
#         #      reward = -10

#         # reward = self.portfolio_value - self.previous_portfolio_value

#         # profit_difference = self.previous_profit - self.current_profit

#         # if price_difference > 0 and action > 0:
#         #     reward +=5
#         # if price_difference < 0 and action < 0:
#         #     reward +=5

#         # reward -= price_difference * self.cash
#         # Nagroda zwiększanie profitu
#         # if previous_shares < self.shares:
#         #     reward -= profit_difference
#         #     if price_difference < 0 and reward > 0:
#         #         reward *= -0.1
#         # if price_difference > 0 and self.cash > self.current_price:
#         #     reward =-100
#         # reward -= price_difference * self.cash # Potencjalny niezrealizowany przychód

#         # Nagroda za zrealizowany profit
#         # elif previous_shares > self.shares:
#         #     reward += profit_difference
#         #     if price_difference > 0 and reward > 0:
#         #         reward *= -0.1

#         # elif previous_shares == self.shares:
#         #     if price_difference > 0:
#         #         reward -= profit_difference
#         # if self.cash > self.current_price:
#         #     reward = -10
#         # else:
#         #     reward -= profit_difference
#         # if price_difference < 0:
#         #     reward -= price_difference * self.cash
#         # if self.cash < self.current_price:
#         #     reward = -10
#         # else:
#         # reward -= price_difference * self.cash
#         # elif price_difference < 0 and self.cash > self.current_price:
#         #     reward = -10

#         self.actions_history.append(action)

#         self.step_count += 1
#         if self.step_count >= self.max_steps:
#             done = True
#         self.last_reward = reward
#         return self.get_observation(), reward, done, False, {}

#     def render(self):
#         log_string = (
#             f"Krok: {self.step_count}|"
#             f"Cena: {self.current_price:.2f}|"
#             f"Gotówka: {self.cash:.2f}|"
#             f"Akcje: {self.shares}|"
#             f"Wartość portfela: {self.portfolio_value:.2f}|"
#             f"Reward: {self.last_reward:.4f}|"
#             f"Profit: {self.current_profit:.4f}|"
#         )
#         print(log_string)
#         return {
#             "step": self.step_count,
#             "price": self.current_price,
#             "cash": self.cash,
#             "shares": self.shares,
#             "portfolio_value": self.portfolio_value,
#             "reward": self.last_reward,
#             "profit_ratio": self.current_profit,
#         }

#     def close(self):
#         pass
