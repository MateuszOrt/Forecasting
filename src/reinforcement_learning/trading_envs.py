import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DiscreteTradingEnv(gym.Env):
    """Proste środowisko handlu"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, prices, initial_cash=100):
        super(DiscreteTradingEnv, self).__init__()

        self.prices = prices
        self.max_steps = len(prices)
        self.initial_cash = initial_cash

        # Akcje: 0 - nic, 1 - kup, 2 - sprzedaj
        self.action_space = spaces.Discrete(3)

        # Obserwacja: [cena, gotówka, liczba akcji]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(4,), dtype=np.float32
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
        mean_price = np.mean(self.prices[: self.step_count + 1])
        return np.array(
            [self.current_price, self.cash, self.shares, mean_price], dtype=np.float32
        )

    @property
    def portfolio_value(self):
        return self.cash + self.shares * self.current_price

    def step(self, action):
        done = False
        reward = 0
        mean_price = np.mean(
            self.prices[: self.step_count + 1]
        )  # Średnia cena dotychczas

        if action == 1:  # Kup
            if self.cash >= self.current_price:
                self.cash -= self.current_price
                self.shares += 1
            if self.current_price < mean_price:
                reward += 0.5  # Nagroda za kupno poniżej średniej

        elif action == 2:  # Sprzedaj
            if self.shares > 0:
                self.cash += self.current_price
                self.shares -= 1
            if self.current_price > mean_price:
                reward += 0.5  # Nagroda za sprzedaż powyżej średniej

        self.actions_history.append(action)

        self.step_count += 1

        if self.step_count >= self.max_steps:
            done = True

        return self.get_observation(), reward, done, False, {}

    def render(self):
        log_string = (
            f"Krok: {self.step_count}|"
            f"Cena: {self.current_price:.2f}|"
            f"Gotówka: {self.cash:.2f}|"
            f"Akcje: {self.shares}|"
            f"Wartość portfela: {self.portfolio_value:.2f}|"
        )
        print(log_string)
        return {
            "step": self.step_count,
            "price": self.current_price,
            "cash": self.cash,
            "shares": self.shares,
            "portfolio_value": self.portfolio_value,
        }

    def close(self):
        pass


class ContinuousTradingEnv(gym.Env):
    """Proste środowisko handlu z ciągłą przestrzenią obserwacji"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, prices, initial_cash=100):
        super(ContinuousTradingEnv, self).__init__()

        self.prices = prices
        self.max_steps = len(prices)
        self.initial_cash = initial_cash

        # Akcje: 1 - kup, (-1) - sprzedaj
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Obserwacja: [cena, gotówka, liczba akcji]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(4,), dtype=np.float32
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
        mean_price = np.mean(self.prices[: self.step_count + 1])
        return np.array(
            [self.current_price, self.cash, self.shares, mean_price], dtype=np.float32
        )

    @property
    def portfolio_value(self):
        return self.cash + self.shares * self.current_price

    def step(self, action):
        done = False
        reward = 0
        mean_price = np.mean(
            self.prices[: self.step_count + 1]
        )  # Średnia cena dotychczas
        if action > 0:  # Kup
            # Ze względu na typ danych operacji  należy najpierw sprawdzić ile akcji fizycznie można kupić
            max_affordable = int(
                self.cash / (self.current_price + self.current_price * 0.01)
            )
            shares_to_buy = int(action * max_affordable)
            if shares_to_buy > 0:
                # Aktualizacja ilości akcji
                new_total_shares = self.shares + shares_to_buy
                self.shares = new_total_shares
                # Aktualizacja ilości gotówki
                cost_of_current_purchase = shares_to_buy * self.current_price
                self.cash -= cost_of_current_purchase
            if self.current_price < mean_price:
                reward += 0.5  # Nagroda za kupno poniżej średniej

        elif action < 0:  # Sprzedaj
            # Sprawdzenie ile akcji można sprzedać
            shares_to_sell = int(-action * self.shares)
            if shares_to_sell > 0:
                # Aktualizacja ilości akcji
                self.shares -= shares_to_sell
                # Aktualizacja ilości gotówki
                self.cash += shares_to_sell * self.current_price
            if self.current_price > mean_price:
                reward += 0.5  # Nagroda za sprzedaż powyżej średniej

        self.actions_history.append(action)

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        self.last_reward = reward
        return self.get_observation(), reward, done, False, {}

    def render(self):
        log_string = (
            f"Krok: {self.step_count}|"
            f"Cena: {self.current_price:.2f}|"
            f"Gotówka: {self.cash:.2f}|"
            f"Akcje: {self.shares}|"
            f"Wartość portfela: {self.portfolio_value:.2f}|"
        )
        print(log_string)
        return {
            "step": self.step_count,
            "price": self.current_price,
            "cash": self.cash,
            "shares": self.shares,
            "portfolio_value": self.portfolio_value,
        }

    def close(self):
        pass


class PriceDiffTradingEnv(gym.Env):
    """Proste środowisko handlu z ciągłą przestrzenią obserwacji
    opierające funkcję nagrody na różnicy cen"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, prices, initial_cash=100):
        super(PriceDiffTradingEnv, self).__init__()

        self.prices = prices
        self.max_steps = len(prices)
        self.initial_cash = initial_cash

        # Akcje: 1 - kup, (-1) - sprzedaj
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Obserwacja: [cena, gotówka, liczba akcji]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.actions_history = []
        self.reset()

    @property
    def current_price(self):
        return self.prices[min(self.step_count, self.max_steps - 1)]

    @property
    def previous_price(self):
        if self.prices[min(self.step_count - 1, self.max_steps - 2)]:
            return self.prices[min(self.step_count - 1, self.max_steps - 2)]
        else:
            return self.prices[min(self.step_count, self.max_steps - 1)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.cash = self.initial_cash
        self.shares = 0

        self.actions_history = []
        return self.get_observation(), {}

    def get_observation(self):
        price_difference = (self.current_price / self.previous_price) - 1
        return np.array(
            [self.current_price, self.cash, self.shares, price_difference],
            dtype=np.float32,
        )

    @property
    def portfolio_value(self):
        return self.cash + self.shares * self.current_price

    def step(self, action):
        done = False
        reward = 0
        price_difference = (
            self.current_price / self.previous_price
        ) - 1  # różnica w cenie (czy cena rośnie czy maleje)
        if action > 0:  # Kup
            # Ze względu na typ danych operacji  należy najpierw sprawdzić ile akcji fizycznie można kupić
            max_affordable = int(
                self.cash / (self.current_price + self.current_price * 0.01)
            )
            shares_to_buy = int(action * max_affordable)
            if shares_to_buy > 0:
                # Aktualizacja ilości akcji
                new_total_shares = self.shares + shares_to_buy
                self.shares = new_total_shares
                # Aktualizacja ilości gotówki
                cost_of_current_purchase = shares_to_buy * self.current_price
                self.cash -= cost_of_current_purchase
            if price_difference > 0:
                reward += 0.5  # Nagroda za kupno gdy cena rośnie

        elif action < 0:  # Sprzedaj
            # Sprawdzenie ile akcji można sprzedać
            shares_to_sell = int(-action * self.shares)
            if shares_to_sell > 0:
                # Aktualizacja ilości akcji
                self.shares -= shares_to_sell
                # Aktualizacja ilości gotówki
                self.cash += shares_to_sell * self.current_price
            if price_difference < 0:
                reward += 0.5  # Nagroda za sprzedaż gdy cena spada

        self.actions_history.append(action)

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        self.last_reward = reward
        return self.get_observation(), reward, done, False, {}

    def render(self):
        log_string = (
            f"Krok: {self.step_count}|"
            f"Cena: {self.current_price:.2f}|"
            f"Gotówka: {self.cash:.2f}|"
            f"Akcje: {self.shares}|"
            f"Wartość portfela: {self.portfolio_value:.2f}|"
            f"Nagroda: {self.last_reward:.2f}|"
        )
        print(log_string)
        return {
            "step": self.step_count,
            "price": self.current_price,
            "cash": self.cash,
            "shares": self.shares,
            "portfolio_value": self.portfolio_value,
            "nagroda": self.last_reward,
        }

    def close(self):
        pass


class PriceDiffTradingEnvUpdated(gym.Env):
    """Proste środowisko handlu z ciągłą przestrzenią obserwacji
    opierające funkcję nagrody na różnicy cen, w stosunku do poprzedniego środowiska
    wprowadzona zostaje zmienna trend w get_obserwation"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, prices, initial_cash=100):
        super(PriceDiffTradingEnvUpdated, self).__init__()

        self.prices = prices
        self.max_steps = len(prices)
        self.initial_cash = initial_cash

        # Akcje: 1 - kup, (-1) - sprzedaj
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Obserwacja: [cena, gotówka, liczba akcji]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        self.actions_history = []
        self.reset()

    @property
    def current_price(self):
        return self.prices[min(self.step_count, self.max_steps - 1)]

    @property
    def previous_price(self):
        if self.prices[min(self.step_count - 1, self.max_steps - 2)]:
            return self.prices[min(self.step_count - 1, self.max_steps - 2)]
        else:
            return self.prices[min(self.step_count, self.max_steps - 1)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.cash = self.initial_cash
        self.shares = 0

        self.actions_history = []
        return self.get_observation(), {}

    def get_observation(self):
        price_difference = (self.current_price / self.previous_price) - 1
        if price_difference > 0:
            trend = 10
        elif price_difference < 0:
            trend = -10
        return np.array(
            [self.current_price, self.cash, self.shares, price_difference, trend],
            dtype=np.float32,
        )

    @property
    def portfolio_value(self):
        return self.cash + self.shares * self.current_price

    def step(self, action):
        done = False
        reward = 0
        price_difference = (
            self.current_price / self.previous_price
        ) - 1  # różnica w cenie (czy cena rośnie czy maleje)
        if action > 0:  # Kup
            # Ze względu na typ danych operacji  należy najpierw sprawdzić ile akcji fizycznie można kupić
            max_affordable = int(
                self.cash / (self.current_price + self.current_price * 0.01)
            )
            shares_to_buy = int(action * max_affordable)
            if shares_to_buy > 0:
                # Aktualizacja ilości akcji
                new_total_shares = self.shares + shares_to_buy
                self.shares = new_total_shares
                # Aktualizacja ilości gotówki
                cost_of_current_purchase = shares_to_buy * self.current_price
                self.cash -= cost_of_current_purchase
            if price_difference > 0:
                reward += 0.5  # Nagroda za kupno gdy cena rośnie

        elif action < 0:  # Sprzedaj
            # Sprawdzenie ile akcji można sprzedać
            shares_to_sell = int(-action * self.shares)
            if shares_to_sell > 0:
                # Aktualizacja ilości akcji
                self.shares -= shares_to_sell
                # Aktualizacja ilości gotówki
                self.cash += shares_to_sell * self.current_price
            if price_difference < 0:
                reward += 0.5  # Nagroda za sprzedaż gdy cena spada

        self.actions_history.append(action)

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        self.last_reward = reward
        return self.get_observation(), reward, done, False, {}

    def render(self):
        log_string = (
            f"Krok: {self.step_count}|"
            f"Cena: {self.current_price:.2f}|"
            f"Gotówka: {self.cash:.2f}|"
            f"Akcje: {self.shares}|"
            f"Wartość portfela: {self.portfolio_value:.2f}|"
            f"Nagroda: {self.last_reward:.2f}|"
        )
        print(log_string)
        return {
            "step": self.step_count,
            "price": self.current_price,
            "cash": self.cash,
            "shares": self.shares,
            "portfolio_value": self.portfolio_value,
            "nagroda": self.last_reward,
        }

    def close(self):
        pass


class PortfolioValueTradingEnv(gym.Env):
    """Proste środowisko handlu z ciągłą przestrzenią obserwacji
    opierające funkcję nagrody na różnicy wartości portfolio"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, prices, initial_cash=100):
        super(PortfolioValueTradingEnv, self).__init__()

        self.prices = prices
        self.max_steps = len(prices)
        self.initial_cash = initial_cash

        # Akcje: 1 - kup, (-1) - sprzedaj
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Obserwacja: [cena, gotówka, liczba akcji]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.actions_history = []
        self.reset()

    @property
    def current_price(self):
        return self.prices[min(self.step_count, self.max_steps - 1)]

    @property
    def previous_price(self):
        if self.prices[min(self.step_count - 1, self.max_steps - 2)]:
            return self.prices[min(self.step_count - 1, self.max_steps - 2)]
        else:
            return self.prices[min(self.step_count, self.max_steps - 1)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.cash = self.initial_cash
        self.shares = 0

        self.actions_history = []
        return self.get_observation(), {}

    def get_observation(self):
        portfolio_value_difference = (
            self.portfolio_value - self.previous_portfolio_value
        ) * 10
        price_difference = (self.current_price / self.previous_price) - 1
        if price_difference > 0:
            trend = 10
        elif price_difference < 0:
            trend = -10

        return np.array(
            [
                self.current_price,
                self.cash,
                self.shares,
                self.portfolio_value,
                self.previous_portfolio_value,
                price_difference,
                portfolio_value_difference,
                trend,
            ],
            dtype=np.float32,
        )

    @property
    def previous_portfolio_value(self):
        return self.cash + self.shares * self.previous_price

    @property
    def portfolio_value(self):
        return self.cash + self.shares * self.current_price

    def step(self, action):
        done = False
        reward = 0
        if action > 0:  # Kup
            # Ze względu na typ danych operacji  należy najpierw sprawdzić ile akcji fizycznie można kupić
            max_affordable = int(
                self.cash / (self.current_price + self.current_price * 0.01)
            )
            shares_to_buy = int(action * max_affordable)
            if shares_to_buy > 0:
                # Aktualizacja ilości akcji
                new_total_shares = self.shares + shares_to_buy
                self.shares = new_total_shares
                # Aktualizacja ilości gotówki
                cost_of_current_purchase = shares_to_buy * self.current_price
                self.cash -= cost_of_current_purchase

        elif action < 0:  # Sprzedaj
            # Sprawdzenie ile akcji można sprzedać
            shares_to_sell = int(-action * self.shares)
            if shares_to_sell > 0:
                # Aktualizacja ilości akcji
                self.shares -= shares_to_sell
                # Aktualizacja ilości gotówki
                self.cash += shares_to_sell * self.current_price

        reward = self.portfolio_value - self.previous_portfolio_value

        price_difference = (self.current_price / self.previous_price) - 1
        if price_difference > 0 and reward == 0:
            reward = -100

        self.actions_history.append(action)

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        self.last_reward = reward
        return self.get_observation(), reward, done, False, {}

    def render(self):
        log_string = (
            f"Krok: {self.step_count}|"
            f"Cena: {self.current_price:.2f}|"
            f"Gotówka: {self.cash:.2f}|"
            f"Akcje: {self.shares}|"
            f"Wartość portfela: {self.portfolio_value:.2f}|"
            f"Nagroda: {self.last_reward:.2f}|"
        )
        print(log_string)
        return {
            "step": self.step_count,
            "price": self.current_price,
            "cash": self.cash,
            "shares": self.shares,
            "portfolio_value": self.portfolio_value,
            "nagroda": self.last_reward,
        }

    def close(self):
        pass


class PortfolioValueTradingEnvUpdated(gym.Env):
    """Proste środowisko handlu z ciągłą przestrzenią obserwacji
    opierające funkcję nagrody na różnicy wartości portfolio"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, prices, initial_cash=100):
        super(PortfolioValueTradingEnvUpdated, self).__init__()

        self.prices = prices
        self.max_steps = len(prices)
        self.initial_cash = initial_cash

        # Akcje: 1 - kup, (-1) - sprzedaj
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Obserwacja: [cena, gotówka, liczba akcji]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.actions_history = []
        self.reset()

    @property
    def current_price(self):
        return self.prices[min(self.step_count, self.max_steps - 1)]

    @property
    def previous_price(self):
        if self.prices[min(self.step_count - 1, self.max_steps - 2)]:
            return self.prices[min(self.step_count - 1, self.max_steps - 2)]
        else:
            return self.prices[min(self.step_count, self.max_steps - 1)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.cash = self.initial_cash
        self.shares = 0

        self.actions_history = []
        return self.get_observation(), {}

    def get_observation(self):
        portfolio_value_difference = (
            self.portfolio_value - self.previous_portfolio_value
        ) * 10
        price_difference = (self.current_price / self.previous_price) - 1
        if price_difference > 0:
            trend = 10
        elif price_difference < 0:
            trend = -10

        return np.array(
            [
                self.current_price,
                self.cash,
                self.shares,
                self.portfolio_value,
                self.previous_portfolio_value,
                price_difference,
                portfolio_value_difference,
                trend,
            ],
            dtype=np.float32,
        )

    @property
    def previous_portfolio_value(self):
        return self.cash + self.shares * self.previous_price

    @property
    def portfolio_value(self):
        return self.cash + self.shares * self.current_price

    def step(self, action):
        done = False
        reward = 0
        if action > 0:  # Kup
            # Ze względu na typ danych operacji  należy najpierw sprawdzić ile akcji fizycznie można kupić
            max_affordable = int(
                self.cash / (self.current_price + self.current_price * 0.01)
            )
            shares_to_buy = int(action * max_affordable)
            if shares_to_buy > 0:
                # Aktualizacja ilości akcji
                new_total_shares = self.shares + shares_to_buy
                self.shares = new_total_shares
                # Aktualizacja ilości gotówki
                cost_of_current_purchase = shares_to_buy * self.current_price
                self.cash -= cost_of_current_purchase

        elif action < 0:  # Sprzedaj
            # Sprawdzenie ile akcji można sprzedać
            shares_to_sell = int(-action * self.shares)
            if shares_to_sell > 0:
                # Aktualizacja ilości akcji
                self.shares -= shares_to_sell
                # Aktualizacja ilości gotówki
                self.cash += shares_to_sell * self.current_price

        reward = self.portfolio_value - self.previous_portfolio_value

        price_difference = (self.current_price / self.previous_price) - 1
        if price_difference > 0 and reward == 0:
            reward = -10000

        self.actions_history.append(action)

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        self.last_reward = reward
        return self.get_observation(), reward, done, False, {}

    def render(self):
        log_string = (
            f"Krok: {self.step_count}|"
            f"Cena: {self.current_price:.2f}|"
            f"Gotówka: {self.cash:.2f}|"
            f"Akcje: {self.shares}|"
            f"Wartość portfela: {self.portfolio_value:.2f}|"
            f"Nagroda: {self.last_reward:.2f}|"
        )
        print(log_string)
        return {
            "step": self.step_count,
            "price": self.current_price,
            "cash": self.cash,
            "shares": self.shares,
            "portfolio_value": self.portfolio_value,
            "nagroda": self.last_reward,
        }

    def close(self):
        pass


class ProfitTradingEnv(gym.Env):
    """Środowisko handlu z zakodowanym profitem"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, prices, initial_cash=100):
        super(ProfitTradingEnv, self).__init__()

        self.prices = prices
        self.max_steps = len(prices)
        self.initial_cash = initial_cash

        # Akcje: 0 - nic, 1 - kup, 2 - sprzedaj
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Obserwacja: [cena, gotówka, liczba akcji]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        self.actions_history = []
        self.reset()

    @property
    def current_price(self):
        return self.prices[min(self.step_count, self.max_steps - 1)]

    @property
    def previous_price(self):
        if self.prices[min(self.step_count - 1, self.max_steps - 2)]:
            return self.prices[min(self.step_count - 1, self.max_steps - 2)]
        else:
            return self.prices[min(self.step_count, self.max_steps - 1)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.cash = self.initial_cash
        self.shares = 20
        self.cost_basis_shares = 100
        self.average_cost_per_share = 5
        self.actions_history = []
        return self.get_observation(), {}

    def get_observation(self):
        profit_difference = self.previous_profit - self.current_profit
        price_difference = (self.current_price / self.previous_price) - 1

        if price_difference > 0:
            price_trend = 10
        elif price_difference < 0:
            price_trend = -10
        elif price_difference == 0:
            price_trend = 0

        if profit_difference > 0:
            profit_change = 10
        elif profit_difference < 0:
            profit_change = -10
        elif profit_difference == 0:
            profit_change = 0

        # if price_difference > 0 and profit_difference < 0:
        #     profit_trend=10

        # elif price_difference < 0 and profit_difference > 0:
        #     profit_trend=10

        return np.array(
            [
                self.current_price,
                self.cash,
                self.shares,
                profit_change,
                price_trend,
            ],
            dtype=np.float32,
        )

    @property
    def portfolio_value(self):
        return self.cash + self.shares * self.current_price

    @property
    def previous_portfolio_value(self):
        return self.cash + self.shares * self.previous_price

    @property
    def current_profit(self):
        market_value = self.shares * self.current_price
        invested_value = self.cost_basis_shares
        profit = market_value - invested_value
        return profit

    @property
    def previous_profit(self):
        market_value = self.shares * self.previous_price
        invested_value = self.cost_basis_shares
        profit = market_value - invested_value
        return profit

    def step(self, action):
        done = False
        reward = 0
        # previous_price=self.prices[min(self.step_count-1, self.max_steps - 2)]
        previous_profit = self.previous_profit
        previous_shares = self.shares

        if action > 0:  # Kup
            max_affordable = int(
                self.cash / (self.current_price + self.current_price * 0.01)
            )
            shares_to_buy = int(action * max_affordable)
            if shares_to_buy > 0:
                # Aktualizacja ksztów zakupionych akcji
                new_total_shares = self.shares + shares_to_buy
                total_cost_so_far = self.average_cost_per_share * self.shares
                cost_of_current_purchase = shares_to_buy * self.current_price

                # Opłata transakcyjna 1 %
                cost_of_current_purchase += 0.01 * cost_of_current_purchase

                self.average_cost_per_share = (
                    total_cost_so_far + cost_of_current_purchase
                ) / new_total_shares
                self.cost_basis_shares = self.average_cost_per_share * new_total_shares

                # Update cash and shares
                self.shares = new_total_shares
                self.cash -= cost_of_current_purchase

        elif action < 0:  # Sprzedaj
            shares_to_sell = int(-action * self.shares)
            if shares_to_sell > 0:
                self.shares -= shares_to_sell
                if self.shares == 0:
                    self.average_cost_per_share = 0
                    self.cost_basis_shares = 0
                else:
                    self.cost_basis_shares = self.average_cost_per_share * self.shares
                self.cash += shares_to_sell * self.current_price

        price_difference = (self.current_price / self.previous_price) - 1
        # profit_difference = self.previous_profit - self.current_profit

        profit_difference = previous_profit - self.current_profit

        # Nagroda zwiększanie profitu
        if previous_shares < self.shares:
            reward -= profit_difference
            if price_difference < 0 and reward > 0:
                reward *= 0.01

        # Nagroda za zrealizowany profit
        elif previous_shares > self.shares:
            reward += profit_difference
            if price_difference > 0 and reward > 0:
                reward *= 0.01

        elif previous_shares == self.shares:
            if price_difference < 0 and self.cash < self.current_price:
                reward = -1000
            if price_difference > 0 and self.cash > self.current_price:
                reward = -1000

        self.actions_history.append(action)

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        self.last_reward = reward
        self.last_profit_difference = profit_difference
        return self.get_observation(), reward, done, False, {}

    def render(self):
        log_string = (
            f"Krok: {self.step_count}|"
            f"Cena: {self.current_price:.2f}|"
            f"Gotówka: {self.cash:.2f}|"
            f"Akcje: {self.shares}|"
            f"Wartość portfela: {self.portfolio_value:.2f}|"
            f"Reward: {self.last_reward:.4f}|"
            f"Profit: {self.current_profit:.4f}|"
        )
        print(log_string)
        return {
            "step": self.step_count,
            "price": self.current_price,
            "cash": self.cash,
            "shares": self.shares,
            "portfolio_value": self.portfolio_value,
            "reward": self.last_reward,
            "profit_ratio": self.current_profit,
        }

    def close(self):
        pass
