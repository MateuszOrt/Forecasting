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
            elif price_difference < 0:
                reward -= 1  # Kara za kupno gdy cena spada

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
            elif price_difference > 0:
                reward -= 1  # Kara za kupno gdy cena rośnie

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
            elif price_difference < 0:
                reward -= 1  # Kara za kupno gdy cena spada

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
            elif price_difference > 0:
                reward -= 1  # Kara za kupno gdy cena rośnie

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
                reward *= -0.1
            # if price_difference > 0 and self.cash > self.current_price:
            #     reward =-100
            # reward -= price_difference * self.cash # Potencjalny niezrealizowany przychód

        # Nagroda za zrealizowany profit
        elif previous_shares > self.shares:
            reward += profit_difference
            if price_difference > 0 and reward > 0:
                reward *= -0.1

        elif previous_shares == self.shares:
            if price_difference < 0 and self.cash < self.current_price:
                reward = -1000
            if price_difference > 0 and self.cash > self.current_price:
                reward = -1000

        # elif previous_shares == self.shares:
        #     if price_difference > 0 and self.cash > self.current_price:
        #         reward -= price_difference * self.cash
        #     if price_difference > 0:
        #         reward -= profit_difference
        #         # if self.cash > self.current_price:
        #         #     reward = -10
        #         # else:
        #     #     reward -= profit_difference
        #     if price_difference < 0 and self.cash > self.current_price:
        #         reward -= price_difference * self.cash
        #         # if self.cash < self.current_price:
        #         #     reward = -10
        #         # else:
        #         # reward -= price_difference * self.cash
        #     # elif price_difference < 0 and self.cash > self.current_price:
        #     #     reward = -10

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
        self.profit_change = 0
        self.trend = 0
        # self.total_profit_loss = 0
        self.actions_history = []
        return self.get_observation(), {}

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

        return self.get_observation(), reward, done, False, {}

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


class AdvancedTradingEnvNew(gym.Env):
    """Zaawansowane środowisko handlu"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self, prices, initial_cash=100, short_window=5, long_window=20, trend_window=5
    ):
        super(AdvancedTradingEnvNew, self).__init__()
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
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
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
        self.shares = 20  # Ustalamy liczbe akcji na start
        self.average_cost_per_share = 5  # Ustalamy liczbe akcji na start
        self.cost_basis_shares = 100  # Ustalamy koszt na start
        self.profit_change = 0
        self.trend = 0
        # self.total_profit_loss = 0
        self.actions_history = []
        return self.get_observation(), {}

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
        price_diff = 1 - (self.current_price / self.prices[self.step_count - 1])

        return np.array(
            [
                self.current_price,
                self.cash,
                self.shares,
                short_ma,
                long_ma,
                self.trend,
                profit_ratio,
                price_diff,
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

        # previous_portfolio_value = self.portfolio_value
        previous_profit = self.current_profit
        previous_price = self.current_price
        previous_shares = self.shares
        if action > 0:  # Kup
            max_affordable = int(
                self.cash / (self.current_price + self.current_price * 0.01)
            )
            shares_to_buy = int(action * max_affordable)
            if shares_to_buy > 0:
                # Update average cost per share
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

        profit_difference = previous_profit - self.current_profit
        price_difference = (self.current_price / previous_price) - 1

        # if previous_profit < 0 and self.current_profit < 0:
        #     profit_difference = -profit_difference

        # Nagroda zwiększanie profitu
        if previous_shares <= self.shares:
            reward -= profit_difference
            if price_difference < 0 and reward > 0:
                reward *= 0.1
            # if price_difference > 0 and self.cash > self.current_price:
            #     # reward =-100
            #     reward -= price_difference * self.cash # Potencjalny niezrealizowany przychód

            # if price_difference < 0 and self.shares > 0: # to był warunek po to żeby też sprzedawał
            #     reward += price_difference * self.shares
            # if price_difference < 0 and self.cash < self.current_price: # to był warunek po to żeby też sprzedawał
            #     reward =-100

        # Nagroda za zrealizowany profit
        if previous_shares > self.shares:
            reward += profit_difference
            if price_difference > 0 and reward > 0:
                reward *= 0.1
            # if price_difference < 0 and self.shares > 0: # to był warunek po to żeby też sprzedawał
            #     reward += price_difference * self.shares # Straty związane z brakiem sprzedaży

        # if previous_shares == self.shares:
        #     if price_difference < 0 and self.cash < self.current_price:
        #         reward =-10
        #     if price_difference > 0 and self.shares==0:
        #         reward =-10

        # Obliczamy zmianę wartości portfolio w procentach
        # portfolio_difference = 1 - (self.portfolio_value / previous_portfolio_value)

        # if self.step_count > 1:
        #     self.profit_change=self.current_profit-previous_profit

        self.last_reward = reward

        return self.get_observation(), reward, done, False, {}

    def render(self):
        log_string = (
            f"Krok: {self.step_count} | "
            f"Cena: {self.current_price:.2f} | "
            f"Gotówka: {self.cash:.2f} | "
            f"Akcje: {self.shares} | "
            f"Wartość portfela: {self.portfolio_value:.2f} | "
            f"Reward: {self.last_reward:.4f} | "
            f"Profit: {self.current_profit:.4f} | "
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
            "profit_ratio": self.current_profit,
        }

    def close(self):
        pass
