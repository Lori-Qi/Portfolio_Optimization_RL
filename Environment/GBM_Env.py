import numpy as np

class GBM_Env:
    def __init__(self, S0, W0, mu, sigma, r, T, num_steps=125, future_steps=5, dt=1/252):
        # parameters of the GBM Environment
        self.S0 = S0
        self.W0 = W0
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.T = T
        self.num_steps = num_steps
        self.dt = dt
        self.future_steps = future_steps

        # Initialise variables for stock prices and portfolio wealth
        self.S_hist = np.zeros(num_steps+1)  # 126 days (0 to 125)
        self.W_hist = np.zeros(num_steps+1)

        # Initialize future arrays
        self.S_future = np.zeros(future_steps+1)  # Days 126 to 131
        self.W_future = np.zeros(future_steps+1)

        self.S_hist[0] = self.S0
        self.W_hist[0] = self.W0

        # current step and done flag
        self.current_step = 0
        self.done = False

        # Generate historical data
        self._historical_data()

    def _historical_data(self):
        for t in range(1, self.num_steps+1):  # [1,125], 126 days total
            dW = np.random.normal(0, np.sqrt(self.dt))
            self.S_hist[t] = self.S_hist[t-1] * np.exp((self.mu-self.sigma**2/2) * self.dt + self.sigma*dW)
            self.W_hist[t] = self.W_hist[t-1]  # Wealth doesn't change during historical period

    def reset(self):
        # Reset only the future arrays
        self.S_future = np.zeros(self.future_steps+1)
        self.W_future = np.zeros(self.future_steps+1)

        # Set the starting point to the last historical value
        self.S_future[0] = self.S_hist[-1]
        self.W_future[0] = self.W_hist[-1]  # Keep the wealth from the end of historical period

        self.current_step = 0
        self.done = False

        return self.get_state()

    def step(self, action, simulator=None):
        if simulator is None:
            simulator = self.true_simulator

        if self.done:
            raise RuntimeError('The episode has ended, please call reset()')

        # Get portfolio weight
        w = action  # Can add clipping if needed: np.clip(action, -1, 2)
        
        # Simulate just the next day's price
        full_history = self.get_full_price_history()
        next_price = simulator.simulate_one_day(full_history)
        
        # Store the simulated price
        self.S_future[self.current_step + 1] = next_price
        
        # Calculate portfolio return for the day
        r_stock = (self.S_future[self.current_step + 1] / self.S_future[self.current_step]) - 1
        r_riskfree = np.exp(self.r * self.dt) - 1
        r_portfolio = w * r_stock + (1 - w) * r_riskfree
        
        # Update portfolio wealth
        self.W_future[self.current_step + 1] = self.W_future[self.current_step] * (1 + r_portfolio)
        
        # Calculate daily return as reward
        reward = r_portfolio
        
        # Move to the next step
        self.current_step += 1
        self.done = self.current_step >= self.future_steps
        
        # Get the next state
        next_state = self.get_state()

        info = {
            'stock_price': self.S_future[self.current_step],
            'wealth': self.W_future[self.current_step],
            'stock_return': r_stock,
            'portfolio_return': r_portfolio
        }
        
        return next_state, reward, self.done, info

    def get_full_price_history(self):
        """Get the full price history including historical and simulated data"""
        # Combine historical data with already simulated future data
        if self.current_step == 0:
            return self.S_hist
        else:
            return np.concatenate([
                self.S_hist, 
                self.S_future[1:self.current_step+1]  # Only include simulated days so far
            ])

    def get_state(self, window=20):
        """Get the current state based on the rolling window"""
        # Get the full price history up to the current step
        full_history = self.get_full_price_history()
        
        # Calculate the start index for the window
        start_idx = max(0, len(full_history) - window)
        
        # Get the window of prices
        window_data = full_history[start_idx:]
        
        # Calculate log returns
        log_returns = np.diff(np.log(window_data))
        
        # Get current stock price and wealth
        current_price = full_history[-1]
        current_wealth = self.W_future[self.current_step]
        
        # Calculate volatility and latest log return
        volatility = np.std(log_returns) if len(log_returns) > 0 else 0
        log_return = log_returns[-1] if len(log_returns) > 0 else 0
        
        # Construct state vector
        state = np.array([current_price,
                          current_wealth,
                          volatility, 
                          log_return], dtype=np.float32)
        
        return state

    def get_total_reward(self):
        """Calculate the total reward for the episode"""
        if self.current_step == 0:
            return 0
        
        total_return = 0
        for t in range(min(self.current_step, self.future_steps)):
            daily_return = (self.W_future[t+1] / self.W_future[t]) - 1
            total_return += daily_return
        return total_return

    def get_historical_data(self):
        return self.S_hist

    def evaluate_policy(self, policy_fn, n_episodes = 10000, simulator = None):
        
        total_returns = []

        for _ in range(n_episodes):
            state = self.reset()
            done = False

            while not done:
                action = policy_fn(state)
                next_state, reward, done, _ = self.step(action, simulator)
                state = next_state

            total_returns.append(self.get_total_reward())

        return np.mean(total_returns)
