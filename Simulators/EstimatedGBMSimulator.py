# GBM with estimated parameters
import numpy as np
from Simulators.PriceSimulatorBase import PriceSimulatorBase

class EstimatedGBMSimulator(PriceSimulatorBase):
    def __init__(self, dt = 1/252):
        self.dt = dt
        self.mu = None
        self.sigma = None
        
    def update_parameters(self, historical_prices):
        
        # use log returns to estimate mu and sigma
        log_returns = np.diff(np.log(historical_prices))

        self.mu = np.mean(log_returns)/self.dt
        self.sigma = np.std(log_returns)/np.sqrt(self.dt)

        return self

    def simulate_one_day(self, historical_prices):

        if self.mu is None or self.sigma is None:
            raise ValueError("Simulator must be fitted before simulation")

        last_price = last_price = historical_prices[-1]
        dW = np.random.normal(0, np.sqrt(self.dt))
        next_price = last_price * np.exp((self.mu - self.sigma**2/2) * self.dt + self.sigma * dW)
        return next_price