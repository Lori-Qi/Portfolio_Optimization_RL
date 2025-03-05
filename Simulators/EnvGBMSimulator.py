# GBM with known parameters 
from numpy import np
from Simulators.PriceSimulatorBase import PriceSimulatorBase

class EnvGBMSimulator(PriceSimulatorBase):
    def __init__(self, mu, sigma, r , dt = 1/252):
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.dt = dt

    def simulate_one_day(self, historical_prices):
        last_price = historical_prices[-1]
        dW = np.random.normal(0, np.sqrt(self.dt))
        next_price = last_price * np.exp((self.mu - self.sigma**2/2) * self.dt + self.sigma * dW)
        return next_price