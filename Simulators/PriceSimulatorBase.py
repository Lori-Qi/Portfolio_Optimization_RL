import numpy as np

# Base Price Simulator Interface
class PriceSimulatorBase:
    def simulate_one_day(self, historical_prices):
        raise NotImplementedError('Subclasses must implement this method')
    
    def simulate_days(self, historical_prices, days = 5):
        prices = []
        current_history = historical_prices.copy()

        for _ in range(days):
            next_price = self.simulate_one_day(current_history)
            prices.append(next_price)
            current_history = np.append(current_history, next_price)

        return np.array(prices)