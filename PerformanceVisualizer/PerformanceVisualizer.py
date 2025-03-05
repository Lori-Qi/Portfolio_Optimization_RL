import numpy as np
import matplotlib.pyplot as plt
import torch

class PerformanceVisualizer:
    def __init__(self, env, agent, simulator=None):
        self.env = env
        self.agent = agent
        self.simulator = simulator
    
    # visualize portfolio allocation (actions) and performance for a few episodes
    def visualize_episode(self, max_episodes=5):
        for episode in range(max_episodes):
            state = self.env.reset()
            done = False
            
            # track the data in each episode
            steps = []
            stock_prices = []
            portfolio_values = []
            allocations = []
            
            step = 0
            
            while not done:
                # select the best action (no exploration)
                self.agent.epsilon = 0
                action, _ = self.agent.select_action(state)
                
                # record data before step
                steps.append(step)
                stock_prices.append(self.env.S_future[step])
                portfolio_values.append(self.env.W_future[step])
                allocations.append(action)
                
                # take action
                next_state, reward, done, info = self.env.step(action, self.simulator)
                state = next_state
                step += 1
            
            # add final state
            steps.append(step)
            stock_prices.append(self.env.S_future[step])
            portfolio_values.append(self.env.W_future[step])
            
            # create plots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            
            # stock price
            ax1.plot(steps, stock_prices, 'b-', label='Stock Price')
            ax1.set_ylabel('Price')
            ax1.set_title(f'Episode {episode+1} - Stock Price')
            ax1.legend()
            
            # portfolio value
            ax2.plot(steps, portfolio_values, 'g-', label='Portfolio Value')
            ax2.set_ylabel('Value')
            ax2.set_title('Portfolio Value')
            ax2.legend()
            
            # portfolio allocation
            ax3.bar(steps[:-1], allocations, width=0.8, label='Stock Allocation (w)')
            ax3.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            ax3.axhline(y=1, color='r', linestyle='-', alpha=0.3)
            ax3.set_ylim(-0.6, 1.6)
            ax3.set_ylabel('Allocation')
            ax3.set_xlabel('Step')
            ax3.set_title('Portfolio Allocation')
            ax3.legend()
            
            plt.tight_layout()
            plt.savefig(f'episode_{episode+1}_performance.png')
            plt.show()
            
            # calculate returns
            stock_returns = [(stock_prices[i+1]/stock_prices[i])-1 for i in range(len(stock_prices)-1)]
            portfolio_returns = [(portfolio_values[i+1]/portfolio_values[i])-1 for i in range(len(portfolio_values)-1)]
            
            print(f"Episode {episode+1} Summary:")
            print(f"  Initial Stock Price: {stock_prices[0]:.2f}")
            print(f"  Final Stock Price: {stock_prices[-1]:.2f}")
            print(f"  Stock Return: {((stock_prices[-1]/stock_prices[0])-1)*100:.2f}%")
            print(f"  Portfolio Return: {((portfolio_values[-1]/portfolio_values[0])-1)*100:.2f}%")
            print(f"  Average Allocation: {np.mean(allocations):.2f}")
            print()
    
    # evaluate agent performance over multiple episodes
    # compare to the baseline strategies (stock only; 50/50 allocation)    
    # Fixed evaluate_performance method for PerformanceVisualizer class
    def evaluate_performance(self, num_episodes=100):
        
        # tracking metrics
        agent_returns = []
        stock_only_returns = []
        fixed_alloc_returns = []  # 50/50 allocation
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            
            # initial values
            initial_price = self.env.S_future[0]
            initial_wealth = self.env.W_future[0]
            
            # debug initial values
            # print(f"Episode {episode+1}: Initial price = {initial_price:.2f}, Initial wealth = {initial_wealth:.2f}")
            
            # initialize wealth trackers (3 strategies: agent + 2 baseline methods)
            agent_wealth = initial_wealth
            stock_only_wealth = initial_wealth
            fixed_alloc_wealth = initial_wealth
            
            step = 0
            while not done:
                # get agent's action 
                self.agent.epsilon = 0 # no exploration
                action, _ = self.agent.select_action(state)

                # debug
                if episode == 0 and step % 2 == 0:
                    print(f"Step {step}: Action = {action:.4f}, Current price = {self.env.S_future[self.env.current_step]:.2f}")
                
                # take action in environment
                next_state, reward, done, info = self.env.step(action, self.simulator)
                
                # update agent's wealth
                agent_wealth = self.env.W_future[self.env.current_step]
                
                # calculate returns for baseline strategies
                stock_return = (self.env.S_future[self.env.current_step] / 
                              self.env.S_future[self.env.current_step-1]) - 1
                risk_free_return = np.exp(self.env.r * self.env.dt) - 1
                
                # update baseline strategies' wealth
                stock_only_wealth *= (1 + stock_return)
                fixed_alloc_wealth *= (1 + 0.5*stock_return + 0.5*risk_free_return)
                
                # debug
                if episode == 0 and step % 2 == 0:
                    print(f"  Stock return = {stock_return*100:.2f}%, Risk-free = {risk_free_return*100:.4f}%")
                    print(f"  Agent wealth = {agent_wealth:.2f}, Stock-only = {stock_only_wealth:.2f}, Fixed = {fixed_alloc_wealth:.2f}")
                
                # move to next state
                state = next_state
                step += 1
            
            # Final values at episode end
            final_stock_price = self.env.S_future[self.env.current_step]
            
            # debug final values
            # print(f"Episode {episode+1} complete: Steps = {step}")
            # print(f"  Final price = {final_stock_price:.2f}, Price change = {((final_stock_price/initial_price)-1)*100:.2f}%")
            # print(f"  Final agent wealth = {agent_wealth:.2f}, Return = {((agent_wealth/initial_wealth)-1)*100:.2f}%")
            # print(f"  Final stock-only wealth = {stock_only_wealth:.2f}, Return = {((stock_only_wealth/initial_wealth)-1)*100:.2f}%")
            # print(f"  Final fixed alloc wealth = {fixed_alloc_wealth:.2f}, Return = {((fixed_alloc_wealth/initial_wealth)-1)*100:.2f}%")
            
            # calculate total returns for the episode
            agent_return = (agent_wealth / initial_wealth) - 1
            stock_only_return = (stock_only_wealth / initial_wealth) - 1
            fixed_alloc_return = (fixed_alloc_wealth / initial_wealth) - 1
            
            # store returns
            agent_returns.append(agent_return)
            stock_only_returns.append(stock_only_return)
            fixed_alloc_returns.append(fixed_alloc_return)
        
        # calculate statistics
        agent_mean = np.mean(agent_returns)
        agent_std = np.std(agent_returns)
        stock_mean = np.mean(stock_only_returns)
        stock_std = np.std(stock_only_returns)
        fixed_mean = np.mean(fixed_alloc_returns)
        fixed_std = np.std(fixed_alloc_returns)
        
        # Print results
        print(f"Performance Evaluation ({num_episodes} episodes):")
        print(f"  Agent: Mean Return = {agent_mean*100:.2f}%, Std Dev = {agent_std*100:.2f}%")
        print(f"  Stock Only: Mean Return = {stock_mean*100:.2f}%, Std Dev = {stock_std*100:.2f}%")
        print(f"  50/50 Fixed: Mean Return = {fixed_mean*100:.2f}%, Std Dev = {fixed_std*100:.2f}%")
        
        # calculate Sharpe ratios (annualized)
        # assuming 252 trading days per year
        agent_sharpe = (agent_mean / agent_std) * np.sqrt(252/self.env.future_steps) if agent_std > 0 else 0
        stock_sharpe = (stock_mean / stock_std) * np.sqrt(252/self.env.future_steps) if stock_std > 0 else 0
        fixed_sharpe = (fixed_mean / fixed_std) * np.sqrt(252/self.env.future_steps) if fixed_std > 0 else 0
        
        print(f"  Sharpe Ratios:")
        print(f"    Agent: {agent_sharpe:.2f}")
        print(f"    Stock Only: {stock_sharpe:.2f}")
        print(f"    50/50 Fixed: {fixed_sharpe:.2f}")
        
        # visualize return distributions
        plt.figure(figsize=(12, 8))
        
        plt.hist(agent_returns, bins=20, alpha=0.5, label='DQN Agent')
        plt.hist(stock_only_returns, bins=20, alpha=0.5, label='Stock Only')
        plt.hist(fixed_alloc_returns, bins=20, alpha=0.5, label='50/50 Fixed')
        
        plt.axvline(agent_mean, color='blue', linestyle='dashed', linewidth=2)
        plt.axvline(stock_mean, color='orange', linestyle='dashed', linewidth=2)
        plt.axvline(fixed_mean, color='green', linestyle='dashed', linewidth=2)
        
        plt.title('Return Distribution Comparison')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.savefig('return_distribution.png')
        plt.show()
        
        # return results in a dictionary
        return {
            'agent_mean': agent_mean,
            'agent_std': agent_std,
            'agent_sharpe': agent_sharpe,
            'stock_mean': stock_mean,
            'stock_std': stock_std,
            'stock_sharpe': stock_sharpe,
            'fixed_mean': fixed_mean,
            'fixed_std': fixed_std,
            'fixed_sharpe': fixed_sharpe,
            'agent_returns': agent_returns,
            'stock_returns': stock_only_returns,
            'fixed_returns': fixed_alloc_returns
        }