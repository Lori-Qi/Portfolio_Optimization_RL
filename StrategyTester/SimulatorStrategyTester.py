import numpy as np
import pandas as pd
import torch
import time
from tqdm import tqdm

# test the strategies in GBM_Env
class SimulatorStrategyTester:
    def __init__(self, env, true_params = None):
        self.env = env
        self.true_params = true_params if true_params is not None else {
            'mu': env.mu,
            'sigma': env.sigma,
            'r': env.r
        }

        # store resylts
        self.results = {}
        self.strategy_names = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_agent(self, agent, simulator, num_episode = 500, simulator_name = 'Unknown'):
        print(f'Training agent with {simulator_name}.')

        # create the training manager
        from TrainingManager.TrainingManager import TrainingManager
        trainer = TrainingManager(
            env = self.env,
            agent = agent,
            simulator = simulator,
            target_update_freq = 5,
            max_episodes = num_episode,
            log_freq = 100,
            save_freq = 50,
            model_path = f'dqn_model_{simulator_name.lower()}.pth'
        )

        # train the agent
        start_time = time.time()
        trained_agent = trainer.train()
        training_time = time.time() - start_time

        print(f'Training completed in {training_time:.2f} seconds')

        return trained_agent, trainer

    def load_trained_agent(self, agent, model_path):
        agent.load(model_path)
        
        return agent

    def add_strategy(self, agent, strategy_name, model_path = None):
        if model_path:
            agent = self.load_trained_agent(agent, model_path)
        
        self.strategy_names.append(strategy_name)
        self.results[strategy_name] = {'agent': agent} # results is a dict.

    # evaluate a strategy in the true Env.
    def evaluate_startegy(self, strategy_name, n_episodes = 1000, seed = None):
        if strategy_name not in self.results:
            raise ValueError(f'Strategy {strategy_name} not found. Add it first using add_strategy()')

        if seed is not None:
            np.random.seed(seed)
        
        agent = self.results[strategy_name]['agent']
        
        # turn on the evaluation mode (epsilon = 0)
        original_epsilon = agent.epsilon
        agent.epsilon = 0

        # true env. simulator
        from Simulators.EnvGBMSimulator import EnvGBMSimulator
        true_simulator = EnvGBMSimulator(
            mu = self.true_params['mu'],
            sigma = self.true_params['sigma'],
            r = self.true_params['r'],
            dt = self.env.dt
        )

        # metrics
        total_returns = []
        final_wealths = []
        sharpe_ratios = []
        max_drawdowns = []
        weights_history = []
        price_history = []
        wealth_history = []

        # run evaluation episode
        for _ in tqdm(range(n_episodes), desc = f'Evaluating {strategy_name}'):
            state = self.env.reset()

            episode_rewards = []
            episode_weights = []
            episode_prices = [self.env.S_future[0]]
            episode_wealth = [self.env.W_future[0]]

            done = False
            while not done:
                # select action based on trained policy
                # agent is trained
                action, _ = agent.select_action(state)
                episode_weights.append(action)

                # take action in the true env.
                next_state, reward, done, info = self.env.step(action, true_simulator)

                # track metrics
                episode_rewards.append(reward)
                episode_prices.append(info['stock_price'])
                episode_wealth.append(info['wealth'])

                # update state
                state = next_state

            # compute episode metrics
            total_return = self.env.get_total_reward()
            final_wealth = episode_wealth[-1]

            # compute the Sharpe ratio of one episode
            returns_array = np.array(episode_rewards)
            sharp = np.mean(returns_array)/np.std(returns_array)

            # compute the max drawdown
            wealth_array = np.array(episode_wealth)
            cummax = np.maximum.accumulate(wealth_array)
            drawdown = (cummax - wealth_array) / cummax
            max_drawdown = np.max(drawdown)

            # save metrics
            total_returns.append(total_return)
            final_wealths.append(final_wealth)
            sharpe_ratios.append(sharp)
            max_drawdowns.append(max_drawdown)
            weights_history.append(episode_weights)
            price_history.append(episode_prices)
            wealth_history.append(episode_wealth)

        # restore original epsilon
        agent.epsilon = original_epsilon # ?

        # store evaluation results
        evaluation_results = {
            'total_returns': total_returns,
            'final_wealths': final_wealths,
            'sharpe_ratios': sharpe_ratios,
            'max_drawdowns': max_drawdowns,
            'weights_history': weights_history,
            'price_history': price_history,
            'wealth_history': wealth_history,
            'mean_return': np.mean(total_returns),
            'std_return': np.std(total_returns),
            'mean_wealth': np.mean(final_wealths),
            'mean_sharpe': np.mean(sharpe_ratios),
            'mean_drawdown': np.mean(max_drawdowns)
        }

        print(f'\nEvaluation results for {strategy_name}:')
        print(f'  Mean Return: {evaluation_results["mean_return"]:.4f}')
        print(f'  Mean Final Wealth: {evaluation_results["mean_wealth"]:.2f}')
        print(f'  Mean Sharpe Ratio: {evaluation_results["mean_sharpe"]:.4f}')
        print(f'  Mean Max Drawdown: {evaluation_results["mean_drawdown"]:.4f}')

        return evaluation_results

    def evaluate_all_strategies(self, n_episodes = 1000, seed = None):
        
        for strategy_name in self.strategy_names:
            evaluation_results = self.evaluate_startegy(strategy_name, n_episodes, seed)

            self.results[strategy_name].update(evaluation_results)

    # compare all strategies based on evaluation metrics
    def compare_strategies(self, metrics = None):
        if not metrics:
            metrics = ['mean_return', 'mean_wealth', 'mean_sharpe', 'mean_drawdown']

        comparison = {}
        for metric in metrics:
            comparison[metric] = {
                strategy: self.results[strategy].get(metric, None)
                for strategy in self.strategy_names
            }

        df = pd.DataFrame(comparison)

        return df

    # compute the differences between policies (average action difference)
    def compute_policy_difference(self):
        if len(self.strategy_names) < 2:
            return "Need at least 2 strategies to compare"

        # create grid for policy comparison
        test_states = self._generate_test_states(n_states = 1000)

        # compute actions for each policy on test states
        policy_actions = {}
        for strategy in self.strategy_names:
            agent = self.results[strategy]['agent']
            
            # turn on the evaluation mode
            original_epsilon = agent.epsilon
            agent.epsilon = 0
                
            actions = []
            for state in test_states:
                action, _ = agent.select_action(state)
                actions.append(action)

            policy_actions[strategy] = np.array(actions)

            # restore original epsilon
            agent.epsilon = original_epsilon

        # compute pairwise difference
        n_strategies = len(self.strategy_names)
        diff_matrix = np.zeros((n_strategies, n_strategies))

        for i in range(n_strategies):
            for j in range(n_strategies):
                strat1 = self.strategy_names[i]
                strat2 = self.strategy_names[j]
                
                # mean absolute diffeence
                diff = np.mean(np.abs(policy_actions[strat1] - policy_actions[strat2]))
                diff_matrix[i, j] = diff
        
        # create DataFrame
        diff_df = pd.DataFrame(diff_matrix, 
                              index=self.strategy_names,
                              columns=self.strategy_names)
        
        return diff_df

    def _generate_test_states(self, n_states = 1000):
        # Get state ranges from historical data
        historical_data = self.env.get_historical_data()
        log_returns = np.diff(np.log(historical_data))
        
        price_min, price_max = np.min(historical_data), np.max(historical_data)
        wealth_min, wealth_max = self.env.W0 * 0.5, self.env.W0 * 2.0
        vol_min, vol_max = np.std(log_returns) * 0.5, np.std(log_returns) * 2.0
        ret_min, ret_max = np.min(log_returns), np.max(log_returns)
        
        # Generate random states within these ranges
        test_states = []
        for _ in range(n_states):
            price = np.random.uniform(price_min, price_max)
            wealth = np.random.uniform(wealth_min, wealth_max)
            vol = np.random.uniform(vol_min, vol_max)
            ret = np.random.uniform(ret_min, ret_max)
            
            state = np.array([price, wealth, vol, ret], dtype=np.float32)
            test_states.append(state)

        return test_states

    def summarize_results(self):
        summary = {}

        for strategy in self.strategy_names:
            results = self.results[strategy]
            
            # compute the key statistics
            summary[strategy] = {
                'Mean Return': results['mean_return'],
                'Return Std': results['std_return'],
                'Mean Final Wealth': results['mean_wealth'],
                'Mean Sharpe Ratio': results['mean_sharpe'],
                'Mean Max Drawdown': results['mean_drawdown'],
                'Success Rate': np.mean(np.array(results['final_wealths']) > self.env.W0),
                'Avg Weight': np.mean([np.mean(w) for w in results['weights_history']]),
                'Avg Weight Std': np.mean([np.std(w) for w in results['weights_history']])
            }
        
        return pd.DataFrame(summary).T
