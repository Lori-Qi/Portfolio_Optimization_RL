import numpy as np
import torch

# Assuming these imports are defined elsewhere in your project
from Environment.GBM_Env import GBM_Env
from Simulators.EnvGBMSimulator import EnvGBMSimulator
from Simulators.EstimatedGBMSimulator import EstimatedGBMSimulator
from Simulators.WGANSimulator import WGANSimulator
from Agent.dqn_agent import DQNAgent
from PerformanceVisualizer.PerformanceVisualizer import PerformanceVisualizer
from StrategyTester.SimulatorStrategyTester import SimulatorStrategyTester

def main():
    # Environment parameters
    S0 = 100
    W0 = 1000
    mu = 0.3
    sigma = 0.4
    r = 0.05
    T = 1
    num_steps = 125
    future_steps = 5
    dt = 1/252
    
    print('\nCreating environment and simulators')
    # Create environment
    env = GBM_Env(S0=S0, W0=W0, mu=mu, sigma=sigma, r=r, T=T, 
                 num_steps=num_steps, future_steps=future_steps, dt=dt)
    
    # Create simulators
    # true GBM simulator
    true_simulator = EnvGBMSimulator(mu=mu, sigma=sigma, r=r, dt=dt)
    
    # estimated GBM simulator
    estimated_simulator = EstimatedGBMSimulator()
    estimated_simulator.update_parameters(env.get_historical_data())
    
    # Testing WGAN generation before training agents
    print("\nTesting WGAN simulator")
    wgan_simulator = WGANSimulator(hidden_dim=128) 
    
    # Get the historical data
    historical_data = env.get_historical_data()
    print(f"Historical data shape: {historical_data.shape}, range: [{np.min(historical_data):.2f}, {np.max(historical_data):.2f}]")
    
    # Fit WGAN with more epochs and visualize samples
    print('\nFitting WGAN simulator')
    wgan_simulator.fit(historical_data, epochs=2000)  
    
    # Visualize WGAN predictions before agent training
    print('\nVisualizing WGAN price predictions')
    wgan_simulator.visualize_comparison(historical_data, simulated_days=5, num_simulations=10)
    
    # Test WGAN simulator by generating samples
    print('\nTesting WGAN sample generation.')
    last_price = historical_data[-1]
    print(f'Last price: {last_price:.2f}')
    for _ in range(5):
        next_price = wgan_simulator.simulate_one_day(historical_data)
        print(f'WGAN predicted next price: {next_price:.2f}, percent change: {((next_price/last_price)-1)*100:.2f}%')
    
    # Initialize agents with improved parameters
    print("\nInitializing agents...")
    agents = []
    for i in range(3):
        agent = DQNAgent(state_size=4,
                         action_space=np.linspace(-1, 2, 50),  
                         memory_size=3000,  
                         gamma=0.99,  
                         epsilon=1,
                         epsilon_min=0.01,
                         epsilon_decay=0.999,  
                         learning_rate=0.0005, 
                         batch_size=128)  
        agents.append(agent)
    
    # Initialize tester
    tester = SimulatorStrategyTester(env)
    
    # Training configurations with fewer episodes for testing
    training_configs = [
        {"agent": agents[0], "simulator": true_simulator, "episodes": 2000, "name": "True GBM"},
        {"agent": agents[1], "simulator": estimated_simulator, "episodes": 2000, "name": "Estimated GBM"},
        {"agent": agents[2], "simulator": wgan_simulator, "episodes": 5000, "name": "WGAN"}
    ]
    
    # Train agents and visualize training process
    trained_agents = []
    trainers = []
    
    for config in training_configs:
        print(f"\n{'='*50}")
        print(f"Training agent with {config['name']} simulator")
        print(f"{'='*50}")
        
        trained_agent, trainer = tester.train_agent(
            config["agent"], 
            config["simulator"],
            num_episode=config["episodes"],
            simulator_name=config["name"]
        )
        
        # Visualize training process
        print(f"\nVisualizing training process for {config['name']} strategy")
        trainer.visualize_training()
        
        trained_agents.append(trained_agent)
        trainers.append(trainer)
        
        # Add strategy to tester
        tester.add_strategy(trained_agent, f"{config['name']} Strategy")
    
    # Before evaluation, create a performance visualizer to inspect agent behavior
    print("\nVisualizing agent performance in sample episodes")
    for i, agent in enumerate(trained_agents):
        visualizer = PerformanceVisualizer(env, agent, training_configs[i]["simulator"])
        print(f"\nVisualizing {training_configs[i]['name']} agent:")
        
        # debug: evaluate the specific agent
        print(f"\nDetailed performance evaluation for {training_configs[i]['name']} agent:")
        perf_results = visualizer.evaluate_performance(num_episodes=100)
    
    # Evaluate all strategies
    print("\nEvaluating all strategies...")
    tester.evaluate_all_strategies(n_episodes=100)
    
    # Compare and visualize results
    comparison_df = tester.compare_strategies()
    print("\nStrategy Comparison:")
    print(comparison_df)
    
    # Summary of results
    summary_df = tester.summarize_results()
    print("\nSummary of Results:")
    print(summary_df)
if __name__ == '__main__':
    main()