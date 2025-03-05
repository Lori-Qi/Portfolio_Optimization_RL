import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

class TrainingManager:
    def __init__(
        self,
        env,
        agent,
        simulator = None,
        target_update_freq = 5,
        max_episodes = 1000,
        log_freq = 10,
        save_freq = 50,
        model_path = 'dqn_model.pth'
    ):
        self.env = env
        self.agent = agent
        self.simulator = simulator
        self.target_update_freq = target_update_freq
        self.max_episodes = max_episodes
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.model_path = model_path
        
        # Training metrics
        self.episode_rewards = []
        self.avg_rewards = []
        self.training_times = []
        self.evaluation_scores = []

    def train(self):
        start_time = time.time()

        for episode in range(self.max_episodes):
            episode_start = time.time()
            state = self.env.reset()
            episode_reward = 0
            done = False

            # one episode loop
            while not done:
                #select action
                action, action_idx = self.agent.select_action(state)

                # take action in the environment
                next_state, reward, done, _ = self.env.step(action, self.simulator)

                # store experience in agent's memory
                self.agent.remember(state, action_idx, reward, next_state, done)

                # update state
                state = next_state
                episode_reward += reward

                # train agent by reply
                self.agent.replay()

            # update target model periodically
            if episode % self.target_update_freq == 0:
                self.agent.update_target_model()

              # save metrics
            self.episode_rewards.append(episode_reward)
            avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            self.avg_rewards.append(avg_reward)
            self.agent.reward_history.append(episode_reward)
            self.agent.epsilon_history.append(self.agent.epsilon)

            episode_time = time.time() - episode_start
            self.training_times.append(episode_time)

            # logging
            if episode % self.log_freq == 0:
                print(f"Episode: {episode}/{self.max_episodes}, Reward: {episode_reward:.4f}, Avg Reward: {avg_reward:.4f}, Epsilon: {self.agent.epsilon:.4f}, Time: {episode_time:.2f}s")
                
                # evaluate current policy
                if episode > 0:
                    eval_score = self.evaluate(5)
                    self.evaluation_scores.append(eval_score)
                    print(f"Evaluation Score: {eval_score:.4f}")
            
            # save model periodically
            if episode % self.save_freq == 0 and episode > 0:
                self.agent.save(f"{self.model_path}_ep{episode}")
                
        # final save
        self.agent.save(self.model_path)
        
        # total training time
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
        return self.agent

    # evaluate current policy without exploration
    # return the mean reward of the testing episodes
    def evaluate(self, num_episodes=10):

        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0  # turn off exploration for evaluation
            
        total_rewards = []
            
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
                
            while not done:
                action, _ = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action, self.simulator)
                state = next_state
                episode_reward += reward
                
            total_rewards.append(episode_reward)
            
        # restore original epsilon
        self.agent.epsilon = original_epsilon
            
        return np.mean(total_rewards)
        
    def visualize_training(self):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            
        # episode rewards
        axs[0, 0].plot(self.episode_rewards)
        axs[0, 0].set_title('Episode Rewards')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Reward')
            
        # moving average rewards
        axs[0, 1].plot(self.avg_rewards)
        axs[0, 1].set_title('Average Rewards (last 100 episodes)')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Avg Reward')
            
        # loss history
        if self.agent.loss_history:
            axs[1, 0].plot(self.agent.loss_history)
            axs[1, 0].set_title('Loss History')
            axs[1, 0].set_xlabel('Training Step')
            axs[1, 0].set_ylabel('Loss')
            
        # epsilon decay
        axs[1, 1].plot(self.agent.epsilon_history)
        axs[1, 1].set_title('Exploration Rate (Epsilon)')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Epsilon')
            
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.show()
            
        if self.evaluation_scores:
            plt.figure(figsize=(10, 5))
            eval_episodes = [i * self.log_freq for i in range(len(self.evaluation_scores))]
            plt.plot(eval_episodes, self.evaluation_scores)
            plt.title('Policy Evaluation During Training')
            plt.xlabel('Episode')
            plt.ylabel('Evaluation Score')
            plt.savefig('evaluation_scores.png')
            plt.show()
