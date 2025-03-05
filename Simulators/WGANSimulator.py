# WGAN-GP generator
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.autograd import grad

# WGAN-GP generator with improved architecture
class Generator(nn.Module):
    def __init__(self, noise_dim=100, hidden_dim=128, output_dim=1):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        self.model = nn.Sequential(
            nn.Linear(noise_dim + 1, hidden_dim), # +1 is the additional condition
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Initialize weight
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                layer.bias.data.fill_(0.01) # set the bias of nn.Linear to 0.01

    def forward(self, z, condition):
        x = torch.cat([z, condition], dim=1)
        return self.model(x)

# WGAN-GP discriminator with improved architecture
class Discriminator(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                layer.bias.data.fill_(0.01)

    def forward(self, z, condition):
        x = torch.cat([z, condition], dim=1)
        return self.model(x)

class WGAN_GP:
    def __init__(
        self,
        noise_dim=100,
        hidden_dim=128,
        discriminator_iter=5,
        lambda_gp=4,
        device=None,
        lr=0.0001 
    ):
        self.noise_dim = noise_dim
        self.discriminator_iter = discriminator_iter
        self.lambda_gp = lambda_gp
        self.lr = lr
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_log_return = 1.0  # Initialize with default value

        self.generator = Generator(noise_dim, hidden_dim).to(self.device)
        self.discriminator = Discriminator(hidden_dim=hidden_dim).to(self.device)

        # Update optimizer betas for better stability
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.9))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

        # track training history
        self.g_losses = []
        self.d_losses = []
        self.trained = False

    # calculate WGAN-GP gradient penalty
    # real_data: [batch_size, ...]
    def compute_gradient_penalty(self, real_data, fake_data, condition):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device)
        
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.clone().detach().requires_grad_(True)

        dis_interpolates = self.discriminator(interpolates, condition)

        gradients = grad(
            outputs=dis_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(dis_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # we train on log-returns    
    def train(self, price_history, epochs=1000, batch_size=32, log_freq=10):
        # data preprocessing with normalization
        log_returns = np.diff(np.log(price_history))
        self.max_log_return = np.max(np.abs(log_returns)) + 1e-8  # prevent division by zero
        log_returns_normalized = log_returns / self.max_log_return


        returns_tensor = torch.FloatTensor(log_returns_normalized).to(self.device)
        prices_tensor = torch.FloatTensor(price_history[:-1]).to(self.device).view(-1, 1)

        dataset = torch.utils.data.TensorDataset(returns_tensor.view(-1, 1), prices_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        start_time = time.time()
        
        for epoch in range(epochs):
            d_loss_epoch = 0
            g_loss_epoch = 0
            batches = 0

            for returns, conditions in dataloader:
                batch_size = returns.size(0)
                batches += 1
                real_data = returns.to(self.device)
                cond = conditions.to(self.device)

                # train discriminator (5 iters)
                for _ in range(self.discriminator_iter):
                    self.discriminator.zero_grad()
                    
                    # real data
                    d_real = self.discriminator(real_data, cond).mean()
                    
                    # fake data with detach
                    z = torch.randn(batch_size, self.noise_dim, device=self.device)
                    fake_data = self.generator(z, cond).detach()

                    # score the fake data
                    d_fake = self.discriminator(fake_data, cond).mean()
                    
                    # gradient penalty
                    gp = self.compute_gradient_penalty(real_data, fake_data, cond)
                    # loss of dicriminator
                    d_loss = d_fake - d_real + self.lambda_gp * gp
                    d_loss.backward() 
                    self.optimizer_d.step() # update the parameters of D
                    d_loss_epoch += d_loss.item()

                # train generator
                self.generator.zero_grad()
                z = torch.randn(batch_size, self.noise_dim, device=self.device)
                gen_samples = self.generator(z, cond)

                # loss of generator
                g_loss = -self.discriminator(gen_samples, cond).mean()
                g_loss.backward()
                self.optimizer_g.step()
                g_loss_epoch += g_loss.item()

            # calculate average losses for this epoch
            avg_d_loss = d_loss_epoch / batches
            avg_g_loss = g_loss_epoch / batches
            
            # Record losses
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)

            # Logging
            if epoch % log_freq == 0 or epoch == epochs - 1:
                elapsed = time.time() - start_time
                print(f'Epoch [{epoch}/{epochs}] | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f} | Time: {elapsed:.2f}s')
            
        self.trained = True
        print(f'Training completed in {time.time() - start_time:.2f}s')

    def sample(self, condition, num_samples=1):
        if not self.trained:
            raise ValueError('Model must be trained before sampling')

        self.generator.eval()
        with torch.no_grad():
            condition_tensor = torch.FloatTensor([condition]).view(1, 1).to(self.device)
            condition_tensor = condition_tensor.repeat(num_samples, 1)
            z = torch.randn(num_samples, self.noise_dim, device=self.device)
            samples = self.generator(z, condition_tensor)
            samples = samples * self.max_log_return  # Denormalize
        self.generator.train()
        return samples

    def visualize_training(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.d_losses)
        plt.title('Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(self.g_losses)
        plt.title('Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.tight_layout()
        plt.savefig('wgan_gp_training.png')
        plt.show()

    def save(self, path='wgan_gp_model.pth'):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.optimizer_g.state_dict(),
            'd_optimizer_state_dict': self.optimizer_d.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'max_log_return': self.max_log_return,
            'trained': self.trained
        }, path)
        print(f'Model has saved to {path}')

    def load(self, path='wgan_gp_model.pth'):
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']
        self.max_log_return = checkpoint.get('max_log_return', 1.0)  # Backward compatibility
        self.trained = checkpoint['trained']
        print(f'Model has loaded from {path}')


# WGAN-GP Simulator
class WGANSimulator:
    def __init__(self, noise_dim=100, hidden_dim=128, wgan_epochs=2000, device=None):
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.wgan_epochs = wgan_epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = None

    def fit(self, historical_prices, epochs=None):

        self.model = WGAN_GP(
            noise_dim=self.noise_dim,
            hidden_dim=self.hidden_dim,
            device=self.device,
            lr=0.0001 
        )
        self.model.train(historical_prices, epochs=epochs or self.wgan_epochs)
        return self

    # simulate next one day price
    def simulate_one_day(self, historical_prices):
        if not self.model or not self.model.trained:
            raise ValueError('Model must be trained before simulation.')
        
        last_price = historical_prices[-1]

        # set generator to eval mode
        self.model.generator.eval()
        
        # generate new random noise for each call
        z = torch.randn(1, self.model.noise_dim, device=self.model.device)
        
        # convert last price to tensor and proper shape
        condition_tensor = torch.FloatTensor([last_price]).view(1, 1).to(self.model.device)
        
        # generate sample
        with torch.no_grad():
            log_return = self.model.generator(z, condition_tensor).cpu().numpy()[0][0]
            # denormalize
            log_return = log_return * self.model.max_log_return
        
        # return to train mode
        self.model.generator.train()
        
        # calculate next price
        return last_price * np.exp(log_return)

    def visualize_comparison(self, historical_prices, simulated_days=5, num_simulations=10):
        if self.model is None or not self.model.trained:
            raise ValueError("Simulator must be fitted before visualization")
        
        plt.figure(figsize=(12, 6))

        # plot historical prices
        days = np.arange(len(historical_prices))
        plt.plot(days, historical_prices, 'b-', label='Historical Prices')

        # generate multiple simulations
        for i in range(num_simulations):
            simulation = [historical_prices[-1]]
            current_history = historical_prices.copy()

            for day in range(simulated_days):
                next_price = self.simulate_one_day(current_history)
                simulation.append(next_price)
                current_history = np.append(current_history, next_price)

            sim_days = np.arange(len(historical_prices)-1, len(historical_prices) + simulated_days)
            plt.plot(sim_days, simulation, 'r-', alpha=0.3)

        # Only add the label once to the legend
        plt.plot([], [], 'r-', label='Simulated Paths')

        plt.title('WGAN-GP Historical Prices vs. Simulated Paths')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig('wgan_gp_comparison.png')
        plt.show()

    def sample_many_log_returns(self, historical_prices, num_samples = 1000):
        if not self.model or not self.model.trained:
            raise ValueError('Model must be trained before simulation.')

        last_prices = historical_prices[-1]
        self.model.generator.eval()

        log_returns = []
        with torch.no_grad():
            for _ in range(num_samples):
                z = torch.randn(1, self.model.noise_dim, device=self.model.device)
                condition_tensor = torch.FloatTensor([last_prices]).view(1, 1).to(self.model.device)
                log_return = self.model.generator(z, condition_tensor).cpu().numpy()[0][0]

                # denormalize
                log_return = log_return * self.model.max_log_return
                log_returns.append(log_return)

        self.model.generator.train()

        return np.array(log_returns)

    def save(self, path='wgan_gp_simulator.pth'):
        if self.model is not None:
            self.model.save(path)
    
    def load(self, path='wgan_gp_simulator.pth'):
        self.model = WGAN_GP(
            noise_dim=self.noise_dim,
            hidden_dim=self.hidden_dim,
            device=self.device
        )
        self.model.load(path)