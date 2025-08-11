#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse

class PINN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=1, num_layers=4):
        super(PINN, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)

class HeatEquationPINN(PINN):
    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(input_dim=2, output_dim=1, **kwargs)
        self.alpha = alpha

    def pde_loss(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)
        xt = torch.cat([x, t], dim=1)
        u = self.forward(xt)
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        pde_residual = u_t - self.alpha * u_xx
        return torch.mean(pde_residual**2)

    def boundary_loss(self, x_boundary, t_boundary, u_boundary):
        xt_boundary = torch.cat([x_boundary, t_boundary], dim=1)
        u_pred = self.forward(xt_boundary)
        return torch.mean((u_pred - u_boundary)**2)

    def initial_loss(self, x_initial, u_initial):
        t_initial = torch.zeros_like(x_initial)
        xt_initial = torch.cat([x_initial, t_initial], dim=1)
        u_pred = self.forward(xt_initial)
        return torch.mean((u_pred - u_initial)**2)

class PINNTrainer:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.losses = {'total': [], 'pde': [], 'boundary': [], 'initial': []}

    def train_heat_equation(self, n_epochs=5000, domain_bounds=None):
        if domain_bounds is None:
            domain_bounds = {'x': [0, 1], 't': [0, 0.5]}
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            n_pde, n_boundary, n_initial = 1000, 100, 100
            x_pde = torch.rand(n_pde, 1) * (domain_bounds['x'][1] - domain_bounds['x'][0]) + domain_bounds['x'][0]
            t_pde = torch.rand(n_pde, 1) * (domain_bounds['t'][1] - domain_bounds['t'][0]) + domain_bounds['t'][0]
            x_boundary = torch.cat([torch.zeros(n_boundary//2,1)+domain_bounds['x'][0],
                                    torch.zeros(n_boundary//2,1)+domain_bounds['x'][1]])
            t_boundary = torch.rand(n_boundary,1)*(domain_bounds['t'][1]-domain_bounds['t'][0]) + domain_bounds['t'][0]
            u_boundary = torch.zeros(n_boundary,1)
            x_initial = torch.rand(n_initial,1)*(domain_bounds['x'][1]-domain_bounds['x'][0]) + domain_bounds['x'][0]
            u_initial = torch.sin(np.pi * x_initial)
            pde_loss = self.model.pde_loss(x_pde, t_pde)
            boundary_loss = self.model.boundary_loss(x_boundary, t_boundary, u_boundary)
            initial_loss = self.model.initial_loss(x_initial, u_initial)
            total_loss = pde_loss + 10 * boundary_loss + 10 * initial_loss
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.losses['total'].append(total_loss.item())
            self.losses['pde'].append(pde_loss.item())
            self.losses['boundary'].append(boundary_loss.item())
            self.losses['initial'].append(initial_loss.item())
            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}, PDE: {pde_loss.item():.6f}, BC: {boundary_loss.item():.6f}, IC: {initial_loss.item():.6f}")
        return self.losses

    def predict_heat_solution(self, x_test, t_test):
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_test, dtype=torch.float32)
            t_tensor = torch.tensor(t_test, dtype=torch.float32)
            xt_tensor = torch.cat([x_tensor, t_tensor], dim=1)
            u_pred = self.model(xt_tensor)
        return u_pred.numpy()

    def plot_solution(self, domain_bounds=None, save_path=None):
        if domain_bounds is None:
            domain_bounds = {'x': [0, 1], 't': [0, 0.5]}
        x_test = np.linspace(domain_bounds['x'][0], domain_bounds['x'][1], 100)
        t_test = np.linspace(domain_bounds['t'][0], domain_bounds['t'][1], 50)
        X, T = np.meshgrid(x_test, t_test)
        x_flat = X.flatten().reshape(-1, 1)
        t_flat = T.flatten().reshape(-1, 1)
        u_pred = self.predict_heat_solution(x_flat, t_flat)
        U = u_pred.reshape(X.shape)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        c1 = ax1.contourf(X, T, U, levels=20, cmap='viridis')
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.set_title('PINN Solution: u(x,t)')
        plt.colorbar(c1, ax=ax1)
        ax2.semilogy(self.losses['total'], label='Total Loss')
        ax2.semilogy(self.losses['pde'], label='PDE Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training History')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

def analytical_heat_solution(x, t, alpha=0.01):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * alpha * t)

def compare_with_analytical():
    model = HeatEquationPINN(alpha=0.01, hidden_dim=50, num_layers=4)
    trainer = PINNTrainer(model, lr=1e-3)
    trainer.train_heat_equation(n_epochs=2000)
    x_test = np.linspace(0, 1, 100).reshape(-1, 1)
    t_test = np.full_like(x_test, 0.1)
    u_pinn = trainer.predict_heat_solution(x_test, t_test)
    u_analytical = analytical_heat_solution(x_test.flatten(), t_test.flatten(), alpha=0.01)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(x_test.flatten(), u_pinn.flatten(), 'r-', label='PINN', linewidth=2)
    plt.plot(x_test.flatten(), u_analytical, 'b--', label='Analytical', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x, t=0.1)')
    plt.title('PINN vs Analytical Solution')
    plt.legend()
    plt.grid(True)
    plt.show()
    error = np.mean((u_pinn.flatten() - u_analytical)**2)
    print(f"Mean Squared Error: {error:.6f}")

def main():
    parser = argparse.ArgumentParser(description='Train PINNs for various PDEs')
    parser.add_argument('--equation', choices=['heat', 'navier_stokes'], default='heat', help='PDE to solve')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--compare', action='store_true', help='Compare with analytical solution')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    args = parser.parse_args()

    if args.equation == 'heat':
        if args.compare:
            compare_with_analytical()
        else:
            model = HeatEquationPINN(alpha=0.01)
            trainer = PINNTrainer(model, lr=args.lr)
            trainer.train_heat_equation(n_epochs=args.epochs)
            if args.plot:
                trainer.plot_solution()
    elif args.equation == 'navier_stokes':
        print("Navier-Stokes PINN training not implemented in this basic version")

if __name__ == "__main__":
    main()