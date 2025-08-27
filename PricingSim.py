import numpy as np
import matplotlib.pyplot as plt

#Initialize Variables
S0 = 100  # Initial stock price
K = 105   # Strike price
T = 1     # Time to maturity in years
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility of the underlying stock
num_simulations = 1000  # Number of Monte Carlo simulations
num_steps = 1000  # Number of time steps for each simulation

# Function to simulate stock price paths using Geometric Brownian Motion
def simulate_stock_price(S0, T, r, sigma, num_steps):
    dt = T / num_steps
    prices = np.zeros(num_steps)
    prices[0] = S0
    for t in range(1, num_steps):
        z = np.random.standard_normal()  # Generate a standard normal random variable
        prices[t] = prices[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return prices

# Run Monte Carlo simulations and calculate the average stock price at maturity
time = np.linspace(0, T, num_steps)
stock_prices_at_maturity = np.zeros(num_simulations)
for i in range(num_simulations):
    stock_prices = simulate_stock_price(S0, T, r, sigma, num_steps)
    plt.plot(time, stock_prices, color='blue', alpha=0.01)  # Plot each simulation with low opacity
    stock_prices_at_maturity[i] = stock_prices[-1]

# Calculate the average stock price at maturity
average_price_at_maturity = np.mean(stock_prices_at_maturity)
print(f"Average stock price at maturity: {average_price_at_maturity:.2f}")

# Calculate the European call option price using the average stock price at maturity
profits = np.maximum(0, stock_prices_at_maturity - K * np.ones(num_simulations))
average_profit = np.mean(profits)
call_option_price = np.exp(-r * T) * average_profit
print(f"European Call Option Price: {call_option_price:.2f}")

# Plot the stock price paths
plt.title('Monte Carlo Simulations of Stock Price Paths')
plt.xlabel('Time (years)')
plt.ylabel('Stock Price')
plt.grid(True)
plt.show()
