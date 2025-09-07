import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(1, os.getcwd())

# Overriding the default constants with Injective-specific ones
import abm1559.utils as utils
import abm1559.injective_config as config

utils.constants = {
    "INITIAL_BASEFEE": config.INITIAL_BASE_FEE,
    "BASEFEE_MAX_CHANGE_DENOMINATOR": 1 / config.MAX_CHANGE_RATE, # Convert rate to denominator
    "MAX_GAS_EIP1559": config.BLOCK_MAX_GAS,
    "SIMPLE_TRANSACTION_GAS": 21000, # A default value, can be adjusted
}

from abm1559.txpool import TxPool
from abm1559.users import User1559
from abm1559.userpool import UserPool
from abm1559.chain import Chain, Block1559
from abm1559.simulator import (
    spawn_poisson_demand,
    update_basefee_injective,
)

# --- Agent Definitions ---

class SpammerUser(User1559):
    """A user who always bids the minimum possible fee."""
    def __init__(self, wakeup_block, **kwargs):
        super().__init__(wakeup_block, **kwargs)
        # Spammers have very low value, just enough to cover the min base fee + a small tip
        self.value = config.MIN_BASE_FEE + 1 * (10**9)

    def decide_parameters(self, env):
        basefee = env["basefee"]
        gas_premium = 1 * (10**9) # A nominal 1 Gwei tip
        max_fee = basefee + gas_premium
        return {
            "max_fee": max_fee,
            "gas_premium": gas_premium,
            "start_block": self.wakeup_block,
        }

# --- Simulation & Plotting ---

def run_injective_sim(params):
    """Main simulation function, parameterized for experimentation."""

    utils.constants["MAX_GAS_EIP1559"] = params["block_max_gas"]
    utils.constants["TARGET_GAS_USED"] = int(params["block_max_gas"] * config.TARGET_GAS_RATIO)

    txpool = TxPool()
    basefee = utils.constants["INITIAL_BASEFEE"]
    chain = Chain()
    metrics = []
    user_pool = UserPool()

    for t in range(params["num_blocks"]):
        if t > 0 and t % config.RESET_INTERVAL == 0:
            basefee = utils.constants["INITIAL_BASEFEE"]

        env = {"basefee": basefee, "current_block": t}

        demand_this_block = params["demand_lambda_list"][t]
        users = spawn_poisson_demand(t, demand_this_block, params["user_class"])
        decided_txs = user_pool.decide_transactions(users, env)
        txpool.add_txs(decided_txs)

        selected_txs = txpool.select_transactions(env)
        txpool.remove_txs([tx.tx_hash for tx in selected_txs])
        block = Block1559(txs=selected_txs, parent_hash=chain.current_head, height=t, basefee=basefee)
        chain.add_block(block)

        basefee = update_basefee_injective(chain, basefee, {
            "moving_avg_window": params["moving_avg_window"],
            "target_gas": utils.constants["TARGET_GAS_USED"],
            "max_change_rate": config.MAX_CHANGE_RATE,
            "min_base_fee": config.MIN_BASE_FEE
        })

        row_metrics = {
            "block": t,
            "basefee": basefee / (10**9),
            "gas_used": block.gas_used(),
            "demand_lambda": demand_this_block,
            "block_max_gas": params["block_max_gas"],
            "moving_avg_window": params["moving_avg_window"],
        }
        metrics.append(row_metrics)

    return pd.DataFrame(metrics)

def millions_formatter(x, pos):
    return f'{int(x/1e6)}M'

def generate_plots(df, scenario_name, block_max_gas):
    """Generates and saves plots of the simulation results."""
    file_suffix_base = f"{int(block_max_gas/1e6)}M_{scenario_name}.pdf"

    # Plot 1: Base Fee
    plt.figure(figsize=(14, 7))
    plt.plot(df['block'], df['basefee'], label='Base Fee (Gwei)')
    plt.title(f'Base Fee Dynamics ({scenario_name} - {int(block_max_gas/1e6)}M Max Gas)')
    plt.xlabel('Block Number')
    plt.ylabel('Base Fee (Gwei)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'injective_basefee_comparison_{file_suffix_base}')
    plt.close()
    print(f"Saved plot: injective_basefee_comparison_{file_suffix_base}")

    # Plot 2: Gas Usage vs Target (Raw Values)
    fig, ax = plt.subplots(figsize=(14, 7))
    target_gas = int(block_max_gas * config.TARGET_GAS_RATIO)
    ax.plot(df['block'], df['gas_used'], label='Gas Used')
    ax.axhline(y=target_gas, color='r', linestyle='--', label=f'Target Gas ({int(target_gas/1e6)}M)')
    ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax.set_title(f'Gas Used vs. Target (Raw) ({scenario_name} - {int(block_max_gas/1e6)}M Max Gas)')
    ax.set_xlabel('Block Number')
    ax.set_ylabel('Gas Used')
    ax.legend()
    ax.grid(True)
    plt.savefig(f'injective_gas_usage_raw_{file_suffix_base}')
    plt.close()
    print(f"Saved plot: injective_gas_usage_raw_{file_suffix_base}")

    # Plot 3: Block Occupancy Ratio
    plt.figure(figsize=(14, 7))
    target_gas_ratio = config.TARGET_GAS_RATIO
    plot_df = df.copy()
    plot_df['gas_used_ratio'] = plot_df['gas_used'] / block_max_gas
    plt.plot(plot_df['block'], plot_df['gas_used_ratio'], label='Gas Used Ratio')
    plt.axhline(y=target_gas_ratio, color='r', linestyle='--', label=f'Target Occupancy ({target_gas_ratio:.0%})')
    plt.title(f'Block Occupancy Ratio ({scenario_name} - {int(block_max_gas/1e6)}M Max Gas)')
    plt.xlabel('Block Number')
    plt.ylabel('Block Occupancy Ratio')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'injective_gas_usage_ratio_{file_suffix_base}.pdf')
    plt.close()
    print(f"Saved plot: injective_gas_usage_ratio_{file_suffix_base}.pdf")

if __name__ == "__main__":
    num_blocks = 400
    all_results = []

    # --- Define Scenarios ---
    # We now run a more complex set of scenarios, including a spam test.
    
    # Base parameters
    base_params_150M = {"block_max_gas": 150_000_000, "moving_avg_window": 25, "num_blocks": num_blocks}
    base_params_300M = {"block_max_gas": 300_000_000, "moving_avg_window": 25, "num_blocks": num_blocks}

    # Regular User Scenarios
    regular_scenarios = {
        "low_demand": 2200,
        "high_demand": 9000,
        "spike_demand": 2200 
    }

    print("Running Regular Demand Scenarios for 150M Max Gas...")
    for name, demand in regular_scenarios.items():
        params = base_params_150M.copy()
        params["user_class"] = User1559
        if name == "spike_demand":
            demand_list = [demand] * num_blocks
            for i in range(40, num_blocks, 40): demand_list[i] = 13500
            params["demand_lambda_list"] = demand_list
        else:
            params["demand_lambda_list"] = [demand] * num_blocks
        
        results_df = run_injective_sim(params)
        results_df['scenario'] = name
        all_results.append(results_df)
        generate_plots(results_df, name, 150_000_000)

    print("\nRunning Spam Demand Scenario for 150M Max Gas...")
    spam_params_150M = base_params_150M.copy()
    spam_params_150M["user_class"] = SpammerUser
    spam_params_150M["demand_lambda_list"] = [2200] * num_blocks # Low demand context
    results_df = run_injective_sim(spam_params_150M)
    results_df['scenario'] = "spam_demand"
    all_results.append(results_df)
    generate_plots(results_df, "spam_demand", 150_000_000)

    # --- Save all results ---
    final_df = pd.concat(all_results)
    output_path = "injective_simulation_results_all_scenarios.csv"
    final_df.to_csv(output_path, index=False)
    print(f"\nSimulations complete. All results saved to {output_path}")
