### **Research Methodology: An Agent-Based Simulation of EIP-1559**

The study investigates the dynamic properties of Ethereum's EIP-1559 transaction fee mechanism, with a specific focus on the behavior of the `basefee` in a stationary environment. The core of the methodology is an **agent-based model (ABM)**, which simulates the interactions of heterogeneous, economically rational agents within a structured market environment that mimics the Ethereum protocol.

#### **1. Modeling Framework**

The research adapts the agent-based framework from Huberman et al. (2019) to model the EIP-1559 fee market. The model consists of three primary components:
1.  **User Agents**: Economically motivated actors who generate transactions.
2.  **Market Environment**: A system composed of a transaction pool, a block production process, and a blockchain.
3.  **Protocol Rules**: The EIP-1559 mechanism for fee setting, transaction validation, and block size targeting.

The simulation proceeds in discrete time steps, where each step represents the creation of a new block.

#### **2. Agent Specification and Behavior**

User agents are the fundamental actors driving the simulation. Their behavior is defined by the following characteristics:

*   **Arrival Process**: The arrival of new users wishing to transact is modeled as a **Poisson process**, with a parameter λ representing the mean number of user arrivals per time step (i.e., per block).
*   **Economic Profile**: Each agent is heterogeneous and defined by two private attributes drawn from uniform distributions:
    *   **Value (`v`)**: The intrinsic value an agent derives from having their transaction included, representing their maximum willingness to pay.
    *   **Waiting Cost**: A per-block cost for delayed inclusion, representing the agent's time preference or impatience.
*   **Decision-Making and Payoff Function**: Agents are rational and aim to maximize their economic utility. An agent decides whether to submit a transaction by calculating their expected payoff, defined as:
    `payoff = value - (cost_from_waiting * expected_wait_time) - transaction_fee`
    A crucial assumption is that agents forecast their payoff based on the current `basefee` and a fixed expectation of waiting time (5 blocks). If the calculated payoff is negative, the agent **balks** and abstains from participating in the market for that transaction.
*   **Transaction Submission**: If an agent's expected payoff is positive, they submit a transaction. The EIP-1559 parameters are set according to the agent's value: the `max_fee` is set to the agent's `value`, and the `gas_premium` (tip) is set to a fixed, low value (1 Gwei).

#### **3. Market Environment and Protocol Simulation**

The agents interact within a simulated Ethereum environment governed by the EIP-1559 rules.

*   **Transaction Pool (`txpool`)**: Submitted transactions are collected in a pool.
*   **Block Production**: At each time step, a new block is created. Transactions are selected from the pool for inclusion based on their profitability (i.e., their effective tip).
*   **EIP-1559 Mechanics**:
    *   **Validity**: A transaction is considered valid for inclusion only if its `max_fee` is greater than or equal to the prevailing `basefee`.
    *   **Basefee Adjustment**: The `basefee` is the core dynamic component. After a block is produced, the `basefee` for the next block is adjusted algorithmically. It increases if the previous block's gas usage was above the target (50% of the block gas limit) and decreases if it was below. This mechanism is designed to apply economic pressure that steers block fullness toward the target.

#### **4. Experimental Design**

The study focuses on a **stationary scenario** to observe the convergence properties of the `basefee`.

*   **Demand Scenario**: The simulation is run with a constant average demand. The Poisson arrival rate λ is fixed at 2000 new users for each block over a 200-block simulation period.
*   **Simplifying Assumptions**: To isolate the effects of the fee mechanism, all transactions are assumed to have the same `gas_used` (21,000).
*   **Data Collection**: At each time step (block), a range of metrics is recorded, including the `basefee`, the number of new users, the number of submitted transactions, the number of included transactions, the average tip, and the transaction pool length.

The primary methodological contribution is the use of this agent-based simulation to demonstrate that, under conditions of stationary demand, the `basefee` dynamically adjusts to an equilibrium. At this equilibrium, the fee is just high enough to price out a sufficient number of users, causing them to balk, such that the number of included transactions naturally matches the protocol's target block size.