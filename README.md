# simulation-eip

Simulation data and figures for the paper:

**"Dynamic Fee Markets at Sub-Second Timescales: Adapting EIP-1559 for High-Throughput Blockchains"**
Petar Zhivkov, Eric Chen
*Mathematics*, MDPI, 2026.

## Contents

### `latex/`
Final figures used in the published paper:
- `scenario2_*` — Variable demand (sine wave) scenarios
- `scenario3_*` — Demand spikes scenarios
- `scenario4_*` — Spam attack scenarios
- `extended_sine_1000.png`, `extended_spam_1000.png` — Extended 1000-block horizon validation
- `150M_demand_spikes.png`, `150M_spam_attack.png` — Injective mainnet validation figures

### `simulation_results/`
Final simulation CSV data files at the calibrated parameter r_max = 0.05:
- `injective_scenario2_*_rmax005.csv` — Variable demand runs (per-block and MA-25)
- `injective_scenario3_*_rmax005.csv` — Demand spikes runs (per-block and MA-25)
- `injective_scenario4_*_rmax005.csv` — Spam attack runs (per-block and MA-25)
- `sensitivity_h2_ma_window_rmax005.csv` — Moving average window sensitivity analysis

## Abstract

EIP-1559 dynamic fee mechanisms have been extensively studied for Ethereum's 12-second block environment but remain uncharacterized at sub-second timescales. This paper presents an agent-based simulation study of an EIP-1559 adaptation for Injective, a Layer 1 blockchain operating at 600 ms block times. Across twelve simulation runs, the analysis finds that: (1) temporal smoothing mechanisms produce mixed effects at sub-second cadence with per-block adjustment being preferable; (2) transitioning to a 300M gas limit reduces peak fees by 31% under variable demand; and (3) per-block mechanisms establish spam barriers in 17-32 seconds versus Ethereum's 4-6 minutes. Results are validated against live Injective mainnet data.
