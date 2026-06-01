# Sliding Mode Control and Optimal Control for Kuramoto Model

Comprehensive MATLAB implementation comparing four control strategies for synchronizing coupled oscillators in the Kuramoto model:

1. **PMP (Pontryagin's Maximum Principle)** - Optimal indirect control via Forward-Backward Sweep
2. **SMC (Sliding Mode Control)** - Robust discontinuous control with saturation
3. **Feedback** - Simple phase-to-mean regulator (baseline)
4. **Uncontrolled** - Free dynamics reference

## Files

### `kuramoto_smc.m`
Standalone implementation of **Sliding Mode Control** for Kuramoto synchronization.

**Features:**
- SMC law: `u_i = -K1*e_i - K2*sign(e_i)` where `e_i = θ_i - θ̄`
- Smoothed sign function to avoid chattering
- 6 comprehensive visualization plots
- Performance metrics and convergence analysis
- RK4 integration for accuracy

**Parameters:**
- `N = 10` oscillators
- `K = 2.0` coupling strength
- `K1 = 1.5` proportional gain
- `K2 = 0.8` discontinuous gain
- `T = 50 s` simulation time

**Output:**
- Phase evolution trajectories
- Order parameter `r(t)` (synchronization metric)
- Phase velocity convergence
- Synchronization error (log scale)
- Control magnitude evolution
- Phase space portrait
- Summary performance table

---

### `kuramoto_pmp_fbs_ode89_compare.m`
Comprehensive comparison of all four control methods with detailed performance analysis.

**Section 1: PMP (Forward-Backward Sweep)**
- Solves Pontryagin's optimality system iteratively
- Box-constrained control `u ∈ [-1, 1]`
- Convergence criterion on control updates
- Relaxation parameter for stability
- Computes cost functional `J(u)`

**Section 2: SMC**
- Direct integration of SMC dynamics
- Sampling on time grid for fair comparison
- Discontinuous control with saturation

**Section 3: Feedback**
- Tunes proportional gain `k_fb` over 8 candidates
- Selects best performing gain
- Non-optimal but practical baseline

**Section 4: Uncontrolled**
- Free dynamics (`u = 0`)
- Reference for evaluating control effectiveness

**Section 5-8: Analysis**
- Time to synchronization (`r ≥ 0.9`)
- Chattering metric (high-frequency oscillations)
- Cost breakdown (coherence vs. energy)
- Comparative visualizations

**Output Metrics:**
- Total cost `J`
- Coherence penalty (running + terminal)
- Control energy
- Time-average order parameter
- Final order parameter
- RMS and max control magnitudes
- Convergence time

## System Model

### Kuramoto Dynamics
$$\dot{\theta}_i = \omega_i + \sum_{j=1}^N K_{ij} \sin(\theta_j - \theta_i) + u_i$$

where:
- `θ_i` = phase of oscillator i
- `ω_i` = natural frequency (heterogeneous)
- `K_{ij}` = coupling strength (symmetric matrix)
- `u_i` = control input

### Cost Functional
$$J(u) = \int_0^T \left[ \frac{q}{2}(1-r^2(\theta)) + \frac{1}{2}u^T R u \right] dt + \frac{q_T}{2}(1-r^2(\theta(T)))$$

where:
- `r(θ) = |1/N ∑ exp(iθ_j)|` = order parameter
- `q, q_T > 0` = coherence weights
- `R = diag(ρ)` = control penalty matrix

## SMC Design

**Sliding Surface:**
$$s_i = e_i = \theta_i - \bar{\theta}$$

**Control Law:**
$$u_i = \text{sat}_{[-u_{max}, u_{max}]}\left( -K_1 e_i - K_2 \text{sign}(e_i) \right)$$

**Lyapunov Function:**
$$V = \frac{1}{2}\sum_{i=1}^N s_i^2$$

Robustness properties:
- Finite-time convergence
- Disturbance rejection
- Bounded control effort

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N` | 10 | Number of oscillators |
| `T` | 12.0 s | Finite horizon |
| `dt` | 0.01 s | Grid step |
| `q` | 1.0 | Running coherence weight |
| `q_T` | 10.0 | Terminal coherence weight |
| `ρ` | 0.25 | Control penalty (per oscillator) |
| `u_min, u_max` | -1, +1 | Box constraints |
| `K1_smc` | 1.5 | SMC proportional gain |
| `K2_smc` | 0.8 | SMC discontinuous gain |

## Usage

### Run SMC standalone
```matlab
run kuramoto_smc.m
```

### Run full comparison
```matlab
run kuramoto_pmp_fbs_ode89_compare.m
```

## Expected Results

**Order Parameter Evolution:**
- PMP: Rapid convergence, r → 0.95+ by t ≈ 3-4 s
- SMC: Fast convergence, comparable or slightly slower than PMP
- Feedback: Moderate convergence, depends on tuned gain
- Uncontrolled: Oscillates, rarely reaches full synchrony

**Cost Comparison:**
- PMP: Lowest `J` (first-order optimal)
- SMC: Near-optimal `J`, more robust
- Feedback: Higher `J`, simpler to implement
- Uncontrolled: Highest `J` (baseline)

**Control Effort:**
- PMP: Smooth control, full saturation utilization
- SMC: Discontinuous with saturation
- Feedback: Continuous and smooth
- Uncontrolled: Zero control

## Visualization

Generated figures:
1. Order parameter comparison
2. RMS control effort
3. Control bounds utilization (infinity norm)
4. Polar phase snapshots (all controllers, multiple times)
5. Individual control signals (4 subplots)
6. Cost breakdown (coherence vs. energy)

## Numerical Methods

- **ODE Solver:** `ode89` (high-order adaptive Runge-Kutta)
- **Integration:** RK4 in standalone SMC script
- **Interpolation:** Linear (forward) and previous (ZOH for control)
- **Tolerances:** RelTol = 1e-7, AbsTol = 1e-9
- **Convergence:** Relative Frobenius norm < 1e-3

## References

1. **Kuramoto Model:** Y. Kuramoto, "Chemical Oscillations, Waves, and Turbulence", Springer (1984)
2. **Sliding Mode Control:** J.-J. Slotine and W. Li, "Applied Nonlinear Control", Prentice Hall (1991)
3. **Optimal Control (PMP):** L. S. Pontryagin et al., "The Mathematical Theory of Optimal Processes", Wiley (1962)
4. **Forward-Backward Sweep:** A. J. Krener, "The high order maximal principle and its application to singular extremals", SIAM J. Control (1977)

## Author Notes

- Both scripts use reproducible random seeds (`rng(7)`) for consistent comparisons
- SMC implementation uses a smoothing parameter `ε = 0.01` to reduce chattering while maintaining control effectiveness
- PMP uses relaxation (`α = 0.2`) in FBS iterations to improve convergence
- All controllers respect box constraints on control effort
- Performance metrics are computed on the same cost functional for fair comparison

## Future Extensions

- Extend to complex networks (scale-free, small-world topologies)
- Add communication delays and noise robustness analysis
- Implement model predictive control (MPC) for comparison
- Analyze phase-locked patterns and chimera states
- Real-time hardware implementation (embedded systems)

---

**Last Updated:** June 2026
