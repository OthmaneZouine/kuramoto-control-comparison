%% Sliding Mode Control for Kuramoto Model Synchronization
% This script implements SMC for synchronizing N coupled oscillators
% Kuramoto dynamics with sliding mode control law

clear all; close all; clc;

%% ========== SYSTEM PARAMETERS ==========
N = 10;                        % Number of oscillators
K = 2.0;                       % Coupling strength
dt = 0.01;                     % Time step
t_final = 50;                  % Final simulation time
t = 0:dt:t_final;             % Time vector
num_steps = length(t);

% Natural frequencies (heterogeneous)
omega = 1 + 0.3*randn(N, 1);   % Random frequencies around 1 rad/s

%% ========== SMC PARAMETERS ==========
K1 = 1.5;                      % SMC gain 1 (proportional)
K2 = 0.8;                      % SMC gain 2 (discontinuous)
lambda = 0.5;                  % Sliding surface gain (optional)

%% ========== INITIAL CONDITIONS ==========
% Random initial phases
theta0 = 2*pi*rand(N, 1);
theta = zeros(N, num_steps);
theta(:, 1) = theta0;

% Control input history
u_history = zeros(N, num_steps);

%% ========== COUPLING MATRIX ==========
% Fully connected network (can be modified for other topologies)
A = ones(N, N) - eye(N);       % Adjacency matrix (exclude self-loops)

%% ========== SIMULATION ==========
for k = 1:num_steps-1
    % Current phase angles
    theta_k = theta(:, k);
    
    % Average phase (order parameter reference)
    theta_bar = mean(theta_k);
    
    % Synchronization error
    e = theta_k - theta_bar;
    
    % ===== SLIDING SURFACE DESIGN =====
    % Simple sliding surface: s_i = e_i
    s = e;
    
    % ===== SMC CONTROL LAW =====
    % u_i = -K1*s_i - K2*sign(s_i)
    % sign function with small epsilon to avoid chattering
    epsilon = 0.01;
    sign_s = s ./ (abs(s) + epsilon);
    u = -K1 * s - K2 * sign_s;
    
    % Store control input
    u_history(:, k) = u;
    
    % ===== KURAMOTO DYNAMICS WITH SMC =====
    % Phase differences for coupling
    phase_diff = repmat(theta_k, 1, N) - repmat(theta_k', N, 1);
    
    % Sine coupling terms
    coupling = K * A .* sin(phase_diff);
    
    % Total phase derivative
    dtheta = omega + sum(coupling, 2) + u;
    
    % Update phases using RK4 method (more accurate)
    theta(:, k+1) = RK4_step(theta_k, dtheta, dt, @kuramoto_dynamics, ...
                             omega, K, A, u, K1, K2, epsilon);
end

%% ========== ANALYSIS ==========
% Order parameter (synchronization metric)
r = zeros(1, num_steps);
psi = zeros(1, num_steps);

for k = 1:num_steps
    % Order parameter: r*exp(i*psi) = (1/N)*sum(exp(i*theta_k))
    r(k) = abs(mean(exp(1i*theta(:, k))));
    psi(k) = angle(mean(exp(1i*theta(:, k))));
end

% Phase velocity
dtheta = diff(theta, 1, 2) / dt;
dtheta_avg = mean(dtheta, 1);

%% ========== VISUALIZATION ==========
figure('Position', [100, 100, 1400, 900]);

% Plot 1: Phase evolution
subplot(2, 3, 1);
plot(t, theta', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Phase (rad)', 'FontSize', 11);
title('Phase Evolution of All Oscillators', 'FontSize', 12, 'FontWeight', 'bold');
grid on; legend(arrayfun(@(i) sprintf('Osc %d', i), 1:N, 'UniformOutput', false), ...
               'NumColumns', 2, 'FontSize', 8);

% Plot 2: Order parameter (synchronization metric)
subplot(2, 3, 2);
plot(t, r, 'LineWidth', 2.5, 'Color', [0 0.45 0.74]);
hold on;
plot(t, 0.95*ones(size(t)), '--r', 'LineWidth', 1.5, 'DisplayName', 'Target (r ≈ 0.95)');
xlabel('Time (s)', 'FontSize', 11);
ylabel('Order Parameter r', 'FontSize', 11);
title('Synchronization Progress (Order Parameter)', 'FontSize', 12, 'FontWeight', 'bold');
grid on; legend('FontSize', 10); ylim([0 1.05]);

% Plot 3: Phase velocity convergence
subplot(2, 3, 3);
plot(t(1:end-1), dtheta_avg(1:end-1), 'LineWidth', 2, 'Color', [0.47 0.67 0.19]);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Mean Phase Velocity (rad/s)', 'FontSize', 11);
title('Phase Velocity Convergence', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Plot 4: Synchronization error
subplot(2, 3, 4);
sync_error = std(theta, 1, 1);  % Standard deviation of phases
semilogy(t, sync_error, 'LineWidth', 2.5, 'Color', [0.85 0.33 0.1]);
xlabel('Time (s)', 'FontSize', 11);
ylabel('Phase Standard Deviation', 'FontSize', 11);
title('Synchronization Error (log scale)', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Plot 5: Control input magnitude
subplot(2, 3, 5);
u_magnitude = sqrt(sum(u_history.^2, 1));
plot(t, u_magnitude, 'LineWidth', 2, 'Color', [0.64 0.08 0.18]);
xlabel('Time (s)', 'FontSize', 11);
ylabel('||u|| (Control Magnitude)', 'FontSize', 11);
title('Control Input Magnitude', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Plot 6: Phase portrait (phase plane at final time)
subplot(2, 3, 6);
theta_final = theta(:, end);
theta_bar_final = mean(theta_final);
e_final = theta_final - theta_bar_final;
scatter(e_final, dtheta(:, end), 100, 1:N, 'filled', 'o', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
xlabel('Phase Error e (rad)', 'FontSize', 11);
ylabel('Phase Velocity (rad/s)', 'FontSize', 11);
title('Phase Space (Final State)', 'FontSize', 12, 'FontWeight', 'bold');
grid on; colorbar; hold on; plot([0 0], ylim, '--r', 'LineWidth', 1.5);

sgtitle('Sliding Mode Control of Kuramoto Model', 'FontSize', 14, 'FontWeight', 'bold');

%% ========== PERFORMANCE METRICS ==========
fprintf('\n');
fprintf('==================== SMC KURAMOTO SYNCHRONIZATION ====================\n');
fprintf('System Parameters:\n');
fprintf('  Number of oscillators: %d\n', N);
fprintf('  Coupling strength K: %.2f\n', K);
fprintf('  Simulation time: %.1f s\n', t_final);
fprintf('\nSMC Parameters:\n');
fprintf('  K1 (Proportional gain): %.2f\n', K1);
fprintf('  K2 (Discontinuous gain): %.2f\n', K2);
fprintf('\nInitial Conditions:\n');
fprintf('  Mean frequency: %.3f rad/s\n', mean(omega));
fprintf('  Frequency std: %.3f rad/s\n', std(omega));
fprintf('  Initial order parameter r(0): %.4f\n', r(1));

% Synchronization achieved check
final_r = r(end);
final_error = sync_error(end);
fprintf('\nFinal Results (t = %.1f s):\n', t_final);
fprintf('  Final order parameter r: %.4f\n', final_r);
fprintf('  Phase std deviation: %.4e rad\n', final_error);
fprintf('  Mean phase velocity: %.6f rad/s\n', mean(dtheta_avg(round(0.8*num_steps):end)));

if final_r > 0.95
    fprintf('\n  ✓ SYNCHRONIZATION ACHIEVED (r > 0.95)\n');
else
    fprintf('\n  ✗ Synchronization not yet achieved (r < 0.95)\n');
end
fprintf('========================================================================\n\n');

%% ========== HELPER FUNCTIONS ==========

function theta_next = RK4_step(theta_k, dtheta, dt, kuramoto_func, omega, K, A, u, K1, K2, epsilon)
    % 4th-order Runge-Kutta integration step
    
    k1 = kuramoto_func(theta_k, omega, K, A, u, K1, K2, epsilon);
    k2 = kuramoto_func(theta_k + 0.5*dt*k1, omega, K, A, u, K1, K2, epsilon);
    k3 = kuramoto_func(theta_k + 0.5*dt*k2, omega, K, A, u, K1, K2, epsilon);
    k4 = kuramoto_func(theta_k + dt*k3, omega, K, A, u, K1, K2, epsilon);
    
    theta_next = theta_k + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);
end

function dtheta = kuramoto_dynamics(theta, omega, K, A, u, K1, K2, epsilon)
    % Kuramoto model with SMC control
    N = length(theta);
    
    % Phase differences for coupling
    phase_diff = repmat(theta, 1, N) - repmat(theta', N, 1);
    
    % Sine coupling terms
    coupling = K * A .* sin(phase_diff);
    
    % Synchronization error
    theta_bar = mean(theta);
    e = theta - theta_bar;
    
    % SMC control law
    sign_e = e ./ (abs(e) + epsilon);
    u_ctrl = -K1 * e - K2 * sign_e;
    
    % Total phase derivative
    dtheta = omega + sum(coupling, 2) + u_ctrl;
end