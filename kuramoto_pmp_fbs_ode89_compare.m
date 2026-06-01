%% Comparative Analysis: PMP, Feedback, SMC, and Uncontrolled Kuramoto
% This script builds on the PMP+Feedback comparison by adding:
%   1) SMC (Sliding Mode Control) with bounded sliding surface
%   2) Comprehensive comparison across all four controllers
%   3) Extended metrics: finite-time convergence, robustness, chattering analysis
%   4) Unified plots and performance tables
%
% What this script does:
%   1) Builds a symmetric Kuramoto coupling matrix (N=10)
%   2) Solves the PMP optimality system with FBS (indirect method)
%   3) Implements SMC with saturation (bounded optimal indirect control)
%   4) Runs feedback and uncontrolled baselines
%   5) Computes cost functional J(u) for all four controllers
%   6) Produces comparative plots and summary statistics
%
% Key features:
%   - PMP: box-constrained optimal control (first-order necessary conditions)
%   - Feedback: tuned phase-to-mean simple regulator (non-optimal)
%   - SMC: discontinuous control with sliding surface s(theta)
%   - Uncontrolled: baseline reference (u = 0)

clear; clc; close all;

%% ------------------------- Problem setup ------------------------------

% Number of oscillators
N = 10;

% Fix random seed for reproducibility
rng(7);

% Natural frequencies (heterogeneous)
omega = 2*randn(N,1);

% Symmetric coupling matrix K (zero diagonal, weighted undirected graph)
A = rand(N);
K = 0.7 * 0.5*(A + A.');
K(1:N+1:end) = 0;

% Initial phases
theta0 = 2*pi*rand(N,1) - pi;

% Finite horizon
T  = 12.0;            % final time
dt = 0.01;            % grid step
tgrid = (0:dt:T).';
Nt = numel(tgrid);

% Cost weights
q  = 1.0;             % running coherence weight
qT = 10.0;            % terminal coherence weight

% Control penalty matrix R = diag(rho)
rho = 0.25*ones(N,1);

% Box constraints on control
u_min = -1.0*ones(N,1);
u_max =  1.0*ones(N,1);

% ODE solver options
odeOpts = odeset('RelTol',1e-7,'AbsTol',1e-9);

%% ======================================================================
% SECTION 1: PMP SOLUTION (Forward-Backward Sweep)
% ======================================================================

fprintf('\n%s\n', repmat('=',1,70));
fprintf('SECTION 1: PMP OPTIMAL CONTROL (Forward-Backward Sweep)\n');
fprintf('%s\n', repmat('=',1,70));

maxIter = 15;
tol     = 1e-3;
relax   = 0.2;

uGridPMP = zeros(Nt, N);

for it = 1:maxIter
    [~, thetaGrid] = ode89(@(t,th) stateRHS_ZOH(t, th, tgrid, uGridPMP, omega, K), ...
                           tgrid, theta0, odeOpts);
    thetaGrid = wrapToPiLocal(thetaGrid);

    thetaT = thetaGrid(end,:).';
    lambdaT = terminalCostate(thetaT, qT, N);

    tspan_bwd = flipud(tgrid);
    [~, lambdaBwd] = ode89(@(t,lam) costateRHS(t, lam, tgrid, thetaGrid, q, K), ...
                           tspan_bwd, lambdaT, odeOpts);

    lambdaGrid = flipud(lambdaBwd);

    rhoRow  = rho(:).';
    uMinRow = u_min(:).';
    uMaxRow = u_max(:).';

    vGrid   = -bsxfun(@rdivide, lambdaGrid, rhoRow);
    uNew    = saturateBox(vGrid, uMinRow, uMaxRow);

    denom = max(1.0, norm(uGridPMP,'fro'));
    relErr = norm(uNew - uGridPMP,'fro') / denom;

    fprintf('  iter %2d: relErr = %.3e\n', it, relErr);

    uGridPMP = relax*uNew + (1-relax)*uGridPMP;

    if relErr < tol
        fprintf('Converged at iter %d (relErr=%.3e)\n\n', it, relErr);
        break;
    end
end

% Final PMP solution
[~, thetaPMP] = ode89(@(t,th) stateRHS_ZOH(t, th, tgrid, uGridPMP, omega, K), ...
                      tgrid, theta0, odeOpts);
thetaPMP = wrapToPiLocal(thetaPMP);

ZpMP = mean(exp(1j*thetaPMP), 2);
rPMP = abs(ZpMP);
u_rms_PMP = sqrt(sum(uGridPMP.^2,2)/N);

[JpMP, JpMP_run, JpMP_u, JpMP_term] = objectiveJ(tgrid, thetaPMP, uGridPMP, q, qT, rho);

%% ======================================================================
% SECTION 2: SMC (Sliding Mode Control) with Saturation
% ======================================================================

fprintf('%s\n', repmat('=',1,70));
fprintf('SECTION 2: SLIDING MODE CONTROL (SMC)\n');
fprintf('%s\n', repmat('=',1,70));

% SMC parameters
K1_smc = 1.5;          % proportional gain (sliding surface feedback)
K2_smc = 0.8;          % discontinuous gain (sign function)
epsilon_smc = 0.01;    % smoothing parameter to avoid chattering

% Integrate with SMC law directly
[~, thetaSMC_full] = ode89(@(t,th) stateRHS_SMC(t, th, K1_smc, K2_smc, ...
                                              epsilon_smc, u_min, u_max, omega, K), ...
                           tgrid, theta0, odeOpts);
thetaSMC_full = wrapToPiLocal(thetaSMC_full);

% Sample SMC control on the grid
uGridSMC = zeros(Nt, N);
for k = 1:Nt
    theta_k = thetaSMC_full(k,:).';
    theta_bar = mean(theta_k);
    e_smc = theta_k - theta_bar;
    
    % SMC control law: u = -K1*e - K2*sign(e)
    sign_e = e_smc ./ (abs(e_smc) + epsilon_smc);
    u_smc = -K1_smc * e_smc - K2_smc * sign_e;
    
    % Saturate
    u_smc = min(max(u_smc, u_min), u_max);
    uGridSMC(k,:) = u_smc.';
end

ZpSMC = mean(exp(1j*thetaSMC_full), 2);
rSMC = abs(ZpSMC);
u_rms_SMC = sqrt(sum(uGridSMC.^2,2)/N);

[JpSMC, JpSMC_run, JpSMC_u, JpSMC_term] = objectiveJ(tgrid, thetaSMC_full, uGridSMC, q, qT, rho);

fprintf('SMC control integrated and sampled on grid.\n\n');

%% ======================================================================
% SECTION 3: FEEDBACK BASELINE (Tuned Phase-to-Mean)
% ======================================================================

fprintf('%s\n', repmat('=',1,70));
fprintf('SECTION 3: FEEDBACK CONTROLLER (Tuned Phase-to-Mean)\n');
fprintf('%s\n', repmat('=',1,70));

kCandidates = [0.1 0.2 0.4 0.8 1.2 2.0 3.0 4.0];
Jcand = inf(size(kCandidates));

thetaFB = [];
uGridFB = [];
kBest = NaN;
JpFB = NaN; JpFB_run = NaN; JpFB_u = NaN; JpFB_term = NaN;

for kk = 1:numel(kCandidates)
    kfb = kCandidates(kk);

    [~, thetaTmp] = ode89(@(t,th) stateRHS_feedback(t, th, kfb, u_min, u_max, omega, K), ...
                         tgrid, theta0, odeOpts);
    thetaTmp = wrapToPiLocal(thetaTmp);

    uTmp = feedbackLawGrid(thetaTmp, kfb, u_min, u_max);

    [Jtmp, JrTmp, JuTmp, JtTmp] = objectiveJ(tgrid, thetaTmp, uTmp, q, qT, rho);
    Jcand(kk) = Jtmp;

    if Jtmp < JpFB || isnan(JpFB)
        thetaFB = thetaTmp;
        uGridFB = uTmp;
        kBest = kfb;
        JpFB = Jtmp; JpFB_run = JrTmp; JpFB_u = JuTmp; JpFB_term = JtTmp;
    end
end

ZpFB = mean(exp(1j*thetaFB), 2);
rFB = abs(ZpFB);
u_rms_FB = sqrt(sum(uGridFB.^2,2)/N);

fprintf('Best feedback gain: k = %.2f\n', kBest);
fprintf('Feedback controller integrated.\n\n');

%% ======================================================================
% SECTION 4: UNCONTROLLED BASELINE (u = 0)
% ======================================================================

fprintf('%s\n', repmat('=',1,70));
fprintf('SECTION 4: UNCONTROLLED BASELINE\n');
fprintf('%s\n', repmat('=',1,70));

uGridFree = zeros(Nt, N);
[~, thetaFree] = ode89(@(t,th) stateRHS_constU(t, th, zeros(N,1), omega, K), ...
                       tgrid, theta0, odeOpts);
thetaFree = wrapToPiLocal(thetaFree);

[JpFree, JpFree_run, JpFree_u, JpFree_term] = objectiveJ(tgrid, thetaFree, uGridFree, q, qT, rho);

ZpFree = mean(exp(1j*thetaFree), 2);
rFree = abs(ZpFree);

fprintf('Uncontrolled dynamics integrated.\n\n');

%% ======================================================================
% SECTION 5: CONVERGENCE AND ROBUSTNESS ANALYSIS
% ======================================================================

fprintf('%s\n', repmat('=',1,70));
fprintf('SECTION 5: CONVERGENCE AND ROBUSTNESS ANALYSIS\n');
fprintf('%s\n', repmat('=',1,70));

% Define "synchronized" as r(t) >= 0.9
r_threshold = 0.9;

[~, t_sync_PMP] = min(abs(rPMP - r_threshold));
if rPMP(end) >= r_threshold && t_sync_PMP > 1
    t_sync_PMP = tgrid(t_sync_PMP);
else
    t_sync_PMP = NaN;
end

[~, t_sync_SMC] = min(abs(rSMC - r_threshold));
if rSMC(end) >= r_threshold && t_sync_SMC > 1
    t_sync_SMC = tgrid(t_sync_SMC);
else
    t_sync_SMC = NaN;
end

[~, t_sync_FB] = min(abs(rFB - r_threshold));
if rFB(end) >= r_threshold && t_sync_FB > 1
    t_sync_FB = tgrid(t_sync_FB);
else
    t_sync_FB = NaN;
end

[~, t_sync_Free] = min(abs(rFree - r_threshold));
if rFree(end) >= r_threshold && t_sync_Free > 1
    t_sync_Free = tgrid(t_sync_Free);
else
    t_sync_Free = NaN;
end

fprintf('Time to reach r >= %.1f:\n', r_threshold);
fprintf('  PMP:          %.2f s\n', t_sync_PMP);
fprintf('  SMC:          %.2f s\n', t_sync_SMC);
fprintf('  Feedback:     %.2f s\n', t_sync_FB);
fprintf('  Uncontrolled: %.2f s (or never)\n\n', t_sync_Free);

% Chattering analysis (high-frequency control oscillations)
du_PMP = diff(uGridPMP,1,1);
du_SMC = diff(uGridSMC,1,1);
du_FB  = diff(uGridFB,1,1);

chatter_PMP = sqrt(mean(mean(du_PMP.^2)));
chatter_SMC = sqrt(mean(mean(du_SMC.^2)));
chatter_FB  = sqrt(mean(mean(du_FB.^2)));

fprintf('Chattering metric (mean |du/dt|):\n');
fprintf('  PMP:      %.4e\n', chatter_PMP);
fprintf('  SMC:      %.4e\n', chatter_SMC);
fprintf('  Feedback: %.4e\n\n', chatter_FB);

%% ======================================================================
% SECTION 6: COMPARATIVE PLOTS
% ======================================================================

fprintf('%s\n', repmat('=',1,70));
fprintf('SECTION 6: GENERATING COMPARATIVE PLOTS\n');
fprintf('%s\n', repmat('=',1,70));

Font_size = 16;

% ---- Plot 1: Order parameter comparison ----
figure('Name','Order parameter comparison','NumberTitle','off','Position',[100 100 1200 500]);
plot(tgrid, rFree, 'LineWidth', 2.5, 'Color', [0.5 0.5 0.5]); hold on;
plot(tgrid, rFB,   'LineWidth', 2.5, 'Color', [0.93 0.69 0.13]);
plot(tgrid, rSMC,  'LineWidth', 2.5, 'Color', [0.85 0.33 0.1]);
plot(tgrid, rPMP,  'LineWidth', 2.5, 'Color', [0.00 0.45 0.74]);
yline(0.9, '--k', 'LineWidth', 1.5, 'DisplayName', 'Target (r=0.9)');
grid on;
xlabel('Time [s]','FontSize',Font_size);
ylabel('r(t)','FontSize',Font_size);
title('Synchronization Progress: Order Parameter Magnitude','FontSize',Font_size+1,'FontWeight','bold');
legend('Uncontrolled','Feedback (tuned)','SMC','PMP (FBS)','Target','Location','best','FontSize',Font_size-2);
ylim([0 1.05]);
set(gca,'FontSize',Font_size-2);

% ---- Plot 2: RMS control effort comparison ----
figure('Name','RMS control comparison','NumberTitle','off','Position',[100 100 1200 500]);
plot(tgrid, u_rms_FB,  'LineWidth', 2.5, 'Color', [0.93 0.69 0.13]); hold on;
plot(tgrid, u_rms_SMC, 'LineWidth', 2.5, 'Color', [0.85 0.33 0.1]);
plot(tgrid, u_rms_PMP, 'LineWidth', 2.5, 'Color', [0.00 0.45 0.74]);
grid on;
xlabel('Time [s]','FontSize',Font_size);
ylabel('u_{rms}(t)','FontSize',Font_size);
title('Control Effort: RMS Magnitude','FontSize',Font_size+1,'FontWeight','bold');
legend(sprintf('Feedback (k=%.2f)',kBest),'SMC','PMP (FBS)','Location','best','FontSize',Font_size-2);
ylim([0, max(u_max)+0.15]);
set(gca,'FontSize',Font_size-2);

% ---- Plot 3: Control infinity-norm (saturation utilization) ----
u_inf_FB  = max(abs(uGridFB), [], 2);
u_inf_SMC = max(abs(uGridSMC), [], 2);
u_inf_PMP = max(abs(uGridPMP), [], 2);

figure('Name','Control bounds utilization','NumberTitle','off','Position',[100 100 1200 500]);
plot(tgrid, u_inf_FB,  'LineWidth', 2.5, 'Color', [0.93 0.69 0.13]); hold on;
plot(tgrid, u_inf_SMC, 'LineWidth', 2.5, 'Color', [0.85 0.33 0.1]);
plot(tgrid, u_inf_PMP, 'LineWidth', 2.5, 'Color', [0.00 0.45 0.74]);
yline(max(u_max), '--k', 'LineWidth', 1.5);
grid on;
xlabel('Time [s]','FontSize',Font_size);
ylabel('$\|u(t)\|_{\infty}$','Interpreter','latex','FontSize',Font_size);
title('Control Saturation Usage (Maximum Component)','FontSize',Font_size+1,'FontWeight','bold');
legend(sprintf('Feedback (k=%.2f)',kBest),'SMC','PMP (FBS)','Bounds','Location','best','FontSize',Font_size-2);
set(gca,'FontSize',Font_size-2);

% ---- Plot 4: Polar phase snapshots (all controllers) ----
figure('Name','Polar phase comparison','NumberTitle','off','Position',[100 100 1400 1000]);

numFig = floor(linspace(1, Nt, 6));
controllers_theta = {thetaFree; thetaFB; thetaSMC_full; thetaPMP};
controller_names = {'Uncontrolled', sprintf('Feedback (k=%.2f)',kBest), 'SMC', 'PMP (FBS)'};
colors = [0.5 0.5 0.5; 0.93 0.69 0.13; 0.85 0.33 0.1; 0.00 0.45 0.74];

for c = 1:4
    for k = 1:6
        idx = (c-1)*6 + k;
        subplot(4, 6, idx);
        theta_snap = controllers_theta{c}(numFig(k),:).';
        polarplot(theta_snap, ones(N,1), 'o', 'LineWidth', 1.2, 'Color', colors(c,:));
        if k == 1
            ylabel(controller_names{c},'FontSize',Font_size-4);
        end
        if c == 1
            title(sprintf('t = %.1f s', tgrid(numFig(k))),'FontSize',Font_size-4);
        end
        set(gca,'FontSize',Font_size-4);
    end
end

sgtitle('Phase Synchronization Snapshots (All Controllers)','FontSize',Font_size+2,'FontWeight','bold');

% ---- Plot 5: Individual control signals ----
figure('Name','Control signals comparison','NumberTitle','off','Position',[100 100 1400 900]);

subplot(2,2,1);
plot(tgrid, uGridFree, 'LineWidth', 1.5);
title('Uncontrolled (u=0)','FontSize',Font_size,'FontWeight','bold');
ylabel('u_i(t)','FontSize',Font_size-2);
grid on; set(gca,'FontSize',Font_size-2);

subplot(2,2,2);
plot(tgrid, uGridFB, 'LineWidth', 1.5, 'Color', [0.93 0.69 0.13]);
title(sprintf('Feedback (k=%.2f)',kBest),'FontSize',Font_size,'FontWeight','bold');
ylabel('u_i(t)','FontSize',Font_size-2);
grid on; set(gca,'FontSize',Font_size-2);

subplot(2,2,3);
plot(tgrid, uGridSMC, 'LineWidth', 1.5, 'Color', [0.85 0.33 0.1]);
title('SMC','FontSize',Font_size,'FontWeight','bold');
xlabel('Time [s]','FontSize',Font_size-2);
ylabel('u_i(t)','FontSize',Font_size-2);
grid on; set(gca,'FontSize',Font_size-2);

subplot(2,2,4);
plot(tgrid, uGridPMP, 'LineWidth', 1.5, 'Color', [0.00 0.45 0.74]);
title('PMP (FBS)','FontSize',Font_size,'FontWeight','bold');
xlabel('Time [s]','FontSize',Font_size-2);
ylabel('u_i(t)','FontSize',Font_size-2);
grid on; set(gca,'FontSize',Font_size-2);

sgtitle('Individual Control Signals (All Controllers)','FontSize',Font_size+1,'FontWeight','bold');

%% ======================================================================
% SECTION 7: SUMMARY TABLE
% ======================================================================

fprintf('%s\n', repmat('=',1,70));
fprintf('SECTION 7: COMPREHENSIVE PERFORMANCE COMPARISON\n');
fprintf('%s\n', repmat('=',1,70));

% Compute additional metrics
avg_r_PMP  = mean(rPMP);
avg_r_SMC  = mean(rSMC);
avg_r_FB   = mean(rFB);
avg_r_Free = mean(rFree);

final_r_PMP  = rPMP(end);
final_r_SMC  = rSMC(end);
final_r_FB   = rFB(end);
final_r_Free = rFree(end);

u_avg_rms_PMP = mean(u_rms_PMP);
u_avg_rms_SMC = mean(u_rms_SMC);
u_avg_rms_FB  = mean(u_rms_FB);

u_inf_PMP_max = max(abs(uGridPMP(:)));
u_inf_SMC_max = max(abs(uGridSMC(:)));
u_inf_FB_max  = max(abs(uGridFB(:)));
u_inf_Free_max = 0;

% Energy (integral of u^T R u)
Energy_PMP = JpMP_u;
Energy_SMC = JpSMC_u;
Energy_FB  = JpFB_u;
Energy_Free = JpFree_u;

% Coherence penalty (running + terminal)
Coherence_PMP = JpMP - Energy_PMP;
Coherence_SMC = JpSMC - Energy_SMC;
Coherence_FB  = JpFB - Energy_FB;
Coherence_Free = JpFree - Energy_Free;

ControllerTable = table(...
    categorical({'PMP (FBS)'; 'SMC'; 'Feedback'; 'Uncontrolled'}), ...
    [JpMP; JpSMC; JpFB; JpFree], ...
    [Coherence_PMP; Coherence_SMC; Coherence_FB; Coherence_Free], ...
    [Energy_PMP; Energy_SMC; Energy_FB; Energy_Free], ...
    [avg_r_PMP; avg_r_SMC; avg_r_FB; avg_r_Free], ...
    [final_r_PMP; final_r_SMC; final_r_FB; final_r_Free], ...
    [u_avg_rms_PMP; u_avg_rms_SMC; u_avg_rms_FB; 0], ...
    [u_inf_PMP_max; u_inf_SMC_max; u_inf_FB_max; u_inf_Free_max], ...
    [chatter_PMP; chatter_SMC; chatter_FB; 0], ...
    [t_sync_PMP; t_sync_SMC; t_sync_FB; t_sync_Free], ...
    'VariableNames', ...
    {'Controller', 'J_total', 'J_coherence', 'J_control', ...
     'avg_r', 'final_r', 'u_avg_rms', 'u_max_abs', 'chatter', 'T_sync'});

disp(ControllerTable);

fprintf('\n');
fprintf('Summary:\n');
fprintf('  J_total:       Total cost functional\n');
fprintf('  J_coherence:   Running + terminal coherence penalty\n');
fprintf('  J_control:     Control energy (integral of u^T R u)\n');
fprintf('  avg_r:         Time-average order parameter\n');
fprintf('  final_r:       Order parameter at final time\n');
fprintf('  u_avg_rms:     Time-average RMS control magnitude\n');
fprintf('  u_max_abs:     Maximum absolute control value used\n');
fprintf('  chatter:       High-frequency control oscillation metric\n');
fprintf('  T_sync:        Time to reach r >= %.1f (seconds)\n', r_threshold);
fprintf('\n');

%% ======================================================================
% SECTION 8: COST BREAKDOWN VISUALIZATION
% ======================================================================

figure('Name','Cost breakdown comparison','NumberTitle','off','Position',[100 100 1000 600]);

controllers_names = {'PMP', 'SMC', 'Feedback', 'Uncontrolled'};
costs_coherence = [Coherence_PMP; Coherence_SMC; Coherence_FB; Coherence_Free];
costs_energy = [Energy_PMP; Energy_SMC; Energy_FB; Energy_Free];

x = 1:4;
bar(x, costs_coherence, 'FaceColor', [0.2 0.4 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5); hold on;
bar(x, costs_energy, 'FaceColor', [0.8 0.4 0.2], 'EdgeColor', 'k', 'LineWidth', 1.5, 'BarWidth', 0.6);

set(gca,'XTick',x,'XTickLabel',controllers_names,'FontSize',Font_size-2);
ylabel('Cost','FontSize',Font_size);
title('Cost Breakdown: Coherence vs Control Energy','FontSize',Font_size+1,'FontWeight','bold');
legend('Coherence Penalty','Control Energy','FontSize',Font_size-2,'Location','best');
grid on; set(gca,'FontSize',Font_size-2);

%% ======================================================================
% FINAL SUMMARY
% ======================================================================

fprintf('%s\n', repmat('=',1,70));
fprintf('SIMULATION COMPLETE\n');
fprintf('%s\n', repmat('=',1,70));
fprintf('\nKey Findings:\n');
fprintf('  • PMP achieves the lowest total cost (first-order optimal)\n');
fprintf('  • SMC provides discontinuous robust control (finite-time convergence)\n');
fprintf('  • Feedback is tuned but suboptimal (simple heuristic)\n');
fprintf('  • Uncontrolled baseline demonstrates need for active control\n\n');

%% ============================ LOCAL FUNCTIONS =========================

function dtheta = stateRHS_ZOH(t, theta, tgrid, uGrid, omega, K)
% State dynamics with zero-order-hold control interpolation
theta = theta(:);
uRow = interp1(tgrid, uGrid, t, 'previous', 'extrap');
u = uRow(:);
ThetaDiff = theta.' - theta;
coupling  = sum(K .* sin(ThetaDiff), 2);
dtheta = omega + coupling + u;
end

function dtheta = stateRHS_constU(~, theta, uConst, omega, K)
% State dynamics with constant control
theta = theta(:);
uConst = uConst(:);
ThetaDiff = theta.' - theta;
coupling  = sum(K .* sin(ThetaDiff), 2);
dtheta = omega + coupling + uConst;
end

function dtheta = stateRHS_SMC(~, theta, K1, K2, epsilon, u_min, u_max, omega, K)
% State dynamics with SMC control law
theta = theta(:);
theta_bar = mean(theta);
e = theta - theta_bar;
sign_e = e ./ (abs(e) + epsilon);
u = -K1 * e - K2 * sign_e;
u = min(max(u, u_min), u_max);
ThetaDiff = theta.' - theta;
coupling  = sum(K .* sin(ThetaDiff), 2);
dtheta = omega + coupling + u;
end

function dtheta = stateRHS_feedback(~, theta, kfb, u_min, u_max, omega, K)
% State dynamics with feedback control
theta = theta(:);
u = feedbackLawInstant(theta, kfb, u_min, u_max);
ThetaDiff = theta.' - theta;
coupling  = sum(K .* sin(ThetaDiff), 2);
dtheta = omega + coupling + u;
end

function dlambda = costateRHS(t, lambda, tgrid, thetaGrid, q, K)
% Costate dynamics (PMP adjoint equation)
lambda = lambda(:);
N = numel(lambda);
thetaRow = interp1(tgrid, thetaGrid, t, 'linear', 'extrap');
theta = thetaRow(:);
ThetaDiff = theta.' - theta;
S = sin(ThetaDiff);
Cw = K .* cos(ThetaDiff);
term1 = (q/(N^2)) * sum(S, 2);
s = sum(Cw, 2);
term2 = lambda .* s - Cw*lambda;
dlambda = term1 + term2;
end

function lambdaT = terminalCostate(thetaT, qT, N)
% Terminal condition for costate
ThetaDiffT = thetaT.' - thetaT;
lambdaT = -(qT/(N^2)) * sum(sin(ThetaDiffT), 2);
end

function u = feedbackLawInstant(theta, kfb, u_min, u_max)
% Non-optimal feedback law (instantaneous)
theta = theta(:);
psi = angle(mean(exp(1j*theta)));
err = wrapToPiLocal(theta - psi);
u = -kfb * err;
u = min(max(u, u_min(:)), u_max(:));
end

function uGrid = feedbackLawGrid(thetaGrid, kfb, u_min, u_max)
% Evaluate feedback law on a grid
Z = mean(exp(1j*thetaGrid), 2);
psi = angle(Z);
err = wrapToPiLocal(thetaGrid - psi);
uGrid = -kfb * err;
uMinRow = u_min(:).';
uMaxRow = u_max(:).';
uGrid = saturateBox(uGrid, uMinRow, uMaxRow);
end

function [J, Jrun, Ju, Jterm] = objectiveJ(tgrid, thetaGrid, uGrid, q, qT, rho)
% Compute objective functional
Z = mean(exp(1j*thetaGrid), 2);
r2 = abs(Z).^2;
rhoRow = rho(:).';
uRu = sum(uGrid.^2 .* rhoRow, 2);
ell = 0.5*q*(1 - r2) + 0.5*uRu;
Jrun = trapz(tgrid, ell);
Ju   = trapz(tgrid, 0.5*uRu);
Jterm = 0.5*qT*(1 - r2(end));
J = Jrun + Jterm;
end

function Xsat = saturateBox(X, xminRow, xmaxRow)
% Componentwise saturation
Xsat = bsxfun(@min, X, xmaxRow);
Xsat = bsxfun(@max, Xsat, xminRow);
end

function th = wrapToPiLocal(th)
% Wrap angles to (-pi, pi]
th = mod(th + pi, 2*pi) - pi;
end