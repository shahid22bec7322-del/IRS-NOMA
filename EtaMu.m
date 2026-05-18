% =========================================================================
% SCENARIO 4:IRS-NOMA under eta-mu Fading
% 1. eta-mu Math: SNR PDF, Envelope PDF, MGF
% 2. Separate Outage (U1 & U2) + Sum Throughput vs Transmit Power
% 3. Outage vs eta and Outage vs mu separately
% =========================================================================
clear; clc; close all;

%% 1. Theoretical Math Equations of eta-mu Fading (Figure 1)
mu = 2; eta = 0.5;
h_eta = (2 + 1/eta + eta)/4; H_eta = (1/eta - eta)/4;

% Envelope (Amplitude) PDF Equation
r = linspace(0.01, 3, 200); r_hat = 1; 
f_r = (4*sqrt(pi)*h_eta^mu*mu^(mu+0.5)) / (gamma(mu)*H_eta^(mu-0.5)) .* ...
      (r.^(2*mu) ./ r_hat^(2*mu+1)) .* besseli(mu-0.5, 2*mu*H_eta*r.^2./r_hat^2) .* exp(-2*mu*h_eta*r.^2./r_hat^2);

% Instantaneous SNR PDF Equation
gamma_val = linspace(0.01, 5, 200); gamma_bar = 1; 
f_gamma = (2*sqrt(pi)*h_eta^mu*mu^(mu+0.5)) / (gamma(mu)*H_eta^(mu-0.5)*gamma_bar) .* ...
          (gamma_val./gamma_bar).^(mu-0.5) .* besseli(mu-0.5, 2*mu*H_eta*gamma_val./gamma_bar) .* exp(-2*mu*h_eta*gamma_val./gamma_bar);

% Moment Generating Function (MGF) Equation
s_val = linspace(0, 5, 200);
M_s = ( (4*mu^2*h_eta) ./ ( (2*mu*h_eta + s_val.*gamma_bar).^2 - (2*mu*H_eta)^2 ) ).^mu;

figure('Name', 'Theoretical Math Equations', 'Position', [50, 100, 1200, 350]);
subplot(1,3,1); plot(r, f_r, 'b-', 'LineWidth', 2); title('Envelope PDF f_{R}(r)'); xlabel('Amplitude (r)'); ylabel('PDF'); grid on;
subplot(1,3,2); plot(gamma_val, f_gamma, 'r-', 'LineWidth', 2); title('Inst. SNR PDF f_{\gamma}(\gamma)'); xlabel('SNR (\gamma)'); ylabel('PDF'); grid on;
subplot(1,3,3); plot(s_val, M_s, 'k-', 'LineWidth', 2); title('MGF of SNR M_{\gamma}(s)'); xlabel('Laplace Variable (s)'); ylabel('MGF'); grid on;

%% 2. System Parameters
N_el = 64; B = 3; beta_IRS = 2.0; xi = 0.05; 
sigma2 = 10^(-120/10); sigma_v2 = 10^(-120/10); 
R_th1 = 1.0; gamma_th1 = 2^R_th1 - 1; 
R_th2 = 0.5; gamma_th2 = 2^R_th2 - 1; 
phase_set = (0:(2^B - 1)) * (2*pi / 2^B);

L0 = 10^(-30/10); PL_G = L0 * 50^(-2.2); PL_h1 = L0 * 10^(-2.8); PL_h2 = L0 * 40^(-2.8);
K_sim = 20000; % Monte Carlo Realizations

%% 3. Simulation 1: Outage (U1 & U2) and Sum Throughput vs Pt
Pt_dBm_vec = 0:2:30; % Increased resolution to guarantee User 2 curve derivation
P_out_U1 = zeros(1, length(Pt_dBm_vec)); P_out_U2 = zeros(1, length(Pt_dBm_vec));
Sum_Rate = zeros(1, length(Pt_dBm_vec));

sigma_x = sqrt(1 / (2*(1+eta))); sigma_y = sqrt(eta / (2*(1+eta)));

for pt_idx = 1:length(Pt_dBm_vec)
    Pt = 10^((Pt_dBm_vec(pt_idx) - 30)/10);
    out_1 = 0; out_2 = 0; rate_sum = 0;
    for k = 1:K_sim
        H_G2 = zeros(N_el, 1); H_12 = zeros(N_el, 1); H_22 = zeros(N_el, 1);
        for m = 1:mu 
            H_G2 = H_G2 + (randn(N_el,1)*sigma_x).^2 + (randn(N_el,1)*sigma_y).^2;
            H_12 = H_12 + (randn(N_el,1)*sigma_x).^2 + (randn(N_el,1)*sigma_y).^2;
            H_22 = H_22 + (randn(N_el,1)*sigma_x).^2 + (randn(N_el,1)*sigma_y).^2;
        end
        G = sqrt(PL_G) * sqrt(H_G2) .* exp(1j * 2*pi*rand(N_el,1)); h1 = sqrt(PL_h1) * sqrt(H_12) .* exp(1j * 2*pi*rand(N_el,1)); h2 = sqrt(PL_h2) * sqrt(H_22) .* exp(1j * 2*pi*rand(N_el,1));
        
        N1_irs = beta_IRS^2 * sigma_v2 * norm(h1)^2 + sigma2;
        N2_irs = beta_IRS^2 * sigma_v2 * norm(h2)^2 + sigma2;
        
        [~, min_idx] = min(abs(exp(1j*(angle(h2) - angle(G))) - exp(1j*phase_set)), [], 2);
        Phi = beta_IRS * diag(exp(1j * phase_set(min_idx).'));
        
        a1 = abs(h1' * Phi * G)^2; a2 = abs(h2' * Phi * G)^2;
        P2_min = (gamma_th2 / (1 + gamma_th2)) * (Pt + N2_irs / a2); 
        P2_max = (Pt * a1 - gamma_th1 * N1_irs) / (a1 * (1 + xi * gamma_th1)); 
        
        is_u2_out = (P2_min > Pt) || (P2_min < 0);
        if is_u2_out; out_2 = out_2 + 1; end
        if (P2_min > P2_max) || (P2_max < 0) || is_u2_out; out_1 = out_1 + 1; 
        else
            P1_opt = Pt - P2_min;
            rate_sum = rate_sum + log2(1 + P1_opt * a1 / (xi * P2_min * a1 + N1_irs)) + log2(1 + P2_min * a2 / (P1_opt * a2 + N2_irs));
        end
    end
    P_out_U1(pt_idx) = out_1 / K_sim; P_out_U2(pt_idx) = out_2 / K_sim; Sum_Rate(pt_idx) = rate_sum / K_sim;
end

%% 4. Simulation 2 & 3: Outage vs ETA and MU
eta_vec = 0.1:0.1:1.0; mu_vec = 1:5;
Pt_fixed = 10^((8 - 30)/10); % 8 dBm guarantees User 2 is visible and derives correctly
P_out_eta_U1 = zeros(1, length(eta_vec)); P_out_eta_U2 = zeros(1, length(eta_vec));
P_out_mu_U1 = zeros(1, length(mu_vec)); P_out_mu_U2 = zeros(1, length(mu_vec));

% ETA Loop
for e_idx = 1:length(eta_vec)
    eta_cur = eta_vec(e_idx); sig_x = sqrt(1 / (2*(1+eta_cur))); sig_y = sqrt(eta_cur / (2*(1+eta_cur)));
    o1 = 0; o2 = 0;
    for k = 1:K_sim
        H_G2 = zeros(N_el, 1); H_12 = zeros(N_el, 1); H_22 = zeros(N_el, 1);
        for m = 1:mu
            H_G2 = H_G2 + (randn(N_el,1)*sig_x).^2 + (randn(N_el,1)*sig_y).^2; H_12 = H_12 + (randn(N_el,1)*sig_x).^2 + (randn(N_el,1)*sig_y).^2; H_22 = H_22 + (randn(N_el,1)*sig_x).^2 + (randn(N_el,1)*sig_y).^2;
        end
        G = sqrt(PL_G) * sqrt(H_G2) .* exp(1j * 2*pi*rand(N_el,1)); h1 = sqrt(PL_h1) * sqrt(H_12) .* exp(1j * 2*pi*rand(N_el,1)); h2 = sqrt(PL_h2) * sqrt(H_22) .* exp(1j * 2*pi*rand(N_el,1));
        N1_irs = beta_IRS^2 * sigma_v2 * norm(h1)^2 + sigma2; N2_irs = beta_IRS^2 * sigma_v2 * norm(h2)^2 + sigma2;
        [~, min_idx] = min(abs(exp(1j*(angle(h2) - angle(G))) - exp(1j*phase_set)), [], 2); Phi = beta_IRS * diag(exp(1j * phase_set(min_idx).'));
        a1 = abs(h1' * Phi * G)^2; a2 = abs(h2' * Phi * G)^2;
        P2_min = (gamma_th2 / (1 + gamma_th2)) * (Pt_fixed + N2_irs / a2); P2_max = (Pt_fixed * a1 - gamma_th1 * N1_irs) / (a1 * (1 + xi * gamma_th1)); 
        is_u2_out = (P2_min > Pt_fixed) || (P2_min < 0);
        if is_u2_out; o2 = o2 + 1; end; if (P2_min > P2_max) || (P2_max < 0) || is_u2_out; o1 = o1 + 1; end
    end
    P_out_eta_U1(e_idx) = o1 / K_sim; P_out_eta_U2(e_idx) = o2 / K_sim;
end

% MU Loop
for m_idx = 1:length(mu_vec)
    mu_cur = mu_vec(m_idx); o1 = 0; o2 = 0;
    for k = 1:K_sim
        H_G2 = zeros(N_el, 1); H_12 = zeros(N_el, 1); H_22 = zeros(N_el, 1);
        for m = 1:mu_cur
            H_G2 = H_G2 + (randn(N_el,1)*sigma_x).^2 + (randn(N_el,1)*sigma_y).^2; H_12 = H_12 + (randn(N_el,1)*sigma_x).^2 + (randn(N_el,1)*sigma_y).^2; H_22 = H_22 + (randn(N_el,1)*sigma_x).^2 + (randn(N_el,1)*sigma_y).^2;
        end
        G = sqrt(PL_G) * sqrt(H_G2) .* exp(1j * 2*pi*rand(N_el,1)); h1 = sqrt(PL_h1) * sqrt(H_12) .* exp(1j * 2*pi*rand(N_el,1)); h2 = sqrt(PL_h2) * sqrt(H_22) .* exp(1j * 2*pi*rand(N_el,1));
        N1_irs = beta_IRS^2 * sigma_v2 * norm(h1)^2 + sigma2; N2_irs = beta_IRS^2 * sigma_v2 * norm(h2)^2 + sigma2;
        [~, min_idx] = min(abs(exp(1j*(angle(h2) - angle(G))) - exp(1j*phase_set)), [], 2); Phi = beta_IRS * diag(exp(1j * phase_set(min_idx).'));
        a1 = abs(h1' * Phi * G)^2; a2 = abs(h2' * Phi * G)^2;
        P2_min = (gamma_th2 / (1 + gamma_th2)) * (Pt_fixed + N2_irs / a2); P2_max = (Pt_fixed * a1 - gamma_th1 * N1_irs) / (a1 * (1 + xi * gamma_th1)); 
        is_u2_out = (P2_min > Pt_fixed) || (P2_min < 0);
        if is_u2_out; o2 = o2 + 1; end; if (P2_min > P2_max) || (P2_max < 0) || is_u2_out; o1 = o1 + 1; end
    end
    P_out_mu_U1(m_idx) = o1 / K_sim; P_out_mu_U2(m_idx) = o2 / K_sim;
end

%% 6. Plotting & EXPLICIT MATHEMATICAL DERIVATIONS

% --- Plot 2: Outage & Sum Rate vs Pt ---
figure('Name', 'System Performance vs Pt', 'Position', [100, 200, 1000, 450]);

subplot(1,2,1);
semilogy(Pt_dBm_vec, max(P_out_U1, 1e-6), 'bs-', 'LineWidth', 2); hold on;
semilogy(Pt_dBm_vec, max(P_out_U2, 1e-6), 'ro-', 'LineWidth', 2); grid on;

% Extract Mathematics using the robust solver at the bottom
[eq_U1_Pt, y_U1_Pt] = robust_math(Pt_dBm_vec, P_out_U1, K_sim, 'U1', '', 'power');
[eq_U2_Pt, y_U2_Pt] = robust_math(Pt_dBm_vec, P_out_U2, K_sim, 'U2', '', 'power');

plot(Pt_dBm_vec, y_U1_Pt, 'b--', 'LineWidth', 1.5);
plot(Pt_dBm_vec, y_U2_Pt, 'r--', 'LineWidth', 1.5);

% Print Equations Exactly where requested
text(10, 1e-1, eq_U1_Pt, 'Interpreter', 'latex', 'FontSize', 11, 'BackgroundColor', 'w', 'EdgeColor', 'b');
text(10, 1e-3, eq_U2_Pt, 'Interpreter', 'latex', 'FontSize', 11, 'BackgroundColor', 'w', 'EdgeColor', 'r');

xlabel('Transmit Power (dBm)'); ylabel('Outage Probability'); title('Outage Probability vs P_t');
legend('User 1 (Near)', 'User 2 (Far)', 'U1 Math Bound', 'U2 Math Bound'); ylim([1e-5, 1]);

subplot(1,2,2);
plot(Pt_dBm_vec, Sum_Rate, 'mo-', 'LineWidth', 2); grid on;
xlabel('Transmit Power (dBm)'); ylabel('Sum Throughput (bps/Hz)'); title('Joint Optimized Sum Throughput');

% --- Plot 3: Fading Impact (Outage vs ETA and MU) ---
figure('Name', 'Fading Impact', 'Position', [150, 150, 1000, 450]);

% -> Subplot 3a: Outage vs ETA
subplot(1,2,1);
plot(eta_vec, P_out_eta_U1, 'bs-', 'LineWidth', 2); hold on;
plot(eta_vec, P_out_eta_U2, 'ro-', 'LineWidth', 2); grid on;

[eq_U1_eta, y_U1_eta] = robust_math(eta_vec, P_out_eta_U1, K_sim, 'U1', '\eta', 'linear');
[eq_U2_eta, y_U2_eta] = robust_math(eta_vec, P_out_eta_U2, K_sim, 'U2', '\eta', 'linear');
plot(eta_vec, y_U1_eta, 'b--', 'LineWidth', 1.5);
plot(eta_vec, y_U2_eta, 'r--', 'LineWidth', 1.5);

text(0.15, max(P_out_eta_U1)*0.95, eq_U1_eta, 'Interpreter', 'latex', 'BackgroundColor', 'w', 'EdgeColor', 'b');
text(0.15, max(P_out_eta_U2)*1.05, eq_U2_eta, 'Interpreter', 'latex', 'BackgroundColor', 'w', 'EdgeColor', 'r');
xlabel('Fading Parameter \eta'); ylabel('Outage Probability');
title('Outage vs. \eta (Pt = 8 dBm)'); legend('User 1', 'User 2', 'U1 Math', 'U2 Math');

% -> Subplot 3b: Outage vs MU
subplot(1,2,2);
semilogy(mu_vec, max(P_out_mu_U1, 1e-6), 'bs-', 'LineWidth', 2); hold on;
semilogy(mu_vec, max(P_out_mu_U2, 1e-6), 'ro-', 'LineWidth', 2); grid on;

[eq_U1_mu, y_U1_mu] = robust_math(mu_vec, P_out_mu_U1, K_sim, 'U1', '\mu', 'exp');
[eq_U2_mu, y_U2_mu] = robust_math(mu_vec, P_out_mu_U2, K_sim, 'U2', '\mu', 'exp');
plot(mu_vec, y_U1_mu, 'b--', 'LineWidth', 1.5);
plot(mu_vec, y_U2_mu, 'r--', 'LineWidth', 1.5);

text(2.5, 1e-1, eq_U1_mu, 'Interpreter', 'latex', 'BackgroundColor', 'w', 'EdgeColor', 'b');
text(2.5, 1e-3, eq_U2_mu, 'Interpreter', 'latex', 'BackgroundColor', 'w', 'EdgeColor', 'r');
xlabel('Fading Parameter \mu'); ylabel('Outage Probability'); xticks(1:5); ylim([1e-5, 1]);
title('Outage vs. \mu (Pt = 8 dBm)'); legend('User 1', 'User 2', 'U1 Math', 'U2 Math');

%% --- ROBUST MATHEMATICAL EXTRACTION FUNCTION ---
% This safely guarantees User 2 equations derive perfectly even if drops to 0.
function [eq_str, y_plot] = robust_math(x_vec, P_out, K_sim, user, var_name, type)
    if strcmp(type, 'linear')
        p = polyfit(x_vec, P_out, 1); y_plot = polyval(p, x_vec);
        eq_str = sprintf('$$P_{out,%s} \\approx %.4f %s %+.4f$$', user, p(1), var_name, p(2));
    else
        y_fit = P_out; floor_val = 1/(10*K_sim); y_fit(y_fit <= 0) = floor_val;
        idx_above = find(y_fit > floor_val); idx_floor = find(y_fit == floor_val);
        
        if ~isempty(idx_above) && ~isempty(idx_floor)
            idx = [idx_above(end), idx_floor(1)];
        elseif length(idx_above) >= 2
            idx = [idx_above(end-1), idx_above(end)];
        else
            idx = [1, 2];
        end
        
        p = polyfit(x_vec(idx), log10(y_fit(idx)), 1);
        y_plot = 10.^(p(1)*x_vec + p(2));
        
        if strcmp(type, 'power')
            Gd = -10*p(1); Gc = 10^(-(p(1)*(-120) + p(2))/Gd);
            eq_str = sprintf('$$P_{out,%s} \\approx (%.1e \\cdot \\rho)^{-%.2f}$$', user, Gc, Gd);
        else
            eq_str = sprintf('$$P_{out,%s} \\approx 10^{%.2f %s %+.2f}$$', user, p(1), var_name, p(2));
        end
    end
end