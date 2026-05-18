%% =========================================================
%  SCENARIO 1: IRS-NOMA vs Conventional NOMA
%  - Passive IRS, Perfect SIC
%  - Rayleigh & Rician Fading Channels
%  - Joint Power Allocation + Phase Design (no fixed power)
%  - Quantized IRS Phase Shifts
%  - Outage Probability Analysis
%% =========================================================
clc; clear; close all;

%% ---- System Parameters ----
N        = 32;          % Number of IRS elements
M        = 2;           % Number of NOMA users
Pt_dBm   = 0:5:40;      % Total transmit power range (dBm)
Pt_W     = 10.^((Pt_dBm - 30)/10);  % Convert to Watts
sigma2   = 1e-11;       % Noise power (W), -80 dBm
R1_th    = 1.0;         % Target rate for user 1 (weak/cell-edge) bps/Hz
R2_th    = 2.0;         % Target rate for user 2 (strong/near) bps/Hz
gamma1_th = 2^R1_th - 1;
gamma2_th = 2^R2_th - 1;
K_rician = 5;           % Rician K-factor (dB -> linear)
K_lin    = 10^(K_rician/10);
num_bits = 3;           % Phase quantization bits (2^3 = 8 levels)
num_MC   = 5000;        % Monte Carlo iterations
alpha_fair = 0.5;       % Fairness weight

% Path loss model (distance-based)
d_BS_IRS = 50;   % BS to IRS distance (m)
d_IRS_U1 = 20;   % IRS to User1 (cell-edge) (m)
d_IRS_U2 = 10;   % IRS to User2 (near) (m)
d_BS_U1  = 70;   % Direct BS-User1 distance
d_BS_U2  = 30;   % Direct BS-User2 distance
PL_exp   = 2.7;  % Path loss exponent

% Large-scale path loss (simplified)
PL = @(d) (3e8/(4*pi*2.4e9*d))^2 * d^(-PL_exp+2);
beta_BI = PL(d_BS_IRS);
beta_IU1 = PL(d_IRS_U1);
beta_IU2 = PL(d_IRS_U2);
beta_BU1 = PL(d_BS_U1);
beta_BU2 = PL(d_BS_U2);

% Quantized phase levels
phase_levels = exp(1j * 2*pi*(0:2^num_bits-1)/2^num_bits);

%% ---- Storage Arrays ----
Pout_conv_Ray_U1 = zeros(1,length(Pt_dBm));
Pout_conv_Ray_U2 = zeros(1,length(Pt_dBm));
Pout_conv_Ric_U1 = zeros(1,length(Pt_dBm));
Pout_conv_Ric_U2 = zeros(1,length(Pt_dBm));
Pout_IRS_Ray_U1  = zeros(1,length(Pt_dBm));
Pout_IRS_Ray_U2  = zeros(1,length(Pt_dBm));
Pout_IRS_Ric_U1  = zeros(1,length(Pt_dBm));
Pout_IRS_Ric_U2  = zeros(1,length(Pt_dBm));
Rate_IRS_Ray     = zeros(1,length(Pt_dBm));
Rate_IRS_Ric     = zeros(1,length(Pt_dBm));
Rate_conv_Ray    = zeros(1,length(Pt_dBm));
Rate_conv_Ric    = zeros(1,length(Pt_dBm));

%% ---- Helper: Joint Power Allocation via bisection (max sum-rate s.t. fairness) ----
function [a1, a2] = joint_power_alloc(g1, g2, Pt, sigma2, gamma1_th,gamma2_th)
    % a1: power fraction for weak user (cell-edge), a2 for strong user
    % NOMA: strong user (user2) decodes weak user first (SIC)
    % Constraint: a1 + a2 = 1, a1 > a2 (more power to weak user)
    % Maximize: log2(1+a1*g1*Pt/sigma2) + log2(1+a2*g2*Pt/sigma2)
    % Subject to: SINR constraints for outage thresholds
    
    % Minimum power to meet threshold constraints
    a1_min = gamma1_th / (g1*Pt/sigma2 + gamma1_th*g1*Pt/sigma2);
    a1_min = max(a1_min, 0.51); % weak user must get majority
    
    if a1_min >= 1
        a1 = 0.95; a2 = 0.05; return;
    end
    
    % Bisection to find optimal a1 in [a1_min, 0.95]
    lo = a1_min; hi = 0.95;
    for iter = 1:30
        mid = (lo + hi)/2;
        a2_mid = 1 - mid;
        % Gradient check: maximize sum-rate
        % d/da1 [log(1+a1*g1*Pt/sigma2) + log(1+a2*g2*Pt/sigma2)]
        dR_da1 = (g1*Pt/sigma2)/(1+mid*g1*Pt/sigma2) - (g2*Pt/sigma2)/(1+a2_mid*g2*Pt/sigma2);
        if dR_da1 > 0
            lo = mid;
        else
            hi = mid;
        end
    end
    a1 = (lo+hi)/2;
    a2 = 1 - a1;
end

%% ---- Main Monte Carlo Loop ----
rng(42);
for pi = 1:length(Pt_dBm)
    Pt = Pt_W(pi);
    
    out_cR1=0; out_cR2=0; out_cK1=0; out_cK2=0;
    out_iR1=0; out_iR2=0; out_iK1=0; out_iK2=0;
    rR_IRS=0; rK_IRS=0; rR_conv=0; rK_conv=0;
    
    for mc = 1:num_MC
        %% == Rayleigh Channels ==
        % Direct channels
        h_BU1_R = sqrt(beta_BU1/2)*(randn+1j*randn);
        h_BU2_R = sqrt(beta_BU2/2)*(randn+1j*randn);
        % BS->IRS channel (N x 1)
        H_BI_R  = sqrt(beta_BI/2)*(randn(N,1)+1j*randn(N,1));
        % IRS->User channels
        h_IU1_R = sqrt(beta_IU1/2)*(randn(N,1)+1j*randn(N,1));
        h_IU2_R = sqrt(beta_IU2/2)*(randn(N,1)+1j*randn(N,1));
        
        %% == Rician Channels ==
        % LOS component
        phi_BU1 = pi/4; phi_BU2 = pi/6;
        h_BU1_K = sqrt(beta_BU1)*( sqrt(K_lin/(K_lin+1))*exp(1j*phi_BU1) + ...
                  sqrt(1/(K_lin+1))*(randn+1j*randn)/sqrt(2) );
        h_BU2_K = sqrt(beta_BU2)*( sqrt(K_lin/(K_lin+1))*exp(1j*phi_BU2) + ...
                  sqrt(1/(K_lin+1))*(randn+1j*randn)/sqrt(2) );
        % BS->IRS Rician
        theta_BI = pi/3;
        H_BI_K = sqrt(beta_BI)*( sqrt(K_lin/(K_lin+1))*exp(1j*theta_BI*(0:N-1)').* ...
                 ones(N,1) + sqrt(1/(K_lin+1))*(randn(N,1)+1j*randn(N,1))/sqrt(2) );
        h_IU1_K = sqrt(beta_IU1)*( sqrt(K_lin/(K_lin+1))*exp(1j*pi/5*(0:N-1)') + ...
                  sqrt(1/(K_lin+1))*(randn(N,1)+1j*randn(N,1))/sqrt(2) );
        h_IU2_K = sqrt(beta_IU2)*( sqrt(K_lin/(K_lin+1))*exp(1j*pi/7*(0:N-1)') + ...
                  sqrt(1/(K_lin+1))*(randn(N,1)+1j*randn(N,1))/sqrt(2) );
        
        %% == Optimal IRS Phase Design (Rayleigh) ==
        % Maximize |h_IU1^H * Phi * H_BI + h_BU1|^2 for user1 (cell-edge priority)
        % Optimal continuous phase: phi_n = angle(h_BU1) - angle(h_IU1(n)*H_BI(n))
        phi_opt_R = angle(h_BU1_R) - angle(h_IU1_R .* H_BI_R);
        % Quantize phase shifts
        phi_quant_R = zeros(N,1);
        for n = 1:N
            [~,idx] = min(abs(angle(phase_levels) - phi_opt_R(n)));
            phi_quant_R(n) = angle(phase_levels(idx));
        end
        Phi_R = diag(exp(1j*phi_quant_R));
        
        % Effective channels with IRS (Rayleigh)
        h_eff1_R = h_IU1_R' * Phi_R * H_BI_R + h_BU1_R;
        h_eff2_R = h_IU2_R' * Phi_R * H_BI_R + h_BU2_R;
        g1_R = abs(h_eff1_R)^2 / sigma2;
        g2_R = abs(h_eff2_R)^2 / sigma2;
        
        % Direct channels only (conventional NOMA)
        g1_cR = abs(h_BU1_R)^2 / sigma2;
        g2_cR = abs(h_BU2_R)^2 / sigma2;
        
        %% == Optimal IRS Phase Design (Rician) ==
        phi_opt_K = angle(h_BU1_K) - angle(h_IU1_K .* H_BI_K);
        phi_quant_K = zeros(N,1);
        for n = 1:N
            [~,idx] = min(abs(angle(phase_levels) - phi_opt_K(n)));
            phi_quant_K(n) = angle(phase_levels(idx));
        end
        Phi_K = diag(exp(1j*phi_quant_K));
        
        h_eff1_K = h_IU1_K' * Phi_K * H_BI_K + h_BU1_K;
        h_eff2_K = h_IU2_K' * Phi_K * H_BI_K + h_BU2_K;
        g1_K = abs(h_eff1_K)^2 / sigma2;
        g2_K = abs(h_eff2_K)^2 / sigma2;
        
        g1_cK = abs(h_BU1_K)^2 / sigma2;
        g2_cK = abs(h_BU2_K)^2 / sigma2;
        
        %% == Joint Power Allocation ==
        % IRS-NOMA Rayleigh
        [a1_iR, a2_iR] = joint_power_alloc(g1_R*sigma2, g2_R*sigma2, Pt, sigma2, gamma1_th, gamma2_th);
        % IRS-NOMA Rician
        [a1_iK, a2_iK] = joint_power_alloc(g1_K*sigma2, g2_K*sigma2, Pt, sigma2, gamma1_th, gamma2_th);
        % Conv NOMA Rayleigh
        [a1_cR_o, a2_cR_o] = joint_power_alloc(g1_cR*sigma2, g2_cR*sigma2, Pt, sigma2, gamma1_th, gamma2_th);
        % Conv NOMA Rician
        [a1_cK_o, a2_cK_o] = joint_power_alloc(g1_cK*sigma2, g2_cK*sigma2, Pt, sigma2, gamma1_th, gamma2_th);
        
        %% == SINR Computation (Perfect SIC) ==
        % User1 (weak) SINR: treated as interference from user2 at denominator
        % User2 (strong) first decodes user1, cancels it, then decodes own
        
        % IRS Rayleigh
        SINR1_iR = a1_iR*g1_R*Pt / (a2_iR*g1_R*Pt + 1);
        SINR2_iR_SIC = a2_iR*g2_R*Pt;   % After perfect SIC
        
        % IRS Rician
        SINR1_iK = a1_iK*g1_K*Pt / (a2_iK*g1_K*Pt + 1);
        SINR2_iK_SIC = a2_iK*g2_K*Pt;
        
        % Conv Rayleigh
        SINR1_cR = a1_cR_o*g1_cR*Pt / (a2_cR_o*g1_cR*Pt + 1);
        SINR2_cR = a2_cR_o*g2_cR*Pt;
        
        % Conv Rician
        SINR1_cK = a1_cK_o*g1_cK*Pt / (a2_cK_o*g1_cK*Pt + 1);
        SINR2_cK = a2_cK_o*g2_cK*Pt;
        
        %% == Outage Check ==
        % IRS Rayleigh
        if SINR1_iR < gamma1_th, out_iR1 = out_iR1+1; end
        if SINR2_iR_SIC < gamma2_th, out_iR2 = out_iR2+1; end
        % IRS Rician
        if SINR1_iK < gamma1_th, out_iK1 = out_iK1+1; end
        if SINR2_iK_SIC < gamma2_th, out_iK2 = out_iK2+1; end
        % Conv Rayleigh
        if SINR1_cR < gamma1_th, out_cR1 = out_cR1+1; end
        if SINR2_cR < gamma2_th, out_cR2 = out_cR2+1; end
        % Conv Rician
        if SINR1_cK < gamma1_th, out_cK1 = out_cK1+1; end
        if SINR2_cK < gamma2_th, out_cK2 = out_cK2+1; end
        
        %% == Sum Rate ==
        rR_IRS  = rR_IRS  + log2(1+SINR1_iR) + log2(1+SINR2_iR_SIC);
        rK_IRS  = rK_IRS  + log2(1+SINR1_iK) + log2(1+SINR2_iK_SIC);
        rR_conv = rR_conv + log2(1+SINR1_cR)  + log2(1+SINR2_cR);
        rK_conv = rK_conv + log2(1+SINR1_cK)  + log2(1+SINR2_cK);
    end
    
    % Average outage probabilities
    Pout_IRS_Ray_U1(pi)  = out_iR1/num_MC;
    Pout_IRS_Ray_U2(pi)  = out_iR2/num_MC;
    Pout_IRS_Ric_U1(pi)  = out_iK1/num_MC;
    Pout_IRS_Ric_U2(pi)  = out_iK2/num_MC;
    Pout_conv_Ray_U1(pi) = out_cR1/num_MC;
    Pout_conv_Ray_U2(pi) = out_cR2/num_MC;
    Pout_conv_Ric_U1(pi) = out_cK1/num_MC;
    Pout_conv_Ric_U2(pi) = out_cK2/num_MC;
    
    Rate_IRS_Ray(pi)  = rR_IRS/num_MC;
    Rate_IRS_Ric(pi)  = rK_IRS/num_MC;
    Rate_conv_Ray(pi) = rR_conv/num_MC;
    Rate_conv_Ric(pi) = rK_conv/num_MC;
    
    fprintf('Scenario 1 | Pt=%ddBm done\n', Pt_dBm(pi));
end

%% ============================
%  PLOTTING
%% ============================
fig_colors = {'#0072BD','#D95319','#EDB120','#77AC30','#4DBEEE','#A2142F','#7E2F8E','#77AC30'};

%% Figure 1: Outage Probability vs SNR — User 1 (Weak/Cell-Edge)
figure('Name','Outage U1','Position',[100 100 780 520]);
semilogy(Pt_dBm, Pout_conv_Ray_U1, 'o--', 'Color', fig_colors{1}, 'LineWidth',1.8,'MarkerSize',7,'DisplayName','Conv NOMA - Rayleigh U1');
hold on;
semilogy(Pt_dBm, Pout_conv_Ric_U1, 's--', 'Color', fig_colors{2}, 'LineWidth',1.8,'MarkerSize',7,'DisplayName','Conv NOMA - Rician U1');
semilogy(Pt_dBm, Pout_IRS_Ray_U1, 'o-', 'Color', fig_colors{3}, 'LineWidth',2.2,'MarkerSize',8,'DisplayName','IRS-NOMA - Rayleigh U1');
semilogy(Pt_dBm, Pout_IRS_Ric_U1, 's-', 'Color', fig_colors{4}, 'LineWidth',2.2,'MarkerSize',8,'DisplayName','IRS-NOMA - Rician U1');
yline(0.05, 'k-.', 'Target Outage = 5%','LineWidth',1.2,'LabelHorizontalAlignment','left');
grid on; xlabel('Transmit Power P_t (dBm)','FontSize',13);
ylabel('Outage Probability','FontSize',13);
title('Outage Probability - User 1 (Cell-Edge) | Passive IRS, Perfect SIC','FontSize',13);
legend('Location','southwest','FontSize',10);
set(gca,'FontSize',11); ylim([1e-3 1]);

%% Figure 2: Outage Probability vs SNR — User 2 (Strong/Near)
figure('Name','Outage U2','Position',[150 150 780 520]);
semilogy(Pt_dBm, Pout_conv_Ray_U2, 'o--', 'Color', fig_colors{1}, 'LineWidth',1.8,'MarkerSize',7,'DisplayName','Conv NOMA - Rayleigh U2');
hold on;
semilogy(Pt_dBm, Pout_conv_Ric_U2, 's--', 'Color', fig_colors{2}, 'LineWidth',1.8,'MarkerSize',7,'DisplayName','Conv NOMA - Rician U2');
semilogy(Pt_dBm, Pout_IRS_Ray_U2, 'o-', 'Color', fig_colors{3}, 'LineWidth',2.2,'MarkerSize',8,'DisplayName','IRS-NOMA - Rayleigh U2');
semilogy(Pt_dBm, Pout_IRS_Ric_U2, 's-', 'Color', fig_colors{4}, 'LineWidth',2.2,'MarkerSize',8,'DisplayName','IRS-NOMA - Rician U2');
yline(0.05, 'k-.', 'Target = 5%','LineWidth',1.2,'LabelHorizontalAlignment','left');
grid on; xlabel('Transmit Power P_t (dBm)','FontSize',13);
ylabel('Outage Probability','FontSize',13);
title('Outage Probability - User 2 (Near) | Passive IRS, Perfect SIC','FontSize',13);
legend('Location','southwest','FontSize',10);
set(gca,'FontSize',11); ylim([1e-3 1]);

%% Figure 3: Sum Rate vs SNR
figure('Name','Sum Rate','Position',[200 200 780 520]);
plot(Pt_dBm, Rate_conv_Ray, 'o--', 'Color', fig_colors{1}, 'LineWidth',1.8,'MarkerSize',7,'DisplayName','Conv NOMA - Rayleigh');
hold on;
plot(Pt_dBm, Rate_conv_Ric, 's--', 'Color', fig_colors{2}, 'LineWidth',1.8,'MarkerSize',7,'DisplayName','Conv NOMA - Rician');
plot(Pt_dBm, Rate_IRS_Ray,  'o-',  'Color', fig_colors{3}, 'LineWidth',2.2,'MarkerSize',8,'DisplayName','IRS-NOMA - Rayleigh');
plot(Pt_dBm, Rate_IRS_Ric,  's-',  'Color', fig_colors{4}, 'LineWidth',2.2,'MarkerSize',8,'DisplayName','IRS-NOMA - Rician');
grid on; xlabel('Transmit Power P_t (dBm)','FontSize',13);
ylabel('Sum Rate (bps/Hz)','FontSize',13);
title('Sum Throughput vs SNR | Passive IRS, Perfect SIC','FontSize',13);
legend('Location','northwest','FontSize',10);
set(gca,'FontSize',11);



