%% =========================================================
%  SCENARIO 6 : IRS-NOMA Single-Cell vs Multi-Cell
%  - Stochastic Geometry (PPP) for Multi-Cell Interference
%  - Proposed Distributed Algorithm
%  - Active IRS, Imperfect SIC
%  - Joint Power Allocation + Quantized Phase Design
%  - CHANNEL: Eta-Mu Generalized Fading for Desired Cell
%% =========================================================
clc; clear; close all;

%% ---- System Parameters ----
N         = 32;
Pt_dBm    = 0:5:40;
Pt_W      = 10.^((Pt_dBm-30)/10);
sigma2    = 1e-11;
sigma_a2  = 1e-12;
eps_sic   = 0.05;
R1_th     = 1.0;  R2_th = 2.0;
gamma1_th = 2^R1_th - 1;
gamma2_th = 2^R2_th - 1;
num_bits  = 3;
num_MC    = 3000;
PA_dBm    = 20;  Pa_W = 10^((PA_dBm-30)/10);

% Eta-Mu Fading Parameters (Format 1)
eta_f = 0.2;     % Power ratio of In-phase to Quadrature components
mu_f  = 2;       % Number of multipath clusters

% PPP / stochastic geometry
lambda_BS    = 1e-4;   % BS density (per m^2)
r_cell       = 60;     % cell radius (m)
N_interferers= 6;      % number of interfering BSs simulated per sample

% Path loss model
alpha_PL = 3.5;
PL0      = 1e-3;
PL = @(d) PL0 * d^(-alpha_PL);

d_BS_IRS = 40;  d_IRS_U1 = 15;  d_IRS_U2 = 8;
beta_BI  = PL(d_BS_IRS);
beta_IU1 = PL(d_IRS_U1);
beta_IU2 = PL(d_IRS_U2);
beta_BU1 = PL(65);   % cell-edge user distance
beta_BU2 = PL(20);   % near user distance

phase_levels = exp(1j * 2*pi * (0:2^num_bits-1) / 2^num_bits);

%% ============================
%  STORAGE
%% ============================
Pout_SC_U1    = zeros(1, length(Pt_dBm));
Pout_SC_U2    = zeros(1, length(Pt_dBm));
Pout_MC_U1    = zeros(1, length(Pt_dBm));
Pout_MC_U2    = zeros(1, length(Pt_dBm));
Pout_MC_DA_U1 = zeros(1, length(Pt_dBm));
Pout_MC_DA_U2 = zeros(1, length(Pt_dBm));
Rate_SC       = zeros(1, length(Pt_dBm));
Rate_MC       = zeros(1, length(Pt_dBm));
Rate_MC_DA    = zeros(1, length(Pt_dBm));

%% ============================
%  MONTE CARLO MAIN LOOP
%% ============================
rng(314);
for pi = 1:length(Pt_dBm)
    Pt = Pt_W(pi);
    o1sc=0; o2sc=0; o1mc=0; o2mc=0; o1da=0; o2da=0;
    rsc=0;  rmc=0;  rda=0;

    for mc = 1:num_MC

        %% ---- Desired-cell channels (Eta-Mu Fading) ----
        H_BI  = generate_eta_mu([N, 1], beta_BI, eta_f, mu_f);
        h_IU1 = generate_eta_mu([N, 1], beta_IU1, eta_f, mu_f);
        h_IU2 = generate_eta_mu([N, 1], beta_IU2, eta_f, mu_f);
        hBU1  = generate_eta_mu([1, 1], beta_BU1, eta_f, mu_f);
        hBU2  = generate_eta_mu([1, 1], beta_BU2, eta_f, mu_f);

        %% ---- Aggregate inter-cell interference (PPP) ----
        I_agg_U1 = 0;  I_agg_U2 = 0;
        for ifr = 1:N_interferers
            r_int = r_cell + abs(randn) * r_cell * 2;
            r_int = min(r_int, 300);
            I_agg_U1 = I_agg_U1 + Pt * PL(r_int+d_IRS_U1) * (randn^2+randn^2)/2;
            I_agg_U2 = I_agg_U2 + Pt * PL(r_int+d_IRS_U2) * (randn^2+randn^2)/2;
        end

        %% ---- Phase design (shared across SC / MC-naive scenarios) ----
        [amp_sc, phi_q_sc, na1_sc, na2_sc, g1_sc, g2_sc] = ...
            irs_phase_and_gains(H_BI, h_IU1, h_IU2, hBU1, hBU2, ...
                                Pt, Pa_W, sigma_a2, N, phase_levels);

        %% ---- Effective noise per scenario ----
        n1_sc = sigma2 + na1_sc;                    % single-cell
        n2_sc = sigma2 + na2_sc;
        n1_mc = sigma2 + na1_sc + I_agg_U1;         % multi-cell naive
        n2_mc = sigma2 + na2_sc + I_agg_U2;

        %% ---- Single-Cell: no interference, no coordination ----
        [a1sc, a2sc] = pa_dist(g1_sc, g2_sc, Pt, eps_sic, n1_sc, n2_sc);
        S1_sc = a1sc*g1_sc*Pt / (a2sc*g1_sc*Pt + n1_sc);
        S2_sc = a2sc*g2_sc*Pt / (eps_sic*a1sc*g2_sc*Pt + n2_sc);
        if S1_sc < gamma1_th,  o1sc = o1sc+1;  end
        if S2_sc < gamma2_th,  o2sc = o2sc+1;  end
        rsc = rsc + log2(1+S1_sc) + log2(1+S2_sc);

        %% ---- Multi-Cell Naive: same phase, no interference mitigation ----
        [a1mc, a2mc] = pa_dist(g1_sc, g2_sc, Pt, eps_sic, n1_mc, n2_mc);
        S1_mc = a1mc*g1_sc*Pt / (a2mc*g1_sc*Pt + n1_mc);
        S2_mc = a2mc*g2_sc*Pt / (eps_sic*a1mc*g2_sc*Pt + n2_mc);
        if S1_mc < gamma1_th,  o1mc = o1mc+1;  end
        if S2_mc < gamma2_th,  o2mc = o2mc+1;  end
        rmc = rmc + log2(1+S1_mc) + log2(1+S2_mc);

        %% ---- Multi-Cell + Distributed Algorithm ----
        I_est = (I_agg_U1 + I_agg_U2) / 2;   % local interference estimate
        [phi_da, a1da, a2da] = distributed_algo( ...
            H_BI, h_IU1, h_IU2, hBU1, hBU2, ...
            Pt, Pa_W, sigma_a2, N, sigma2, eps_sic, I_est, phase_levels, 3);

        % Recompute gains with DA-optimised phase
        amp_da = min(sqrt(Pa_W/(norm(H_BI)^2*Pt + N*sigma_a2)), 5.0);
        Phi_da = amp_da * diag(exp(1j*phi_da));
        na1_da = amp_da^2 * norm(h_IU1)^2 * sigma_a2;
        na2_da = amp_da^2 * norm(h_IU2)^2 * sigma_a2;
        g1_da  = abs(h_IU1'*Phi_da*H_BI + hBU1)^2;
        g2_da  = abs(h_IU2'*Phi_da*H_BI + hBU2)^2;
        n1_da  = sigma2 + na1_da + I_agg_U1;
        n2_da  = sigma2 + na2_da + I_agg_U2;

        S1_da = a1da*g1_da*Pt / (a2da*g1_da*Pt + n1_da);
        S2_da = a2da*g2_da*Pt / (eps_sic*a1da*g2_da*Pt + n2_da);
        if S1_da < gamma1_th,  o1da = o1da+1;  end
        if S2_da < gamma2_th,  o2da = o2da+1;  end
        rda = rda + log2(1+S1_da) + log2(1+S2_da);
    end

    Pout_SC_U1(pi)    = o1sc/num_MC;  Pout_SC_U2(pi)    = o2sc/num_MC;
    Pout_MC_U1(pi)    = o1mc/num_MC;  Pout_MC_U2(pi)    = o2mc/num_MC;
    Pout_MC_DA_U1(pi) = o1da/num_MC;  Pout_MC_DA_U2(pi) = o2da/num_MC;
    Rate_SC(pi)   = rsc/num_MC;
    Rate_MC(pi)   = rmc/num_MC;
    Rate_MC_DA(pi)= rda/num_MC;
    fprintf('Scenario 6 Main Loop | Pt = %d dBm done\n', Pt_dBm(pi));
end

%% ============================
%  OUTAGE vs BS DENSITY (Fixed Pt = 25 dBm)
%% ============================
lambda_range   = [1e-5  5e-5  1e-4  5e-4  1e-3];
Pout_lambda_MC = zeros(1, length(lambda_range));
Pout_lambda_DA = zeros(1, length(lambda_range));
Pt_lam = 10^((25-30)/10);

for li = 1:length(lambda_range)
    N_int_l = max(min(round(lambda_range(li)*pi*(5*r_cell)^2), 20), 0);
    o_mc = 0;  o_da = 0;

    for mc = 1:2000
        % Desired Channels (Eta-Mu Fading)
        H_BI_l  = generate_eta_mu([N, 1], beta_BI, eta_f, mu_f);
        h_IU1_l = generate_eta_mu([N, 1], beta_IU1, eta_f, mu_f);
        h_IU2_l = generate_eta_mu([N, 1], beta_IU2, eta_f, mu_f);
        hBU1_l  = generate_eta_mu([1, 1], beta_BU1, eta_f, mu_f);
        hBU2_l  = generate_eta_mu([1, 1], beta_BU2, eta_f, mu_f);

        I_l = 0;
        for ifr = 1:N_int_l
            r_l = min(r_cell + abs(randn)*r_cell*2, 300);
            I_l = I_l + Pt_lam * PL(r_l+d_IRS_U1) * (randn^2+randn^2)/2;
        end

        % Phase design
        [~, ~, na1l, na2l, g1l, g2l] = irs_phase_and_gains( ...
            H_BI_l, h_IU1_l, h_IU2_l, hBU1_l, hBU2_l, ...
            Pt_lam, Pa_W, sigma_a2, N, phase_levels);
        n1l = sigma2 + na1l + I_l;
        n2l = sigma2 + na2l + I_l;

        % Multi-cell naive
        [a1l, a2l] = pa_dist(g1l, g2l, Pt_lam, eps_sic, n1l, n2l);
        s1l = a1l*g1l*Pt_lam / (a2l*g1l*Pt_lam + n1l);
        if s1l < gamma1_th,  o_mc = o_mc+1;  end

        % Distributed Algorithm
        [phi_dal, a1dal, a2dal] = distributed_algo( ...
            H_BI_l, h_IU1_l, h_IU2_l, hBU1_l, hBU2_l, ...
            Pt_lam, Pa_W, sigma_a2, N, sigma2, eps_sic, I_l, phase_levels, 2);
        
        amp_dal = min(sqrt(Pa_W/(norm(H_BI_l)^2*Pt_lam+N*sigma_a2)), 5.0);
        Phi_dal = amp_dal * diag(exp(1j*phi_dal));
        na1dal  = amp_dal^2 * norm(h_IU1_l)^2 * sigma_a2;
        g1dal   = abs(h_IU1_l'*Phi_dal*H_BI_l + hBU1_l)^2;
        n1dal   = sigma2 + na1dal + I_l;
        s1dal   = a1dal*g1dal*Pt_lam / (a2dal*g1dal*Pt_lam + n1dal);
        if s1dal < gamma1_th,  o_da = o_da+1;  end
    end
    Pout_lambda_MC(li) = o_mc / 2000;
    Pout_lambda_DA(li) = o_da / 2000;
    fprintf('Scenario 6 Lambda Loop | \lambda = %.0e done\n', lambda_range(li));
end


%% ============================
%  PLOTS
%% ============================

%% Figure 1: Outage U1 — SC vs MC vs MC+DA
figure('Name', 'Outage U1', 'Position',[100 100 840 550], 'Color', 'w');
semilogy(Pt_dBm,Pout_SC_U1,   'o-','Color','#0072BD','LineWidth',2.2,'MarkerSize',8,'DisplayName','Single-Cell IRS-NOMA');
hold on;
semilogy(Pt_dBm,Pout_MC_U1,   's-','Color','#D95319','LineWidth',2.2,'MarkerSize',8,'DisplayName','Multi-Cell (No Coord.)');
semilogy(Pt_dBm,Pout_MC_DA_U1,'^-','Color','#77AC30','LineWidth',2.2,'MarkerSize',8,'DisplayName','Multi-Cell + Dist. Algo');
yline(0.05,'k-.','5% Target','LineWidth',1.2,'LabelHorizontalAlignment','left');
grid on; xlabel('Transmit Power P_t (dBm)','FontSize',13, 'FontWeight', 'bold'); 
ylabel('Outage Probability — User 1','FontSize',13, 'FontWeight', 'bold');
title('Outage: Single-Cell vs Multi-Cell IRS-NOMA (\eta-\mu Fading)','FontSize',14);
legend('Location','southwest','FontSize',11); set(gca,'FontSize',11); ylim([1e-3 1]);

%% Figure 2: Outage U2
figure('Name', 'Outage U2', 'Position',[150 150 840 550], 'Color', 'w');
semilogy(Pt_dBm,Pout_SC_U2,   'o-','Color','#0072BD','LineWidth',2.2,'MarkerSize',8,'DisplayName','Single-Cell');
hold on;
semilogy(Pt_dBm,Pout_MC_U2,   's-','Color','#D95319','LineWidth',2.2,'MarkerSize',8,'DisplayName','Multi-Cell (Naive)');
semilogy(Pt_dBm,Pout_MC_DA_U2,'^-','Color','#77AC30','LineWidth',2.2,'MarkerSize',8,'DisplayName','Multi-Cell + DA');
yline(0.05,'k-.','5% Target','LineWidth',1.2,'LabelHorizontalAlignment','left');
grid on; xlabel('Transmit Power P_t (dBm)','FontSize',13, 'FontWeight', 'bold'); 
ylabel('Outage Probability — User 2','FontSize',13, 'FontWeight', 'bold');
title('Outage — User 2: SC vs MC Active IRS-NOMA (\eta-\mu Fading)','FontSize',14);
legend('Location','southwest','FontSize',11); set(gca,'FontSize',11); ylim([1e-3 1]);

%% Figure 3: Sum Rate
figure('Name', 'Sum Rate', 'Position',[200 200 840 550], 'Color', 'w');
plot(Pt_dBm,Rate_SC,   'o-','Color','#0072BD','LineWidth',2.2,'MarkerSize',8,'DisplayName','Single-Cell');
hold on;
plot(Pt_dBm,Rate_MC,   's-','Color','#D95319','LineWidth',2.2,'MarkerSize',8,'DisplayName','Multi-Cell Naive');
plot(Pt_dBm,Rate_MC_DA,'^-','Color','#77AC30','LineWidth',2.2,'MarkerSize',8,'DisplayName','Multi-Cell + DA');
grid on; xlabel('Transmit Power P_t (dBm)','FontSize',13, 'FontWeight', 'bold'); 
ylabel('Sum Rate (bps/Hz)','FontSize',13, 'FontWeight', 'bold');
title('Sum Throughput: Distributed Algorithm Gain (\eta-\mu Fading)','FontSize',14);
legend('Location','northwest','FontSize',11); set(gca,'FontSize',11);

%% Figure 4: Outage vs BS Density
figure('Name', 'Outage vs Density', 'Position',[250 250 840 550], 'Color', 'w');
semilogy(lambda_range*1e4, Pout_lambda_MC,'s-','Color','#D95319','LineWidth',2.2,'MarkerSize',9,'DisplayName','Multi-Cell Naive');
hold on;
semilogy(lambda_range*1e4, Pout_lambda_DA,'^-','Color','#77AC30','LineWidth',2.2,'MarkerSize',9,'DisplayName','Multi-Cell + DA');
grid on; xlabel('BS Density \lambda_{BS} (\times10^{-4}/m^2)','FontSize',13, 'FontWeight', 'bold');
ylabel('Outage Probability — User 1','FontSize',13, 'FontWeight', 'bold');
title('Outage vs Interferer Density (PPP) | P_t = 25 dBm (\eta-\mu Fading)','FontSize',14);
legend('Location','northwest','FontSize',11); set(gca,'FontSize',11); ylim([0 1]);


%% ============================
%  LOCAL FUNCTIONS
%% ============================

function h = generate_eta_mu(dims, beta, eta_f, mu_f)
    % Generates generalized eta-mu fading channel (NLOS)
    if eta_f == 0
        var_I = 0;
        var_Q = 1;
    else
        var_I = eta_f / (1 + eta_f);
        var_Q = 1 / (1 + eta_f);
    end
    
    h = zeros(dims);
    for m = 1:mu_f
        h = h + sqrt(var_I)*randn(dims) + 1j*sqrt(var_Q)*randn(dims);
    end
    % Normalize power by mu_f to prevent energy blow-up, apply path loss
    h = h * sqrt(beta / mu_f);
end

function [a1, a2] = pa_dist(g1_raw, g2_raw, Pt, eps, noise1, noise2)
    % Joint power allocation — takes RAW channel gains
    best = -Inf;  a1 = 0.7;  a2 = 0.3;
    for a = 0.51:0.01:0.95
        b  = 1 - a;
        s1 = a * g1_raw * Pt / (b * g1_raw * Pt + noise1);
        s2 = b * g2_raw * Pt / (eps * a * g2_raw * Pt + noise2);
        v  = 0.6*log2(1+s1) + 0.4*log2(1+s2);
        if v > best,  best = v;  a1 = a;  a2 = b;  end
    end
end

function [amp, phi_q, na1, na2, g1, g2] = irs_phase_and_gains( ...
        H_BI, h_IU1, h_IU2, hBU1, hBU2, Pt, Pa_W, sigma_a2, N, phase_levels)
    % Compute quantised IRS phase, amplifier noise, and effective channel gains.
    phi_opt = angle(hBU1) - angle(h_IU1 .* H_BI);
    phi_q   = zeros(N,1);
    for n = 1:N
        [~, idx] = min(abs(angle(phase_levels) - phi_opt(n)));
        phi_q(n) = angle(phase_levels(idx));
    end
    amp  = min(sqrt(Pa_W / (norm(H_BI)^2*Pt + N*sigma_a2)), 5.0);
    na1  = amp^2 * norm(h_IU1)^2 * sigma_a2;
    na2  = amp^2 * norm(h_IU2)^2 * sigma_a2;
    Phi  = amp * diag(exp(1j*phi_q));
    g1   = abs(h_IU1'*Phi*H_BI + hBU1)^2;
    g2   = abs(h_IU2'*Phi*H_BI + hBU2)^2;
end

function [phi_best, a1_best, a2_best] = distributed_algo( ...
        H_BI, h_IU1, h_IU2, hBU1, hBU2, ...
        Pt, Pa_W, sigma_a2, N, sigma2, eps_sic, I_inter, phase_levels, max_iter)
    % Distributed iterative phase + power optimisation
    amp = min(sqrt(Pa_W / (norm(H_BI)^2*Pt + N*sigma_a2)), 5.0);
    na1_fixed = amp^2 * norm(h_IU1)^2 * sigma_a2;
    na2_fixed = amp^2 * norm(h_IU2)^2 * sigma_a2;
    n1 = sigma2 + na1_fixed + I_inter;
    n2 = sigma2 + na2_fixed + I_inter;
    v1 = amp * conj(h_IU1) .* H_BI;   
    v2 = amp * conj(h_IU2) .* H_BI;   

    phi_q    = zeros(N, 1);     
    phi_best = phi_q;           
    a1_best  = 0.7;
    a2_best  = 0.3;
    rate_best = -Inf;

    for iter = 1:max_iter
        comb1 = sum(v1 .* exp(1j*phi_q)) + hBU1;   
        comb2 = sum(v2 .* exp(1j*phi_q)) + hBU2;

        for n = 1:N
            base1 = comb1 - v1(n)*exp(1j*phi_q(n));
            base2 = comb2 - v2(n)*exp(1j*phi_q(n));
            best_ph_n = phi_q(n);
            best_r_n  = -Inf;

            for ph_try = angle(phase_levels)
                trial1 = base1 + v1(n)*exp(1j*ph_try);
                trial2 = base2 + v2(n)*exp(1j*ph_try);
                g1_try = abs(trial1)^2;
                g2_try = abs(trial2)^2;

                [a1t, a2t] = pa_dist(g1_try, g2_try, Pt, eps_sic, n1, n2);
                s1_try = a1t*g1_try*Pt / (a2t*g1_try*Pt + n1);
                s2_try = a2t*g2_try*Pt / (eps_sic*a1t*g2_try*Pt + n2);
                r_try  = 0.6*log2(1+s1_try) + 0.4*log2(1+s2_try);

                if r_try > best_r_n
                    best_r_n  = r_try;
                    best_ph_n = ph_try;
                end
            end
            comb1 = base1 + v1(n)*exp(1j*best_ph_n);
            comb2 = base2 + v2(n)*exp(1j*best_ph_n);
            phi_q(n) = best_ph_n;
        end

        g1_cur = abs(comb1)^2;
        g2_cur = abs(comb2)^2;
        [a1, a2] = pa_dist(g1_cur, g2_cur, Pt, eps_sic, n1, n2);
        s1 = a1*g1_cur*Pt / (a2*g1_cur*Pt + n1);
        s2 = a2*g2_cur*Pt / (eps_sic*a1*g2_cur*Pt + n2);
        cur_rate = 0.6*log2(1+s1) + 0.4*log2(1+s2);

        if cur_rate > rate_best
            rate_best = cur_rate;
            phi_best  = phi_q;
            a1_best   = a1;
            a2_best   = a2;
        end
    end
end