%% =========================================================
%  SCENARIO 5: Why NOMA if OFDM is already available?
%  - IRS-assisted NOMA vs IRS-assisted OFDMA
%  - Active IRS, Imperfect SIC (for NOMA)
%  - Quantized Phase Shifts
%  - Metrics: Spectral Efficiency, Fairness, Massive Connectivity
%  - CHANNEL: Eta-Mu Generalized Fading (NLOS)
%% =========================================================
clc; clear; close all;

%% ---- System Parameters ----
N        = 32;
M_users  = [2 4 6 8 10];         % Number of users range
Pt_dBm   = 0:5:40;
Pt_W     = 10.^((Pt_dBm-30)/10);
sigma2   = 1e-11;
sigma_a2 = 1e-12;
eps_sic  = 0.05;
num_bits = 3;
num_MC   = 3000;
PA_dBm   = 20; 
Pa_W     = 10^((PA_dBm-30)/10);
BW_total = 10e6;                 % Total bandwidth 10 MHz

% Eta-Mu Fading Parameters (Format 1)
eta_f = 0.2;                     % Power ratio between I and Q components
mu_f  = 1;                       % Number of multipath clusters

% Path loss parameters (Linear Scale base)
beta_BI = 1e-3; 
beta_IU = 5e-3; 
beta_dir = 2e-4;

phase_levels = exp(1j*2*pi*(0:2^num_bits-1)/2^num_bits);

%% ---- Storage (2-user comparison) ----
R_NOMA_IRS       = zeros(1,length(Pt_dBm));
R_OFDMA_IRS      = zeros(1,length(Pt_dBm));
R_NOMA_noIRS     = zeros(1,length(Pt_dBm));
R_OFDMA_noIRS    = zeros(1,length(Pt_dBm));
Pout_NOMA_IRS_U1 = zeros(1,length(Pt_dBm));
Pout_OFDMA_IRS_U1= zeros(1,length(Pt_dBm));
Fair_NOMA_IRS    = zeros(1,length(Pt_dBm));
Fair_OFDMA_IRS   = zeros(1,length(Pt_dBm));
R1_NOMA_IRS      = zeros(1,length(Pt_dBm));
R2_NOMA_IRS      = zeros(1,length(Pt_dBm));
R1_OFDMA_IRS     = zeros(1,length(Pt_dBm));
R2_OFDMA_IRS     = zeros(1,length(Pt_dBm));

gamma1_th=2^1-1; gamma2_th=2^2-1;

rng(777);
for pi=1:length(Pt_dBm)
    Pt=Pt_W(pi);
    rn=0;ro=0;rn0=0;ro0=0;
    op_n=0;op_o=0;
    fn=0;fo=0;
    r1n=0;r2n=0;r1o=0;r2o=0;
    
    for mc=1:num_MC
        % --- Channels (Eta-Mu Fading) ---
        H_BI  = generate_eta_mu([N, 1], beta_BI, eta_f, mu_f);
        h_IU1 = generate_eta_mu([N, 1], beta_IU, eta_f, mu_f);      % Cell-edge user
        h_IU2 = generate_eta_mu([N, 1], beta_IU*2, eta_f, mu_f);    % Near user (stronger)
        hBU1  = generate_eta_mu([1, 1], beta_dir, eta_f, mu_f);     % Weak direct link
        hBU2  = generate_eta_mu([1, 1], beta_dir*4, eta_f, mu_f);   % Stronger direct link
        
        % Active IRS phase & amplification
        phi_opt=angle(hBU1)-angle(h_IU1.*H_BI);
        phi_q=arrayfun(@(p)angle(phase_levels(find(abs(angle(phase_levels)-p)==min(abs(angle(phase_levels)-p)),1))),phi_opt);
        amp=min(sqrt(Pa_W/(norm(H_BI)^2*Pt+N*sigma_a2)),5.0);
        Phi=amp*diag(exp(1j*phi_q));
        
        na1=norm(h_IU1'*amp*diag(exp(1j*phi_q)))^2*sigma_a2;
        na2=norm(h_IU2'*amp*diag(exp(1j*phi_q)))^2*sigma_a2;
        n1=sigma2+na1; n2=sigma2+na2;
        
        g1_IRS=abs(h_IU1'*Phi*H_BI+hBU1)^2;
        g2_IRS=abs(h_IU2'*Phi*H_BI+hBU2)^2;
        g1_dir=abs(hBU1)^2;
        g2_dir=abs(hBU2)^2;
        
        %% === IRS-NOMA (2 users, full BW) ===
        [a1n,a2n]=pa_noma(g1_IRS/n1,g2_IRS/n2,Pt,eps_sic,gamma1_th,gamma2_th);
        S1n=a1n*g1_IRS*Pt/(a2n*g1_IRS*Pt+n1);
        S2n=a2n*g2_IRS*Pt/(eps_sic*a1n*g2_IRS*Pt+n2);
        r1_n=log2(1+S1n); r2_n=log2(1+S2n);
        rn=rn+r1_n+r2_n;
        r1n=r1n+r1_n; r2n=r2n+r2_n;
        if S1n<gamma1_th, op_n=op_n+1; end
        fn=fn+(r1_n+r2_n)^2/(2*(r1_n^2+r2_n^2));  % Jain's fairness
        
        %% === IRS-OFDMA (2 users, each gets BW/2, same Pt split) ===
        Pt_ofdma = Pt/2;
        S1o = g1_IRS*Pt_ofdma/n1;
        S2o = g2_IRS*Pt_ofdma/n2;
        r1_o=0.5*log2(1+S1o); r2_o=0.5*log2(1+S2o);
        ro=ro+r1_o+r2_o;
        r1o=r1o+r1_o; r2o=r2o+r2_o;
        if S1o<gamma1_th, op_o=op_o+1; end
        fo=fo+(r1_o+r2_o)^2/(2*(r1_o^2+r2_o^2));
        
        %% === Conv NOMA (no IRS) ===
        [a1n0,a2n0]=pa_noma(g1_dir/sigma2,g2_dir/sigma2,Pt,eps_sic,gamma1_th,gamma2_th);
        S1n0=a1n0*g1_dir*Pt/(a2n0*g1_dir*Pt+sigma2);
        S2n0=a2n0*g2_dir*Pt/(eps_sic*a1n0*g2_dir*Pt+sigma2);
        rn0=rn0+log2(1+S1n0)+log2(1+S2n0);
        
        %% === Conv OFDMA (no IRS) ===
        S1o0=g1_dir*Pt_ofdma/sigma2; S2o0=g2_dir*Pt_ofdma/sigma2;
        ro0=ro0+0.5*log2(1+S1o0)+0.5*log2(1+S2o0);
    end
    R_NOMA_IRS(pi)  =rn/num_MC; R_OFDMA_IRS(pi) =ro/num_MC;
    R_NOMA_noIRS(pi)=rn0/num_MC; R_OFDMA_noIRS(pi)=ro0/num_MC;
    Pout_NOMA_IRS_U1(pi)=op_n/num_MC; Pout_OFDMA_IRS_U1(pi)=op_o/num_MC;
    Fair_NOMA_IRS(pi)=fn/num_MC; Fair_OFDMA_IRS(pi)=fo/num_MC;
    R1_NOMA_IRS(pi)=r1n/num_MC; R2_NOMA_IRS(pi)=r2n/num_MC;
    R1_OFDMA_IRS(pi)=r1o/num_MC; R2_OFDMA_IRS(pi)=r2o/num_MC;
    fprintf('Scenario 6 | Pt=%ddBm done\n',Pt_dBm(pi));
end

%% ---- Multi-User Spectral Efficiency (Fixed Pt=25dBm) ----
Pt_mu=10^((25-30)/10);
SE_NOMA_M  = zeros(1,length(M_users));
SE_OFDMA_M = zeros(1,length(M_users));
for mi=1:length(M_users)
    M=M_users(mi);
    rn_m=0; ro_m=0;
    for mc=1:2000
        total_rn=0; total_ro=0;
        for u=1:M
            % --- Channels (Eta-Mu Fading) ---
            bu_u = generate_eta_mu([1, 1], beta_dir*max(1,M-u+1), eta_f, mu_f);
            H_u  = generate_eta_mu([N, 1], beta_BI, eta_f, mu_f);
            iu_u = generate_eta_mu([N, 1], beta_IU, eta_f, mu_f);
            
            phi_u=angle(bu_u)-angle(iu_u.*H_u);
            phi_qu=arrayfun(@(p)angle(phase_levels(find(abs(angle(phase_levels)-p)==min(abs(angle(phase_levels)-p)),1))),phi_u);
            amp_u=min(sqrt(Pa_W/(norm(H_u)^2*Pt_mu+N*sigma_a2)),5.0);
            Phi_u=amp_u*diag(exp(1j*phi_qu));
            na_u=norm(iu_u'*amp_u*diag(exp(1j*phi_qu)))^2*sigma_a2;
            g_u=abs(iu_u'*Phi_u*H_u+bu_u)^2;
            n_u=sigma2+na_u;
            
            % NOMA
            a_u = (M-u+1) / sum(1:M) + 0.1/(M); 
            a_u = min(a_u,0.9/M*M);
            interference_u = max((1-a_u*M)*g_u*Pt_mu/M, 0);
            sinr_noma_u = a_u*g_u*Pt_mu/(interference_u+n_u);
            total_rn = total_rn + log2(1+sinr_noma_u);
            
            % OFDMA
            sinr_ofdma_u = g_u*(Pt_mu/M)/n_u;
            total_ro = total_ro + (1/M)*log2(1+sinr_ofdma_u);
        end
        rn_m=rn_m+total_rn; ro_m=ro_m+total_ro;
    end
    SE_NOMA_M(mi)=rn_m/2000;
    SE_OFDMA_M(mi)=ro_m/2000;
end

%% ============================
%  PLOTTING
%% Figure 1: Jain's Fairness Index
figure('Name', 'Fairness Comparison', 'Position',[250 250 840 550], 'Color', 'w');
plot(Pt_dBm,Fair_NOMA_IRS,'o-','Color','#0072BD','LineWidth',2.2,'MarkerSize',8,'DisplayName','IRS-NOMA Fairness');
hold on;
plot(Pt_dBm,Fair_OFDMA_IRS,'s-','Color','#D95319','LineWidth',2.2,'MarkerSize',8,'DisplayName','IRS-OFDMA Fairness');
yline(1.0,'g--','Perfect Fairness','LineWidth',1.2);
yline(0.5,'r-.','Min Acceptable','LineWidth',1.2);
grid on; xlabel('Transmit Power P_t (dBm)','FontSize',13, 'FontWeight', 'bold'); 
ylabel("Jain's Fairness Index",'FontSize',13, 'FontWeight', 'bold');
title("Fairness Index: NOMA vs OFDMA | Active IRS (\eta-\mu Fading)",'FontSize',14);
legend('Location','south','FontSize',11); set(gca,'FontSize',11); ylim([0 1.15]);

%% Figure 2: SE vs Number of Users (Massive Connectivity)
figure('Name', 'Massive Connectivity', 'Position',[300 300 840 550], 'Color', 'w');
plot(M_users,SE_NOMA_M,'o-','Color','#0072BD','LineWidth',2.2,'MarkerSize',9,'DisplayName','IRS-NOMA');
hold on;
plot(M_users,SE_OFDMA_M,'s-','Color','#D95319','LineWidth',2.2,'MarkerSize',9,'DisplayName','IRS-OFDMA');
grid on; xlabel('Number of Users M','FontSize',13, 'FontWeight', 'bold');
ylabel('Sum Spectral Efficiency (bps/Hz)','FontSize',13, 'FontWeight', 'bold');
title('SE vs Number of Users | NOMA supports Massive Connectivity (\eta-\mu Fading)','FontSize',14);
legend('Location','northeast','FontSize',11); set(gca,'FontSize',11);
annotation('textbox',[0.2 0.15 0.55 0.12],'String',...
    'OFDMA SE drops as users increase (BW splits)\nNOMA maintains SE via power-domain multiplexing',...
    'FontSize',11,'BackgroundColor','w', 'FitBoxToText','on');

%% ============================
%  LOCAL FUNCTIONS
%% ============================
function h = generate_eta_mu(dims, beta, eta_f, mu_f)
    % Generates generalized eta-mu fading channel (NLOS)
    var_I = eta_f / (1 + eta_f);
    var_Q = 1 / (1 + eta_f);
    h = zeros(dims);
    for m = 1:mu_f
        h = h + sqrt(var_I)*randn(dims) + 1j*sqrt(var_Q)*randn(dims);
    end
    % Normalize by mu_f clusters and scale to correct path loss
    h = h * sqrt(beta / mu_f);
end

function [a1,a2] = pa_noma(g1n, g2n, Pt, eps, gth1, gth2)
    % Performs power allocation search to maximize NOMA throughput
    best = -Inf; a1 = 0.7; a2 = 0.3;
    for a = 0.51:0.01:0.95
        b = 1 - a;
        s1 = a * g1n * Pt / (b * g1n * Pt + 1);
        s2 = b * g2n * Pt / (eps * a * g2n * Pt + 1);
        v = 0.6 * log2(1 + s1) + 0.4 * log2(1 + s2);
        if v > best
            best = v; a1 = a; a2 = b; 
        end
    end
end

