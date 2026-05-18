%% =========================================================
%  SCENARIO 2: IRS-NOMA vs Conventional NOMA
%  - ACTIVE IRS (with amplification noise)
%  - Imperfect SIC (residual interference)
%  - Rayleigh & Rician Fading Channels
%  - Joint Power Allocation + Phase Design
%  - Quantized IRS Phase Shifts
%  - Outage Probability Analysis
%% =========================================================
clc; clear; close all;

%% ---- System Parameters ----
N        = 32;
Pt_dBm   = 0:5:40;
Pt_W     = 10.^((Pt_dBm - 30)/10);
sigma2   = 1e-11;           % Receiver noise power (W)
sigma_a2 = 1e-12;           % Active IRS amplifier noise power per element
eps_sic  = 0.05;            % Imperfect SIC residual interference factor (5%)
R1_th    = 1.0;
R2_th    = 2.0;
gamma1_th = 2^R1_th - 1;
gamma2_th = 2^R2_th - 1;
K_lin    = 10^(5/10);       % Rician K = 5 dB
num_bits = 3;               % 3-bit phase quantization
num_MC   = 5000;
PA_max   = 20;              % Max amplification power at active IRS (dBm)
Pa_W     = 10^((PA_max-30)/10);

% Path loss
d_BS_IRS=50; d_IRS_U1=20; d_IRS_U2=10; d_BS_U1=70; d_BS_U2=30;
PL_exp = 2.7;
PL = @(d) (3e8/(4*pi*2.4e9*d))^2 * d^(-PL_exp+2);
beta_BI = PL(d_BS_IRS); beta_IU1=PL(d_IRS_U1); beta_IU2=PL(d_IRS_U2);
beta_BU1=PL(d_BS_U1); beta_BU2=PL(d_BS_U2);

phase_levels = exp(1j*2*pi*(0:2^num_bits-1)/2^num_bits);

%% ---- Storage ----
Pout_conv_Ray_U1=zeros(1,length(Pt_dBm)); Pout_conv_Ray_U2=zeros(1,length(Pt_dBm));
Pout_conv_Ric_U1=zeros(1,length(Pt_dBm)); Pout_conv_Ric_U2=zeros(1,length(Pt_dBm));
Pout_IRS_Ray_U1 =zeros(1,length(Pt_dBm)); Pout_IRS_Ray_U2 =zeros(1,length(Pt_dBm));
Pout_IRS_Ric_U1 =zeros(1,length(Pt_dBm)); Pout_IRS_Ric_U2 =zeros(1,length(Pt_dBm));
Rate_IRS_Ray=zeros(1,length(Pt_dBm)); Rate_IRS_Ric=zeros(1,length(Pt_dBm));
Rate_conv_Ray=zeros(1,length(Pt_dBm)); Rate_conv_Ric=zeros(1,length(Pt_dBm));

%% ---- Joint Power Allocation ----
function [a1, a2] = power_alloc_impSIC(g1_ch, g2_ch, Pt, sigma2, eps, gamma1_th, gamma2_th)
    % With imperfect SIC: residual inter-user interference = eps*a1*Pt*g_self
    % User1 (weak): SINR1 = a1*g1*Pt / (a2*g1*Pt + 1)
    % User2 (strong): SINR2 = a2*g2*Pt / (eps*a1*g2*Pt + 1)  [residual SIC]
    % Maximize fairness-weighted sum: w1*log(1+SINR1) + w2*log(1+SINR2)
    w1=0.6; w2=0.4; % Higher weight to weak user
    
    best_val = -Inf; a1=0.7; a2=0.3;
    for a1_try = 0.51:0.01:0.95
        a2_try = 1 - a1_try;
        S1 = a1_try*g1_ch*Pt/(a2_try*g1_ch*Pt+1);
        S2 = a2_try*g2_ch*Pt/(eps*a1_try*g2_ch*Pt+1);
        val = w1*log2(1+S1) + w2*log2(1+S2);
        if val > best_val && S1 >= gamma1_th && S2 >= gamma2_th
            best_val = val; a1=a1_try; a2=a2_try;
        end
    end
    if best_val == -Inf  % Relax outage constraint if infeasible
        for a1_try = 0.51:0.01:0.95
            a2_try = 1 - a1_try;
            S1 = a1_try*g1_ch*Pt/(a2_try*g1_ch*Pt+1);
            S2 = a2_try*g2_ch*Pt/(eps*a1_try*g2_ch*Pt+1);
            val = w1*log2(1+S1) + w2*log2(1+S2);
            if val > best_val, best_val=val; a1=a1_try; a2=a2_try; end
        end
    end
end

%% ---- Active IRS Amplification Factor ----
function amp = compute_amp_factor(H_BI, Pt, Pa_W, sigma_a2, N)
    % Amplification factor constrained by active IRS power budget
    % |amp|^2 * (||H_BI||^2 * Pt + N*sigma_a2) <= Pa_W
    rx_power = norm(H_BI)^2 * Pt + N*sigma_a2;
    amp = sqrt(Pa_W / rx_power);
    amp = min(amp, 5.0); % Cap amplification
end

rng(42);
%% ---- Monte Carlo ----
for pi = 1:length(Pt_dBm)
    Pt = Pt_W(pi);
    oiR1=0;oiR2=0;oiK1=0;oiK2=0;
    ocR1=0;ocR2=0;ocK1=0;ocK2=0;
    rR_IRS=0;rK_IRS=0;rR_c=0;rK_c=0;
    
    for mc = 1:num_MC
        %% Rayleigh Channels
        h_BU1_R = sqrt(beta_BU1/2)*(randn+1j*randn);
        h_BU2_R = sqrt(beta_BU2/2)*(randn+1j*randn);
        H_BI_R  = sqrt(beta_BI/2)*(randn(N,1)+1j*randn(N,1));
        h_IU1_R = sqrt(beta_IU1/2)*(randn(N,1)+1j*randn(N,1));
        h_IU2_R = sqrt(beta_IU2/2)*(randn(N,1)+1j*randn(N,1));
        
        %% Rician Channels
        h_BU1_K = sqrt(beta_BU1)*(sqrt(K_lin/(K_lin+1))*exp(1j*pi/4) + sqrt(1/(K_lin+1))*(randn+1j*randn)/sqrt(2));
        h_BU2_K = sqrt(beta_BU2)*(sqrt(K_lin/(K_lin+1))*exp(1j*pi/6) + sqrt(1/(K_lin+1))*(randn+1j*randn)/sqrt(2));
        H_BI_K  = sqrt(beta_BI)*(sqrt(K_lin/(K_lin+1))*exp(1j*pi/3*(0:N-1)') + sqrt(1/(K_lin+1))*(randn(N,1)+1j*randn(N,1))/sqrt(2));
        h_IU1_K = sqrt(beta_IU1)*(sqrt(K_lin/(K_lin+1))*exp(1j*pi/5*(0:N-1)') + sqrt(1/(K_lin+1))*(randn(N,1)+1j*randn(N,1))/sqrt(2));
        h_IU2_K = sqrt(beta_IU2)*(sqrt(K_lin/(K_lin+1))*exp(1j*pi/7*(0:N-1)') + sqrt(1/(K_lin+1))*(randn(N,1)+1j*randn(N,1))/sqrt(2));
        
        %% Active IRS Phase + Amplification (Rayleigh)
        amp_R = compute_amp_factor(H_BI_R, Pt, Pa_W, sigma_a2, N);
        phi_opt_R = angle(h_BU1_R) - angle(h_IU1_R .* H_BI_R);
        phi_qR = zeros(N,1);
        for n=1:N
            [~,idx]=min(abs(angle(phase_levels)-phi_opt_R(n)));
            phi_qR(n)=angle(phase_levels(idx));
        end
        % Active IRS: Phi with amplitude > 1 (amplification)
        Phi_aR = amp_R * diag(exp(1j*phi_qR));
        % Active IRS noise: additional noise vector at IRS
        n_IRS_R = sqrt(sigma_a2/2)*(randn(N,1)+1j*randn(N,1));
        
        % Effective channel = h_IU^H * Phi_active * (H_BI * s + n_IRS)
        h_eff1_aR = h_IU1_R' * Phi_aR * H_BI_R + h_BU1_R;
        h_eff2_aR = h_IU2_R' * Phi_aR * H_BI_R + h_BU2_R;
        % Additional noise power from active IRS
        n_add1_R = norm(h_IU1_R' * amp_R * diag(exp(1j*phi_qR)))^2 * sigma_a2;
        n_add2_R = norm(h_IU2_R' * amp_R * diag(exp(1j*phi_qR)))^2 * sigma_a2;
        
        g1_aR = abs(h_eff1_aR)^2;
        g2_aR = abs(h_eff2_aR)^2;
        noise1_R = sigma2 + n_add1_R;
        noise2_R = sigma2 + n_add2_R;
        
        %% Active IRS Phase + Amplification (Rician)
        amp_K = compute_amp_factor(H_BI_K, Pt, Pa_W, sigma_a2, N);
        phi_opt_K = angle(h_BU1_K) - angle(h_IU1_K .* H_BI_K);
        phi_qK = zeros(N,1);
        for n=1:N
            [~,idx]=min(abs(angle(phase_levels)-phi_opt_K(n)));
            phi_qK(n)=angle(phase_levels(idx));
        end
        Phi_aK = amp_K * diag(exp(1j*phi_qK));
        h_eff1_aK = h_IU1_K' * Phi_aK * H_BI_K + h_BU1_K;
        h_eff2_aK = h_IU2_K' * Phi_aK * H_BI_K + h_BU2_K;
        n_add1_K = norm(h_IU1_K' * amp_K * diag(exp(1j*phi_qK)))^2 * sigma_a2;
        n_add2_K = norm(h_IU2_K' * amp_K * diag(exp(1j*phi_qK)))^2 * sigma_a2;
        g1_aK = abs(h_eff1_aK)^2;
        g2_aK = abs(h_eff2_aK)^2;
        noise1_K = sigma2 + n_add1_K;
        noise2_K = sigma2 + n_add2_K;
        
        %% Conventional NOMA channels
        g1_cR = abs(h_BU1_R)^2; g2_cR = abs(h_BU2_R)^2;
        g1_cK = abs(h_BU1_K)^2; g2_cK = abs(h_BU2_K)^2;
        
        %% Power Allocation
        [a1_iR,a2_iR]=power_alloc_impSIC(g1_aR/noise1_R, g2_aR/noise2_R, Pt, sigma2, eps_sic, gamma1_th, gamma2_th);
        [a1_iK,a2_iK]=power_alloc_impSIC(g1_aK/noise1_K, g2_aK/noise2_K, Pt, sigma2, eps_sic, gamma1_th, gamma2_th);
        [a1_cR,a2_cR]=power_alloc_impSIC(g1_cR/sigma2, g2_cR/sigma2, Pt, sigma2, eps_sic, gamma1_th, gamma2_th);
        [a1_cK,a2_cK]=power_alloc_impSIC(g1_cK/sigma2, g2_cK/sigma2, Pt, sigma2, eps_sic, gamma1_th, gamma2_th);
        
        %% SINR with Imperfect SIC
        % Active IRS Rayleigh
        SINR1_iR = a1_iR*g1_aR*Pt / (a2_iR*g1_aR*Pt + noise1_R);
        SINR2_iR = a2_iR*g2_aR*Pt / (eps_sic*a1_iR*g2_aR*Pt + noise2_R); % residual SIC noise
        % Active IRS Rician
        SINR1_iK = a1_iK*g1_aK*Pt / (a2_iK*g1_aK*Pt + noise1_K);
        SINR2_iK = a2_iK*g2_aK*Pt / (eps_sic*a1_iK*g2_aK*Pt + noise2_K);
        % Conv Rayleigh
        SINR1_cR = a1_cR*g1_cR*Pt / (a2_cR*g1_cR*Pt + sigma2);
        SINR2_cR = a2_cR*g2_cR*Pt / (eps_sic*a1_cR*g2_cR*Pt + sigma2);
        % Conv Rician
        SINR1_cK = a1_cK*g1_cK*Pt / (a2_cK*g1_cK*Pt + sigma2);
        SINR2_cK = a2_cK*g2_cK*Pt / (eps_sic*a1_cK*g2_cK*Pt + sigma2);
        
        %% Outage
        if SINR1_iR<gamma1_th, oiR1=oiR1+1; end; if SINR2_iR<gamma2_th, oiR2=oiR2+1; end
        if SINR1_iK<gamma1_th, oiK1=oiK1+1; end; if SINR2_iK<gamma2_th, oiK2=oiK2+1; end
        if SINR1_cR<gamma1_th, ocR1=ocR1+1; end; if SINR2_cR<gamma2_th, ocR2=ocR2+1; end
        if SINR1_cK<gamma1_th, ocK1=ocK1+1; end; if SINR2_cK<gamma2_th, ocK2=ocK2+1; end
        
        rR_IRS=rR_IRS+log2(1+SINR1_iR)+log2(1+SINR2_iR);
        rK_IRS=rK_IRS+log2(1+SINR1_iK)+log2(1+SINR2_iK);
        rR_c=rR_c+log2(1+SINR1_cR)+log2(1+SINR2_cR);
        rK_c=rK_c+log2(1+SINR1_cK)+log2(1+SINR2_cK);
    end
    Pout_IRS_Ray_U1(pi)=oiR1/num_MC; Pout_IRS_Ray_U2(pi)=oiR2/num_MC;
    Pout_IRS_Ric_U1(pi)=oiK1/num_MC; Pout_IRS_Ric_U2(pi)=oiK2/num_MC;
    Pout_conv_Ray_U1(pi)=ocR1/num_MC; Pout_conv_Ray_U2(pi)=ocR2/num_MC;
    Pout_conv_Ric_U1(pi)=ocK1/num_MC; Pout_conv_Ric_U2(pi)=ocK2/num_MC;
    Rate_IRS_Ray(pi)=rR_IRS/num_MC; Rate_IRS_Ric(pi)=rK_IRS/num_MC;
    Rate_conv_Ray(pi)=rR_c/num_MC;  Rate_conv_Ric(pi)=rK_c/num_MC;
    fprintf('Scenario 2 | Pt=%ddBm done\n',Pt_dBm(pi));
end

%% ============================
%  PLOTTING
%% ============================

%% Figure 1: Outage U1 (Weak/Cell-Edge)
figure('Position',[100 100 800 530]);
semilogy(Pt_dBm,Pout_conv_Ray_U1,'o--','Color','#0072BD','LineWidth',1.8,'MarkerSize',7,'DisplayName','Conv NOMA-Rayleigh U1');
hold on;
semilogy(Pt_dBm,Pout_conv_Ric_U1,'s--','Color','#D95319','LineWidth',1.8,'MarkerSize',7,'DisplayName','Conv NOMA-Rician U1');
semilogy(Pt_dBm,Pout_IRS_Ray_U1,'o-','Color','#EDB120','LineWidth',2.2,'MarkerSize',8,'DisplayName','Active IRS-NOMA-Rayleigh U1');
semilogy(Pt_dBm,Pout_IRS_Ric_U1,'s-','Color','#77AC30','LineWidth',2.2,'MarkerSize',8,'DisplayName','Active IRS-NOMA-Rician U1');
yline(0.05,'k-.','5% Target','LineWidth',1.2,'LabelHorizontalAlignment','left');
grid on; xlabel('P_t (dBm)','FontSize',13); ylabel('Outage Probability','FontSize',13);
title('Outage Probability - User 1 (Cell-Edge) | Active IRS, Imperfect SIC','FontSize',13);
legend('Location','southwest','FontSize',10); set(gca,'FontSize',11); ylim([1e-3 1]);

%% Figure 2: Outage U2 (Strong/Near)
figure('Position',[150 150 800 530]);
semilogy(Pt_dBm,Pout_conv_Ray_U2,'o--','Color','#0072BD','LineWidth',1.8,'MarkerSize',7,'DisplayName','Conv NOMA-Rayleigh U2');
hold on;
semilogy(Pt_dBm,Pout_conv_Ric_U2,'s--','Color','#D95319','LineWidth',1.8,'MarkerSize',7,'DisplayName','Conv NOMA-Rician U2');
semilogy(Pt_dBm,Pout_IRS_Ray_U2,'o-','Color','#EDB120','LineWidth',2.2,'MarkerSize',8,'DisplayName','Active IRS-NOMA-Rayleigh U2');
semilogy(Pt_dBm,Pout_IRS_Ric_U2,'s-','Color','#77AC30','LineWidth',2.2,'MarkerSize',8,'DisplayName','Active IRS-NOMA-Rician U2');
yline(0.05,'k-.','5% Target','LineWidth',1.2,'LabelHorizontalAlignment','left');
grid on; xlabel('P_t (dBm)','FontSize',13); ylabel('Outage Probability','FontSize',13);
title('Outage Probability - User 2 (Near) | Active IRS, Imperfect SIC','FontSize',13);
legend('Location','southwest','FontSize',10); set(gca,'FontSize',11); ylim([1e-3 1]);

%% Figure 3: Throughput vs SNR
figure('Position',[200 200 800 530]);
plot(Pt_dBm,Rate_conv_Ray,'o--','Color','#0072BD','LineWidth',1.8,'MarkerSize',7,'DisplayName','Conv NOMA-Rayleigh');
hold on;
plot(Pt_dBm,Rate_conv_Ric,'s--','Color','#D95319','LineWidth',1.8,'MarkerSize',7,'DisplayName','Conv NOMA-Rician');
plot(Pt_dBm,Rate_IRS_Ray,'o-','Color','#EDB120','LineWidth',2.2,'MarkerSize',8,'DisplayName','Active IRS-NOMA-Rayleigh');
plot(Pt_dBm,Rate_IRS_Ric,'s-','Color','#77AC30','LineWidth',2.2,'MarkerSize',8,'DisplayName','Active IRS-NOMA-Rician');
grid on; xlabel('P_t (dBm)','FontSize',13); ylabel('Sum Rate (bps/Hz)','FontSize',13);
title('Sum Throughput vs Transmit Power | Active IRS, Imperfect SIC','FontSize',13);
legend('Location','northwest','FontSize',10); set(gca,'FontSize',11);

%% Figure 4: Effect of SIC imperfection on outage
eps_range = [0, 0.01, 0.05, 0.10, 0.20, 0.30];
Pout_eps_U1 = zeros(1,length(eps_range));
Pout_eps_U2 = zeros(1,length(eps_range));
Pt_eps = 10^((25-30)/10);
for ei = 1:length(eps_range)
    eps_e = eps_range(ei);
    o1=0; o2=0;
    for mc=1:num_MC
        H_BI_e  = sqrt(beta_BI/2)*(randn(N,1)+1j*randn(N,1));
        h_IU1_e = sqrt(beta_IU1/2)*(randn(N,1)+1j*randn(N,1));
        h_IU2_e = sqrt(beta_IU2/2)*(randn(N,1)+1j*randn(N,1));
        hBU1_e  = sqrt(beta_BU1/2)*(randn+1j*randn);
        hBU2_e  = sqrt(beta_BU2/2)*(randn+1j*randn);
        amp_e = sqrt(Pa_W/(norm(H_BI_e)^2*Pt_eps + N*sigma_a2));
        amp_e = min(amp_e,5.0);
        phi_e = angle(hBU1_e)-angle(h_IU1_e.*H_BI_e);
        phi_qe= arrayfun(@(p) angle(phase_levels(find(abs(angle(phase_levels)-p)==min(abs(angle(phase_levels)-p)),1))),phi_e);
        Phi_ae= amp_e*diag(exp(1j*phi_qe));
        g1e=abs(h_IU1_e'*Phi_ae*H_BI_e+hBU1_e)^2;
        g2e=abs(h_IU2_e'*Phi_ae*H_BI_e+hBU2_e)^2;
        na1e=norm(h_IU1_e'*amp_e*diag(exp(1j*phi_qe)))^2*sigma_a2;
        na2e=norm(h_IU2_e'*amp_e*diag(exp(1j*phi_qe)))^2*sigma_a2;
        n1e=sigma2+na1e; n2e=sigma2+na2e;
        [a1e,a2e]=power_alloc_impSIC(g1e/n1e,g2e/n2e,Pt_eps,sigma2,eps_e,gamma1_th,gamma2_th);
        s1e=a1e*g1e*Pt_eps/(a2e*g1e*Pt_eps+n1e);
        s2e=a2e*g2e*Pt_eps/(eps_e*a1e*g2e*Pt_eps+n2e);
        if s1e<gamma1_th, o1=o1+1; end
        if s2e<gamma2_th, o2=o2+1; end
    end
    Pout_eps_U1(ei)=o1/num_MC;
    Pout_eps_U2(ei)=o2/num_MC;
end

figure('Position',[300 300 800 530]);
plot(eps_range*100, Pout_eps_U1, 'o-', 'Color','#EDB120','LineWidth',2.2,'MarkerSize',9,'DisplayName','User 1 (Cell-Edge)');
hold on;
plot(eps_range*100, Pout_eps_U2, 's-', 'Color','#77AC30','LineWidth',2.2,'MarkerSize',9,'DisplayName','User 2 (Near)');
grid on; xlabel('SIC Residual Factor \epsilon (%)','FontSize',13);
ylabel('Outage Probability','FontSize',13);
title('Impact of Imperfect SIC on Outage | Active IRS-NOMA, P_t=25dBm','FontSize',13);
legend('Location','northwest','FontSize',11); set(gca,'FontSize',11);

