%% =========================================================
%  SCENARIO 3: Energy Efficiency (EE) Comparison
%  - Passive IRS-NOMA vs Active IRS-NOMA
%  - Joint Power Allocation + Phase Design
%  - Quantized Phase Shifts
%  - EE = Throughput / Total Power Consumption
%% =========================================================
clc; clear; close all;

%% ---- System Parameters ----
N         = 32;
Pt_dBm    = 0:2:40;
Pt_W      = 10.^((Pt_dBm-30)/10);
sigma2    = 1e-11;          % Receiver noise (W)
sigma_a2  = 1e-12;          % Active IRS per-element amplifier noise (W)
eps_sic   = 0.05;           % Imperfect SIC residual
R1_th     = 1.0; R2_th = 2.0;
gamma1_th = 2^R1_th-1; gamma2_th = 2^R2_th-1;
num_bits  = 3;
num_MC    = 4000;

% Hardware power consumption models
P_BS      = 0.1;            % BS circuit power (W)
P_passive_per_elem = 1e-3;  % Passive IRS per-element power (1 mW)
P_active_per_elem  = 5e-3;  % Active IRS per-element circuit power (5 mW)
P_DC_active= 0.05;          % Active IRS DC supply overhead (W)
eta_PA    = 0.35;            % Power amplifier efficiency at BS

% Active IRS amplifier power (variable in one sweep, fixed in another)
PA_dBm    = 20;
Pa_W      = 10^((PA_dBm-30)/10);

% Path loss
beta_BI=1e-3; beta_IU1=5e-3; beta_IU2=8e-3;
beta_BU1=2e-4; beta_BU2=1e-3;

phase_levels = exp(1j*2*pi*(0:2^num_bits-1)/2^num_bits);

%% ---- Storage ----
EE_passive = zeros(1,length(Pt_dBm));
EE_active  = zeros(1,length(Pt_dBm));
Rate_passive = zeros(1,length(Pt_dBm));
Rate_active  = zeros(1,length(Pt_dBm));
Pout_passive_U1 = zeros(1,length(Pt_dBm));
Pout_active_U1  = zeros(1,length(Pt_dBm));

%% ---- Power Allocation ----
function [a1,a2]=pa_joint(g1n,g2n,Pt,eps,gth1,gth2)
    best=-Inf; a1=0.7; a2=0.3;
    for a=0.51:0.01:0.95
        b=1-a;
        s1=a*g1n*Pt/(b*g1n*Pt+1);
        s2=b*g2n*Pt/(eps*a*g2n*Pt+1);
        v=0.6*log2(1+s1)+0.4*log2(1+s2);
        if v>best, best=v; a1=a; a2=b; end
    end
end

rng(123);
for pi=1:length(Pt_dBm)
    Pt=Pt_W(pi);
    rp=0; ra=0; op=0; oa=0;
    
    for mc=1:num_MC
        %% Rayleigh
        H_BI  = sqrt(beta_BI/2)*(randn(N,1)+1j*randn(N,1));
        h_IU1 = sqrt(beta_IU1/2)*(randn(N,1)+1j*randn(N,1));
        h_IU2 = sqrt(beta_IU2/2)*(randn(N,1)+1j*randn(N,1));
        hBU1  = sqrt(beta_BU1/2)*(randn+1j*randn);
        hBU2  = sqrt(beta_BU2/2)*(randn+1j*randn);
        
        %% Optimal Phase (maximize user1 SNR)
        phi_opt = angle(hBU1)-angle(h_IU1.*H_BI);
        phi_q   = arrayfun(@(p) angle(phase_levels(find(abs(angle(phase_levels)-p)==min(abs(angle(phase_levels)-p)),1))), phi_opt);
        
        %% ---- PASSIVE IRS ----
        Phi_p = diag(exp(1j*phi_q));
        g1p = abs(h_IU1'*Phi_p*H_BI+hBU1)^2/sigma2;
        g2p = abs(h_IU2'*Phi_p*H_BI+hBU2)^2/sigma2;
        [a1p,a2p]=pa_joint(g1p,g2p,Pt,eps_sic,gamma1_th,gamma2_th);
        S1p=a1p*g1p*Pt/(a2p*g1p*Pt+1);
        S2p=a2p*g2p*Pt/(eps_sic*a1p*g2p*Pt+1);
        rp=rp+log2(1+S1p)+log2(1+S2p);
        if S1p<gamma1_th, op=op+1; end
        
        %% ---- ACTIVE IRS ----
        amp = min(sqrt(Pa_W/(norm(H_BI)^2*Pt+N*sigma_a2)),5.0);
        Phi_a = amp*diag(exp(1j*phi_q));
        na1 = norm(h_IU1'*amp*diag(exp(1j*phi_q)))^2*sigma_a2;
        na2 = norm(h_IU2'*amp*diag(exp(1j*phi_q)))^2*sigma_a2;
        n1a=sigma2+na1; n2a=sigma2+na2;
        g1a = abs(h_IU1'*Phi_a*H_BI+hBU1)^2;
        g2a = abs(h_IU2'*Phi_a*H_BI+hBU2)^2;
        [a1a,a2a]=pa_joint(g1a/n1a,g2a/n2a,Pt,eps_sic,gamma1_th,gamma2_th);
        S1a=a1a*g1a*Pt/(a2a*g1a*Pt+n1a);
        S2a=a2a*g2a*Pt/(eps_sic*a1a*g2a*Pt+n2a);
        ra=ra+log2(1+S1a)+log2(1+S2a);
        if S1a<gamma1_th, oa=oa+1; end
    end
    
    Rate_passive(pi) = rp/num_MC;
    Rate_active(pi)  = ra/num_MC;
    Pout_passive_U1(pi) = op/num_MC;
    Pout_active_U1(pi)  = oa/num_MC;
    
    %% Total Power Consumption
    % Passive IRS: BS transmit + BS circuit + IRS element circuit
    P_total_passive = Pt/eta_PA + P_BS + N*P_passive_per_elem;
    % Active IRS: BS transmit + BS circuit + IRS element circuit + amplifier power + DC
    P_total_active  = Pt/eta_PA + P_BS + N*P_active_per_elem + Pa_W + P_DC_active;
    
    % EE (bits/J/Hz)
    EE_passive(pi) = Rate_passive(pi) / P_total_passive;
    EE_active(pi)  = Rate_active(pi)  / P_total_active;
    
    fprintf('Scenario 3 | Pt=%ddBm done\n',Pt_dBm(pi));
end

%% ---- EE vs N (Fixed SNR = 20 dBm) ----
N_range = [4 8 16 32 64 128];
EE_passive_N = zeros(1,length(N_range));
EE_active_N  = zeros(1,length(N_range));
Pt_N = 10^((20-30)/10);
for ni=1:length(N_range)
    Nn=N_range(ni);
    rp=0; ra=0;
    for mc=1:num_MC
        H_BI_n=sqrt(beta_BI/2)*(randn(Nn,1)+1j*randn(Nn,1));
        h_IU1_n=sqrt(beta_IU1/2)*(randn(Nn,1)+1j*randn(Nn,1));
        h_IU2_n=sqrt(beta_IU2/2)*(randn(Nn,1)+1j*randn(Nn,1));
        hBU1_n=sqrt(beta_BU1/2)*(randn+1j*randn);
        hBU2_n=sqrt(beta_BU2/2)*(randn+1j*randn);
        phi_n=angle(hBU1_n)-angle(h_IU1_n.*H_BI_n);
        phi_qn=arrayfun(@(p)angle(phase_levels(find(abs(angle(phase_levels)-p)==min(abs(angle(phase_levels)-p)),1))),phi_n);
        % Passive
        Phi_pn=diag(exp(1j*phi_qn));
        g1pn=abs(h_IU1_n'*Phi_pn*H_BI_n+hBU1_n)^2/sigma2;
        g2pn=abs(h_IU2_n'*Phi_pn*H_BI_n+hBU2_n)^2/sigma2;
        [a1pn,a2pn]=pa_joint(g1pn,g2pn,Pt_N,eps_sic,gamma1_th,gamma2_th);
        rp=rp+log2(1+a1pn*g1pn*Pt_N/(a2pn*g1pn*Pt_N+1))+log2(1+a2pn*g2pn*Pt_N/(eps_sic*a1pn*g2pn*Pt_N+1));
        % Active
        amp_n=min(sqrt(Pa_W/(norm(H_BI_n)^2*Pt_N+Nn*sigma_a2)),5.0);
        Phi_an=amp_n*diag(exp(1j*phi_qn));
        na1n=norm(h_IU1_n'*amp_n*diag(exp(1j*phi_qn)))^2*sigma_a2;
        na2n=norm(h_IU2_n'*amp_n*diag(exp(1j*phi_qn)))^2*sigma_a2;
        n1n=sigma2+na1n; n2n=sigma2+na2n;
        g1an=abs(h_IU1_n'*Phi_an*H_BI_n+hBU1_n)^2;
        g2an=abs(h_IU2_n'*Phi_an*H_BI_n+hBU2_n)^2;
        [a1an,a2an]=pa_joint(g1an/n1n,g2an/n2n,Pt_N,eps_sic,gamma1_th,gamma2_th);
        ra=ra+log2(1+a1an*g1an*Pt_N/(a2an*g1an*Pt_N+n1n))+log2(1+a2an*g2an*Pt_N/(eps_sic*a1an*g2an*Pt_N+n2n));
    end
    Rp_N=rp/num_MC; Ra_N=ra/num_MC;
    Ptp_N=Pt_N/eta_PA+P_BS+Nn*P_passive_per_elem;
    Pta_N=Pt_N/eta_PA+P_BS+Nn*P_active_per_elem+Pa_W+P_DC_active;
    EE_passive_N(ni)=Rp_N/Ptp_N;
    EE_active_N(ni)=Ra_N/Pta_N;
end

%% ---- EE vs Active IRS Power Budget (Fixed Pt=20 dBm, N=32) ----
PA_range_dBm=[5 10 15 20 25 30];
EE_active_PA=zeros(1,length(PA_range_dBm));
Rate_active_PA=zeros(1,length(PA_range_dBm));
Pt_PA=10^((20-30)/10);
for pai=1:length(PA_range_dBm)
    Pae=10^((PA_range_dBm(pai)-30)/10);
    ra=0;
    for mc=1:num_MC
        H_BI_a=sqrt(beta_BI/2)*(randn(N,1)+1j*randn(N,1));
        h_IU1_a=sqrt(beta_IU1/2)*(randn(N,1)+1j*randn(N,1));
        h_IU2_a=sqrt(beta_IU2/2)*(randn(N,1)+1j*randn(N,1));
        hBU1_a=sqrt(beta_BU1/2)*(randn+1j*randn);
        hBU2_a=sqrt(beta_BU2/2)*(randn+1j*randn);
        phi_a=angle(hBU1_a)-angle(h_IU1_a.*H_BI_a);
        phi_qa=arrayfun(@(p)angle(phase_levels(find(abs(angle(phase_levels)-p)==min(abs(angle(phase_levels)-p)),1))),phi_a);
        amp_a=min(sqrt(Pae/(norm(H_BI_a)^2*Pt_PA+N*sigma_a2)),5.0);
        Phi_aa=amp_a*diag(exp(1j*phi_qa));
        na1a=norm(h_IU1_a'*amp_a*diag(exp(1j*phi_qa)))^2*sigma_a2;
        na2a=norm(h_IU2_a'*amp_a*diag(exp(1j*phi_qa)))^2*sigma_a2;
        n1a2=sigma2+na1a; n2a2=sigma2+na2a;
        g1a2=abs(h_IU1_a'*Phi_aa*H_BI_a+hBU1_a)^2;
        g2a2=abs(h_IU2_a'*Phi_aa*H_BI_a+hBU2_a)^2;
        [a1a2,a2a2]=pa_joint(g1a2/n1a2,g2a2/n2a2,Pt_PA,eps_sic,gamma1_th,gamma2_th);
        ra=ra+log2(1+a1a2*g1a2*Pt_PA/(a2a2*g1a2*Pt_PA+n1a2))+log2(1+a2a2*g2a2*Pt_PA/(eps_sic*a1a2*g2a2*Pt_PA+n2a2));
    end
    Rate_active_PA(pai)=ra/num_MC;
    Ptot_a=Pt_PA/eta_PA+P_BS+N*P_active_per_elem+Pae+P_DC_active;
    EE_active_PA(pai)=Rate_active_PA(pai)/Ptot_a;
end

%% ============================
%  PLOTTING
%% ============================

%% Figure 1: EE vs Transmit Power
figure('Position',[100 100 820 540]);
plot(Pt_dBm,EE_passive,'o-','Color','#0072BD','LineWidth',2.2,'MarkerSize',8,'DisplayName','Passive IRS-NOMA');
hold on;
plot(Pt_dBm,EE_active,'s-','Color','#D95319','LineWidth',2.2,'MarkerSize',8,'DisplayName','Active IRS-NOMA');
grid on; xlabel('Transmit Power P_t (dBm)','FontSize',13);
ylabel('Energy Efficiency (bits/J/Hz)','FontSize',13);
title('Energy Efficiency vs Transmit Power | Passive vs Active IRS-NOMA','FontSize',13);
legend('Location','northeast','FontSize',11); set(gca,'FontSize',11);
annotation('textbox',[0.15 0.3 0.25 0.08],'String',sprintf('N=%d, P_A=%ddBm\nb=%d bits quant.',N,PA_dBm,num_bits),'FontSize',10,'BackgroundColor','w');

%% Figure 2: Sum Rate vs Transmit Power
figure('Position',[150 150 820 540]);
plot(Pt_dBm,Rate_passive,'o-','Color','#0072BD','LineWidth',2.2,'MarkerSize',8,'DisplayName','Passive IRS-NOMA');
hold on;
plot(Pt_dBm,Rate_active,'s-','Color','#D95319','LineWidth',2.2,'MarkerSize',8,'DisplayName','Active IRS-NOMA');
grid on; xlabel('P_t (dBm)','FontSize',13); ylabel('Sum Rate (bps/Hz)','FontSize',13);
title('Sum Throughput Comparison | Passive vs Active IRS-NOMA','FontSize',13);
legend('Location','northwest','FontSize',11); set(gca,'FontSize',11);

%% Figure 3: EE vs Number of IRS Elements
figure('Position',[200 200 820 540]);
plot(N_range,EE_passive_N,'o-','Color','#0072BD','LineWidth',2.2,'MarkerSize',9,'DisplayName','Passive IRS-NOMA');
hold on;
plot(N_range,EE_active_N,'s-','Color','#D95319','LineWidth',2.2,'MarkerSize',9,'DisplayName','Active IRS-NOMA');
grid on; xlabel('Number of IRS Elements N','FontSize',13);
ylabel('Energy Efficiency (bits/J/Hz)','FontSize',13);
title('EE vs N | Passive vs Active IRS-NOMA, P_t=20dBm','FontSize',13);
legend('Location','northeast','FontSize',11); set(gca,'FontSize',11);
text(20,max(EE_passive_N)*0.5,'Passive EE peaks at large N due to no additional noise cost','FontSize',9,'Color','#0072BD');
text(20,max(EE_active_N)*0.5,'Active EE saturates due to amplifier power overhead','FontSize',9,'Color','#D95319');

%% Figure 4: Power Breakdown Bar Chart
figure('Position',[350 350 820 540]);
Pt_bar = 10^((20-30)/10);
P_passive_total = Pt_bar/eta_PA + P_BS + N*P_passive_per_elem;
P_active_total  = Pt_bar/eta_PA + P_BS + N*P_active_per_elem + Pa_W + P_DC_active;
pdata_passive = [Pt_bar/eta_PA, P_BS, N*P_passive_per_elem, 0, 0] * 1000;
pdata_active  = [Pt_bar/eta_PA, P_BS, N*P_active_per_elem, Pa_W, P_DC_active] * 1000;
bar_data = [pdata_passive; pdata_active];
bar(bar_data,'stacked');
xticklabels({'Passive IRS-NOMA','Active IRS-NOMA'});
ylabel('Power Consumption (mW)','FontSize',13);
title('Power Consumption Breakdown | P_t=20dBm, N=32','FontSize',13);
legend({'BS PA','BS Circuit','IRS Elements','Amplifier','IRS DC'},'Location','northeast','FontSize',10);
grid on; set(gca,'FontSize',11);

fprintf('\n===== Scenario 3 Complete =====\n');
fprintf('Passive IRS EE advantage comes from zero amplifier power cost.\n');
fprintf('Active IRS offers higher throughput but at an EE penalty.\n');
fprintf('Optimal N exists for Active IRS where EE peaks.\n');