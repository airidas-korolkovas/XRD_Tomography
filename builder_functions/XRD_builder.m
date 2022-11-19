function [Iaxis_fine, qaxis, qaxis_fine, NQF] = XRD_builder(a1,a2,segmentation,XS_grid_edges,NQ,NA,S1_grid,S2_grid)
% Fast X-ray diffraction (XRD) tomography for enhanced identification of materials
% by Airidas Korolkovas, Scientific Reports volume 12, Article number: 19097 (2022) 
% https://doi.org/10.1038/s41598-022-23396-2
% airidas.korolkovas89@gmail.com
% Last modified: 11/19/2022

%load('input_for_Deterministic/FF_dynamite.mat', 'qMaster', 'FMaster')

phys.Esrc = 8.04; % Copper K-alpha source energy, keV
phys.hbarc = 1.973269804; % keV*(1 Angstrom)
phys.mc2 = 510.99895000; % keV


% Build a parabolic q-axis at a lower resolution
qmin = 0.5; % starting edge
qmax = 6.0; % ending edge
dq0 = 0.01; % bin width at the first edge
NWT = 256; % number of reconstruction bins
qedges = qmin + (0:NWT)*dq0 + (qmax-qmin-dq0*NWT)/NWT^2*(0:NWT).^2;
qaxis = 0.5*(qedges(1:end-1) + qedges(2:end));
%dq = qedges(2:end) - qedges(1:end-1); % the width of the q-axis bins


NQF = 768; % Number of qfine bins
qedges_fine = qmin + (0:NQF)/NQF*(qmax-qmin);
qaxis_fine = 0.5*(qedges_fine(1:end-1) + qedges_fine(2:end));
%dq_fine = qaxis_fine(2)-qaxis_fine(1);

log2_qgrid = XS_grid_edges(1) + (0:NQ-1)*(XS_grid_edges(2)-XS_grid_edges(1))/(NQ-1);
log2_agrid = XS_grid_edges(3) + (0:NA-1)*(XS_grid_edges(4)-XS_grid_edges(3))/(NA-1);



%% Material 1: Cellulose
filenow = 'open_source_data/cellulose.mat';
load(filenow, 'D');  % D([WVT, Intensity])

% Discard zero-count entries
validexp = D(:,2)>0;
D = D(validexp,:);

%theta = 2*asin(D(:,1)/2/phys.Esrc); % scattering angle, units rad
%qexp = D(:,1)/phys.hbarc; % wavevector transfer, units 1/Angstrom
theta = D(:,1)/180*pi; % scattering angle, units rad
qexp = sin(theta/2)*(2*phys.Esrc)/phys.hbarc; % wavevector transfer, units 1/Angstrom


% Polarization correction (Lorentz factor is ignored for now)
P = (1+cos(theta).^2)/2; % polarization factor
Iexp = D(:,2); %./P; % correct for polarization
Iaxis_fine1 = 0.00003*interp1(qexp,Iexp,qaxis_fine,'makima',NaN);

% Obtain the theoretical Rayleigh and Compton scattering cross-sections in absolute units
log2q = log2(qaxis_fine*phys.hbarc/phys.mc2/2);
log2a = log2( mean(a1(segmentation==1),'all') /mean(a2(segmentation==1), 'all') );

S1now = mean(a2(segmentation==1),'all')*XS_grid_edges(5)/(2^8-1)* ...
    interpn(log2_qgrid, log2_agrid, S1_grid, log2q, log2a, 'makima');

S2now = mean(a2(segmentation==1),'all')*XS_grid_edges(6)/(2^8-1)* ...
    interpn(log2_qgrid, log2_agrid, S2_grid, log2q, log2a, 'makima');

outofrange = isnan(Iaxis_fine1); 
Iaxis_fine1(outofrange) = S1now(outofrange) + S2now(outofrange);

figure(3)
clf reset
hold on
plot(qaxis_fine, Iaxis_fine1, 'DisplayName', 'Calibrated XRD experiment')
plot(qaxis_fine, S1now,'DisplayName','Rayleigh theory')
plot(qaxis_fine, S2now,'DisplayName','Compton theory')
plot(qaxis_fine, S1now + S2now, 'linewidth',1,'displayname','Rayleigh+Compton theory')
set(gca,'yscale','lin')
xlabel(['Wavevector transfer, ', char(197), '^{-1}'])
ylabel('Differential scattering section per unit volume 1/cm')
box on
lgd = legend();
set(gca,'yscale','log','fontsize',12)
title('Cellulose')
ylim([1e-4,1e-1])
drawnow


%% Material 2: Aluminum alloy
filenow = ['open_source_data/aluminum_alloy.mat'];
load(filenow, 'D');  % D([WVT, Intensity])

% Discard zero-count entries
validexp = D(:,2)>0;
D = D(validexp,:);

%theta = 2*asin(D(:,1)/2/phys.Esrc); % scattering angle, units rad
%qexp = D(:,1)/phys.hbarc; % wavevector transfer, units 1/Angstrom
theta = D(:,1)/180*pi; % scattering angle, units rad
qexp = sin(theta/2)*(2*phys.Esrc)/phys.hbarc; % wavevector transfer, units 1/Angstrom

% Polarization correction (Lorentz factor is ignored for now)
P = (1+cos(theta).^2)/2; % polarization factor
Iexp = D(:,2)./P; % correct for polarization
Iaxis_fine2 = 0.005*interp1(qexp,Iexp,qaxis_fine,'makima',NaN);

% Obtain the theoretical Rayleigh and Compton scattering cross-sections in absolute units
log2q = log2(qaxis_fine*phys.hbarc/phys.mc2/2);
log2a = log2( mean(a1(segmentation==2),'all') /mean(a2(segmentation==2), 'all') );

S1now = mean(a2(segmentation==2),'all')*XS_grid_edges(5)/(2^8-1)* ...
    interpn(log2_qgrid, log2_agrid, S1_grid, log2q, log2a, 'makima');

S2now = mean(a2(segmentation==2),'all')*XS_grid_edges(6)/(2^8-1)* ...
    interpn(log2_qgrid, log2_agrid, S2_grid, log2q, log2a, 'makima');

outofrange = isnan(Iaxis_fine2); 
Iaxis_fine2(outofrange) = S1now(outofrange) + S2now(outofrange);


figure(4)
clf reset
hold on
plot(qaxis_fine, Iaxis_fine2, 'DisplayName', 'Calibrated XRD experiment', 'Linewidth', 1)
plot(qaxis_fine, S1now,'DisplayName','Rayleigh theory', 'LineWidth', 2)
plot(qaxis_fine, S2now,'DisplayName','Compton theory', 'LineWidth', 2)
plot(qaxis_fine, S1now + S2now, '--', 'linewidth',2,'displayname','Rayleigh+Compton theory')
xlabel(['Wavevector transfer, ', char(197), '^{-1}'])
ylabel('Differential scattering section per unit volume 1/cm')
box on
lgd = legend();
set(gca,'yscale','log','fontsize',16,'tickdir','out')
xlim([0.4,6.1])
ylim([1e-2, 1e2])
title('Aluminum alloy')


%% Material 3: TNT
%filenow = ['open_source_data/TNT.mat'];
filenow = ['open_source_data/ammoniumnitrate.mat'];
load(filenow, 'D');  % D([WVT, Intensity])

% Discard zero-count entries
validexp = D(:,2)>0;
D = D(validexp,:);

%theta = 2*asin(D(:,1)/2/phys.Esrc); % scattering angle, units rad
%qexp = D(:,1)/phys.hbarc; % wavevector transfer, units 1/Angstrom

theta = D(:,1)/180*pi; % scattering angle, units rad
qexp = sin(theta/2)*(2*phys.Esrc)/phys.hbarc; % wavevector transfer, units 1/Angstrom

% Polarization correction (Lorentz factor is ignored for now)
P = (1+cos(theta).^2)/2; % polarization factor
Iexp = D(:,2); %./P; % correct for polarization
Iaxis_fine3 = 0.0007*interp1(qexp,Iexp,qaxis_fine,'makima',NaN);

% Obtain the theoretical Rayleigh and Compton scattering cross-sections in absolute units
log2q = log2(qaxis_fine*phys.hbarc/phys.mc2/2);
log2a = log2( mean(a1(segmentation==3),'all') /mean(a2(segmentation==3), 'all') );

S1now = mean(a2(segmentation==3),'all')*XS_grid_edges(5)/(2^8-1)* ...
    interpn(log2_qgrid, log2_agrid, S1_grid, log2q, log2a, 'makima');

S2now = mean(a2(segmentation==3),'all')*XS_grid_edges(6)/(2^8-1)* ...
    interpn(log2_qgrid, log2_agrid, S2_grid, log2q, log2a, 'makima');

% Replace out-of-range values with Rayleigh+Compton background

outofrange = isnan(Iaxis_fine3); 
Iaxis_fine3(outofrange) = S1now(outofrange) + S2now(outofrange);

figure(5)
clf reset
hold on
plot(qaxis_fine, Iaxis_fine3, 'DisplayName', 'Calibrated XRD experiment')
plot(qaxis_fine, S1now,'DisplayName','Rayleigh theory')
plot(qaxis_fine, S2now,'DisplayName','Compton theory')
plot(qaxis_fine, S1now + S2now, 'linewidth',1,'displayname','Rayleigh+Compton theory')
set(gca,'yscale','lin')
xlabel(['Wavevector transfer, ', char(197), '^{-1}'])
ylabel('Differential scattering section per unit volume 1/cm')
box on
lgd = legend();
set(gca,'yscale','log','fontsize',12)
title('Ammonium nitrate')
ylim([1e-2,3])


% Concatenate the outputs
Iaxis_fine = [Iaxis_fine1, Iaxis_fine2, Iaxis_fine3]';
end

