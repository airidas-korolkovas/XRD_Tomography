function [Ndet, Eedges] = spectrum_builder(NRG)
% Fast X-ray diffraction (XRD) tomography for enhanced identification of materials
% by Airidas Korolkovas, Scientific Reports volume 12, Article number: 19097 (2022) 
% https://doi.org/10.1038/s41598-022-23396-2
% airidas.korolkovas89@gmail.com
% Last modified: 11/19/2022

%% Load source spectrum
load('Spectrum_80kVp/spectral_data_HD.mat', 'spectrum')


% Gives the emitted number of photons N/[keV cm^2 mAs] @ 1 meter
%exposure_time = 135e-6; % time per firing, seconds
%source_current = 4; % current in mA
exposure_time = 0.100e-3; % exposure time, seconds
source_current = 10; % current in mA

% Convert to units of N/keV
spectrum.counts = spectrum.counts*exposure_time*source_current*100^2;

% The edges of the HD spectrum bins
Eleft = gpuArray(single(spectrum.energy(1:end-1)));
Eright = gpuArray(single(spectrum.energy(2:end)));

% Mean energy of each HD bin
Esrc_HD = 0.5*(spectrum.energy(1:end-1) + spectrum.energy(2:end));
dE_HD = spectrum.energy(2) - spectrum.energy(1); % width of the bins, should be 0.05 keV

% Number of photons in each bin, per bin width in keV
Nsrc_HD = spectrum.counts(1:end-1);

% Bin the energy spectrum on a linear energy axis
Emin = 8;
Emax = 80;

dE0 = (Emax-Emin)/NRG; % bin width
%dE0 = 1; % starting bin width
%Eedges = Emin + (0:NRG)*dE0 + (Emax-Emin-dE0*NRG)/NRG^2*(0:NRG).^2;
Eedges = Emin + (0:NRG)*dE0;

% The indices of the big bins that contain the fine bins
EbinID = discretize(Esrc_HD, Eedges);

% Find the number of photons per bin width
Nsrc = accumarray(EbinID, Nsrc_HD, [NRG,1], @mean);

figure(1)
clf reset
hold on
stairs(spectrum.energy, spectrum.counts, 'linewidth', 2, 'DisplayName', sprintf('High resolution, %.3f keV', dE_HD))
stairs(Eedges, [Nsrc;0], 'linewidth', 2, 'DisplayName', sprintf('Low resolution, %.3f keV', dE0))
xlabel('Energy, keV')
ylabel('(Photon flux per keV) * (100 cm)^2')
xlim([Emin,Emax])
box on
lgd = legend();
title('SpekCalc simulated x-ray spectrum')
set(lgd,'location','northwest','fontsize',16)
set(gca,'fontsize',16,'tickdir','out')

drawnow


%% Integrate the spectrum to a coarse grid
sigmaE = gpuArray(single(0.5*(1.61 + 0.025*Esrc_HD))); % mean square width of the energy response function

% List the detector energy at high-resolution
Eedges = gpuArray(single(Eedges));
Esrc_HD = gpuArray(single(Esrc_HD));
Edet_HD = Esrc_HD'; % size (1,1440)

Ndet = gpuArray.zeros(NRG,NRG,'single');

% number of photons per bin width measured in keV
expwindow = Nsrc_HD.*exp(-(Esrc_HD - Edet_HD).^2./(2*sigmaE.^2))/sqrt(2*pi)./sigmaE; % size (1440,1440)
tic;
for nrg_det = 1:NRG
    Edet1 = Eedges(nrg_det);
    Edet2 = Eedges(nrg_det+1);
        
    for nrg_src = 1:NRG
        Esrc1 = Eedges(nrg_src);
        Esrc2 = Eedges(nrg_src+1);
        
        
        % Intersection length of the integration bin with the window bin, units keV
        wt_src = max(0, min(Esrc2, Eright) - max(Esrc1, Eleft)); % units: keV
        wt_det = max(0, min(Edet2, Eright) - max(Edet1, Eleft))'; % units: keV
        
        % Perform a double integral over a rectangular window (Edet1,Edet2) x (Esrc1,Esrc2)
        Ndet(nrg_src, nrg_det) = sum( wt_src.*wt_det.*expwindow, 'all');
                
    end
    
end
toc;
figure(2)
clf reset
EE = (Eedges(1:end-1)+Eedges(2:end))/2;
alphamap = Ndet'<(mean(Ndet(:))/1e6);
imagesc(EE, EE, Ndet')
xlabel('Detector energy, keV')
ylabel('Source energy, keV')
title('Number of photons')
set(gca,'ydir','norm','fontsize',16,'colorscale','lin','tickdir','out')
colorbar()
daspect([1 1 1])

