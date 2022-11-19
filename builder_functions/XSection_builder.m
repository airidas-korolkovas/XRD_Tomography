function [S1_grid, S2_grid, grid_edges] = XSection_builder(NQbins, NAbins, visualizeON)
% Fast X-ray diffraction (XRD) tomography for enhanced identification of materials
% by Airidas Korolkovas, Scientific Reports volume 12, Article number: 19097 (2022) 
% https://doi.org/10.1038/s41598-022-23396-2
% airidas.korolkovas89@gmail.com
% Last modified: 11/19/2022

%NQbins = 64;
%NAbins = 64;
%visualizeON = true;
printON = false;


%% Step 1. Decompose the attenuation coefficient for each atom using using two basis functions: photoelectric absorption and Compton scattering.
% Assume that the attenuation coefficient for an atom can be well described
% by $\mu(E)/\rho = (a_1/\rho) f_1(E) + (a_2/\rho) f_2(E)$. Here we obtain
% the coefficinets $a_1/\rho$ and $a_2/\rho$ in units of cm^2/g, for each
% atom.

% Load the attenuation coefficient data from Hubbel 1995. The First column
% is energy in MeV, the second column is the attenuation coefficient per
% mass density in cm^2/g.
load('Physics_inputs/attenuation_data.mat', 'attenuation_coef')
load('Physics_inputs/Hubbell_data.mat', 'wvt_list', 'Fcoh', 'Sincoh')
load('Physics_inputs/physical_constants.mat', 'phys')


% Dimensionless energy basis functions, where e=E/mc^2 is the dimensionless
% x-ray energy
f1 = @(e) e.^(-3.0); % photoelectic absorption
f2 = @(e) ((1+e)./e.^2.*( 2*(1+e)./(1+2*e) - log(1+2*e)./e )) + log(1+2*e)./(2*e) - (1+3*e)./(1+2*e).^2; % Compton scattering

% Fit the basis coefficients (a1,a2), in units of cm^2/g
nzrange = 2:27; % the range of atomic numbers to be fitted
NZ = numel(nzrange); % total number of Z values to be fitted
mc2 = 0.5110; % electron rest energy, units: MeV
MeV2keV = 1000;

% Store the fitted material coefficients (a1,a2)
a1 = zeros(1,NZ);
a2 = zeros(1,NZ);

% Fit one atomic number at a time and display the results

for nzid = 1:NZ
    nz = nzrange(nzid);
    % The attenuation coefficient as a function of energy AC(:,1) = energy
    % in MeV, AC(:,2) = attenuation coefficient per unit density, in units
    % of cm^2/g^{-1}
    AC = attenuation_coef{nz};
    
    % Predictor variables are the two energy basis functions
    predictors = [f1(AC(:,1)/mc2), f2(AC(:,1)/mc2)];
    
    % Run the linear regression and extract parameters
    mdl = fitlm(predictors, AC(:,2),'Intercept',false);
    fitparams = mdl.Coefficients.Estimate;
    a1(nzid) = fitparams(1);
    a2(nzid) = fitparams(2);
    
    % Evaluate the predicted fit, which is the fitted mu(E)/\rho
    fiteval = fitparams(end-1)*predictors(:,1) + fitparams(end)*predictors(:,2);
    
    if visualizeON == true
        hf = figure(3);
        set(hf,'WindowStyle','docked')
        clf reset
        hold on
        scatter(AC(:,1)*MeV2keV, AC(:,2), 'DisplayName', 'Hubbel 1995 data')
        plot(AC(:,1)*MeV2keV, fiteval, 'DisplayName', 'Photoelectric+Compton fit')
        xlabel('Energy, keV')
        ylabel('Mass attenuation coefficient, cm^2/g')
        title(['Atomic number = ', num2str(nz)])
        box on
        lgd = legend();
        set(gca,'yscale','lin','tickdir','out','fontsize',12)
        drawnow
        snapnow
        %pause;
    end
end


%% Map to scattering cross-sections
% The incoherent scattering cross-section for a multi-atom compound is given by dsigma/dOmega/V =
% r_e*N_A*rho*(N_1*S_1^2 + ...)/(N_1*M_1 + ...)

% The coherent scattering cross-section, ignoring interference, for a multi-atom compound is given by dsigma/dOmega/V =
% r_e*N_A*rho*(N_1*|F_1|^2 + ...)/(N_1*M_1 + ...)

% Fit this many of the tabulated q-values
% The maximum WVT that we may encounter is 2*E/hbarc = 163 1/Angstrom
nqrange = 2:38;

% Coherent and incoherent scattering cross-sections
S1 = phys.r_e^2*phys.N_A*Fcoh(nqrange,nzrange).^2./phys.M(nzrange)';
S2 = phys.r_e^2*phys.N_A*Sincoh(nqrange,nzrange)./phys.M(nzrange)';


%%
% Tabulate the data on log_2 axes:
q_axis = log2(wvt_list(nqrange)*phys.hbarc/phys.mc2/2);
a_axis = log2(a1./a2);
S1_axis = (S1./a2);
S2_axis = (S2./a2);



% Re-map on a regular grid, using log scale for both WVT and a2/a1 axes
q_min = q_axis(1);
q_max = q_axis(end);
a_min = a_axis(1);
a_max = a_axis(end);
S1_max = max(S1_axis(:));
S2_max = max(S2_axis(:));

uint8_max = 2^8 - 1;
S1_axis = S1_axis*uint8_max/S1_max;
S2_axis = S2_axis*uint8_max/S2_max;

q_grid = q_min + (0:NQbins-1)*(q_max-q_min)/(NQbins-1);
a_grid = a_min + (0:NAbins-1)*(a_max-a_min)/(NAbins-1);

S1_grid = interpn(q_axis, a_axis, S1_axis, q_grid', a_grid, 'makima');
S2_grid = interpn(q_axis, a_axis, S2_axis, q_grid', a_grid, 'makima');

% Cross-section builder settings for CUDA
grid_edges = [q_min, q_max, a_min, a_max, S1_max, S2_max];

if visualizeON == true
    S_axes = {S1_axis; S2_axis};
    S_grids = {S1_grid; S2_grid};
    titles = {'Coherent scattering cross-section d\sigma/d\Omega/V/a_2*255/max, dimensionless';
        'Incoherent scattering cross-section d\sigma/d\Omega/V/a_2*255/max, dimensionless'};
    views = {[75,15]; [-130,15]};
    
    
    for figid=1:2
        hf = figure(figid+3);
        set(hf,'WindowStyle','docked')
        clf reset
        tt = tiledlayout(1,2,'TileSpacing','Compact');
        nexttile;
        surf(q_axis, a_axis, S_axes{figid}')
        set(gca,'xscale','lin','yscale','lin','fontsize',12)
        cbar = colorbar();
        set(cbar,'location','northoutside')
        box on
        view(views{figid})
        xlabel('log_2(WVT*hbar*c/mc^2/2)')
        title('Hubbell data')
        zlim([0,260])
        
        nexttile;
        surf(q_grid, a_grid, S_grids{figid}')
        set(gca,'xscale','lin','yscale','lin','fontsize',12)
        cbar = colorbar();
        set(cbar,'location','northoutside')
        box on
        view(views{figid})
        ylabel('log_2(a_1/a_2)')
        zlim([0,260])
        
        title('Mapped on a regular grid')
        
        title(tt, titles{figid})
        drawnow
        if printON == true
            exportgraphics(tt,sprintf('Figures/xsections_%i.png', figid),'Resolution',250)
        end
    end
    
end


