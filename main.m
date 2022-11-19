% Fast X-ray diffraction (XRD) tomography for enhanced identification of materials
% by Airidas Korolkovas, Scientific Reports volume 12, Article number: 19097 (2022) 
% https://doi.org/10.1038/s41598-022-23396-2
% airidas.korolkovas89@gmail.com
% Last modified: 11/19/2022

% Main driver function
clear
addpath('builder_functions')


%% Load the source spectrum, pre-smeared over the response function width
NRG = 64; % number of energy bins
[Ndet, Eedges] = spectrum_builder(NRG);


%% Load the differential scattering cross-sections on a grid of desired resolution
% Warning! NQ and NA are hardcoded in the CUDA code. The forward projector
% needs to be recompiled if (NQ,NA) are changed
NQ = 64; % number of q (wavector transfer) grid points
NA = 64; % number of aratio = a1/a2 grid points
visualizeON = false;
[S1_grid, ... % dimensionless Rayleigh cross-section, multiply by XS_grid_edges(5)/255 to convert to float
    S2_grid, ... % dimensionless Compton cross-section, multiply by XS_grid_edges(6)/255 to convert to float
    XS_grid_edges] = XSection_builder(NQ, NA, visualizeON);


%% Phantom
[a1, ... % units: 1/cm
    a2, ... % units: 1/cm
    segmentation_map, ... % dimensionless density in the range [0,1]
    ] = phantom_generator();


%% Load the diffraction patterns of the phantom objects
[Iaxis_fine,qaxis,qaxis_fine,NQF] = XRD_builder(a1,a2,segmentation_map,XS_grid_edges,NQ,NA,S1_grid,S2_grid);


%% Launch CUDA
spectrumGPU = gpuArray(single(Ndet)); % size (NRG,1)
IfineGPU = gpuArray(single(Iaxis_fine)); % size (NQF,1)
segmentationGPU = gpuArray(int32(segmentation_map)); % size(NX,NY)
a1GPU = gpuArray(single(a1)); % size(NX,NY)
a2GPU = gpuArray(single(a2)); % size(NX,NY)
S2_grid = gpuArray(single(S2_grid));


% Length unit conversion factors
mm_over_vox = size(a1,1)/100;
vox_over_mm = 1.0/mm_over_vox;
vox_over_cm = 0.1*vox_over_mm;


fprintf('Building the forward projector...\n')
tic;
[NphotonsGPU, CphotonsGPU, AmatrixGPU] = Forward_XRD( ...
    spectrumGPU, ... % source spectrum = number of photons at each energy bin
    IfineGPU, ... % the high-resolution q-axis and scattering cross-section, for forward projection
    segmentationGPU, ... % the segmented material ID at each voxel
    a1GPU*vox_over_cm, ... % photoelectric coefficient
    a2GPU*vox_over_cm, ... % Compton coefficient
    S2_grid, ... % Incoherent scattering cross-section on a [NQ,NA] grid
    XS_grid_edges(1), XS_grid_edges(2), XS_grid_edges(3), XS_grid_edges(4) ... % min and max values of the cross-section grid axes
    );

% Gather outputs
Nphotons = double(gather(NphotonsGPU)); % XRD photons
Cphotons = double(gather(CphotonsGPU))*XS_grid_edges(6)/255; % Compton photons
Amatrix = double(gather(AmatrixGPU));
toc;

% figure(1)
% clf reset
% nn = squeeze(Nphotons);
% imagesc(nn)
% return
% size(Amatrix) = [NWT, MAT, SRC, COL, ROW, NRG]
[NWT,MAT,SRC,COL,ROW,NRG] = size(Amatrix);

src = ceil(SRC/2);
col = ceil(COL/2);
row = ceil(ROW/2);
%%
cc = squeeze(Cphotons(src,:,:));
nn = squeeze(Nphotons(src,:,:));
dd = diag(Ndet)';
figure(8)
clf reset
imagesc((nn+cc))
set(gca,'colorscale','lin')
colorbar()


%% Visualize the model matrix

figure(5)

for src=1:SRC
    % Select the model matrix component
    AmatrixCUDA = reshape(AmatrixGPU(:,:,src,col,row,:), [NWT, MAT, NRG]);
    
    
    clf reset
    tt = tiledlayout(3,1,'TileSpacing','compact');
    xlabel(tt,'Energy bin')
    ylabel(tt,'Wavevector transfer bin')
    title(tt,sprintf('Model matrix at src = %i/%i, col = %i/%i', src, SRC, col, COL))
    mtitles = {'Cellulose', 'Aluminum', 'Ammonium nitrate'};
    for ff=1:3
        nexttile
        imagesc(squeeze(AmatrixCUDA(:,ff,:)))
        colorbar()
        set(gca,'colorscale','lin','ydir','norm','fontsize',12)
        title(mtitles{ff})
    end
    pause;

end

%% Visualize the number of photons
figure(6)
%for src=1:SRC
src = 11;
    clf reset
    imagesc(poissrnd(squeeze(NphotonsGPU(src,:,row,:)))')
    %imagesc(squeeze(N_detected(src,:,row,:))')
    colorbar()
    xlabel('Detector column index')
    ylabel('Energy bin')
    set(gca,'ydir','norm','fontsize',12,'tickdir','out')
    title(sprintf('Number of photons detected from source %i/%i',src,SRC))
    colormap jet
    %plot(1:NRG, N)
    %scatter(1:NRG, squeeze(NphotonsGPU(1,col,:)))
    %pause;
%end


%% Reconstruct the diffraction pattern
%AmatrixCPU = reshape(double(gather(AmatrixGPU)), [NWT*MAT, SRC*COL*ROW*NRG])';
NphotonsCPU = reshape(poissrnd(Nphotons+Cphotons), [SRC*COL*ROW*NRG,1]);
%AtA = AmatrixCPU'*AmatrixCPU;

%Irecon_lsq = lsqminnorm(AmatrixCPU,NphotonsCPU-0*Cphotons(:));
%Irecon_lsq = lsqminnorm(AtA+1*eye(size(AtA)), AmatrixCPU'*NphotonsCPU);
%Irecon_lsq = Amatrix_lores\N;
%fprintf('Running the LASSO solver...\n')
%tic;
%[ILasso,fitinfo] = lasso(AmatrixCPU,NphotonsCPU,'Options',statset('UseParallel',true),'MaxIter',1000);
%[ILasso,fitinfo] = lasso(AmatrixCPU,NphotonsCPU,'Options',statset('UseParallel',true));
%toc;

% Iterative solver
Irecon = ones(NWT*MAT,1)*10;
%Irecon = ILasso(:,10);
%Irecon = Irecon_lsq;
%Irecon(Irecon<0) = 0;

ITR = 1000;
for itr=1:ITR
    %Irecon = Irecon.*(AmatrixCPU'*(NphotonsCPU./(AmatrixCPU*Irecon + 1e-12)))./(sum(AmatrixCPU,1)'+1e-7);
    Irecon = Irecon.*(AmatrixCPU'*((NphotonsCPU-0*Cphotons(:))./(AmatrixCPU*Irecon + Cphotons(:)+1e-20)))./(sum(AmatrixCPU,1)'+1e-7);
    Irecon(~isfinite(Irecon)) = 0;
    
    % Resample at a higher spacing
    %qaxis_HD = 0.35 + (1:1024)'*0.007;
    
    %Irecon_HD = interp1(qaxis, Irecon, qaxis_HD, 'makima', 0);
    % Denoise
    %Irecon_denoise = double_S1D(Irecon_HD,0.001);
    %Irecon_denoise = Irecon_HD;
    
    % Downsample at the usual spacing
    %Irecon = interp1(qaxis_HD, Irecon_denoise, qaxis', 'makima', 0);
    
    Irecon(Irecon<0) = 0;
    
    if mod(itr,1) == 0
        figure(7)
        clf reset
        tt = tiledlayout(3,1,'TileSpacing','compact');
        xlabel(tt, ['Wavevector transfer, ', char(197), '^{-1}'])
        ylabel(tt, 'Scattering cross-section, 1')
        title(tt, sprintf('Iteration %i/%i', itr, ITR))
        mtitles = {'Cellulose', 'Aluminum', 'Ammonium nitrate'};
        ylimlist = {[1e-3,0.05], [0.1, 40], [0.03,5]};
        for ff=1:3
            nexttile
            hold on
            plot(qaxis_fine, Iaxis_fine((1:NQF) + (ff-1)*NQF), 'k', 'DisplayName', 'Ground Truth','linewidth',2)
            plot(qaxis, Irecon((1:NWT) + (ff-1)*NWT), '-.', 'DisplayName', 'Reconstructed', 'linewidth',2)
            title(mtitles{ff})
            box on
            lgd = legend();
            set(gca,'yscale','lin','fontsize',12)
            xlim([0.4,6.1])
            %ylim(ylimlist{ff})
        end
     
        
        drawnow
        
        drawnow
        %pause;
    end
    
end

%% Show measured vs reconstructed photon counts
Nphotons_noisefree = reshape(Nphotons+Cphotons, [SRC,COL,ROW,NRG]);
         figure(10)
         clf reset
         % Compute the scattering pattern based on the model matrix
         N_model = reshape(AmatrixCPU*Irecon, [SRC,COL,ROW,NRG]);
         %N_detected = reshape(NphotonsCPU, [SRC, COL, ROW, NRG]);
         hold on
         stairs(Eedges, [squeeze(N_detected(src,col,row,:));0], 'DisplayName', 'Measured photons', 'linewidth', 2)
         plot(0.5*(Eedges(1:end-1)+Eedges(2:end)), squeeze(N_model(src,col,row,:)), '-o', 'DisplayName', 'Model matrix times reconstructed XRD', 'linewidth', 1)
         plot(0.5*(Eedges(1:end-1)+Eedges(2:end)), squeeze(Nphotons_noisefree(src,col,row,:)), '-.k', 'DisplayName', 'Expected number of photons, noise-free', 'linewidth', 2)
         xlabel('Detected energy, keV')
         ylabel('Number of photons')
         title(sprintf('Source %i/%i, detector column %i/%i', src, SRC, col, COL))
         box on
         lgd = legend();
         set(gca,'fontsize',14,'tickdir','out')
         xlim([8,80])
        




% Matlab-GPU code
% %% Source and detector
% % the ROI in mm
% Lx = 100;
% Ly = 100;
% 
% % source trajectory radius
% Rsrc = 75; % approx Lx/sqrt(2)
% 
% % detectr trajectory radius
% Rdet = 85; % somewhat bigger than Rsrc, to have more zoom and space
% 
% 
% % Source positions, counterclockwise rotation starting from SOUTH
% phi = 0; % source rotation angle
% xsrc = Lx/2 + Rsrc*sin(phi);
% ysrc = Ly/2 - Rsrc*cos(phi);
% zsrc = 0;
% 
% % Flat panel detector positions
% col_pitch = 1; % detector pixel pitch in mm
% col = 150; % detector column id, [1,COL]
% COL = 256; % total number of detetor columns
% rdet = (col - COL/2 - 0.5)*col_pitch;
% xdet = Lx/2 - Rdet*sin(phi) + rdet*cos(phi);
% ydet = Ly/2 + Rdet*cos(phi) + rdet*sin(phi);
% zdet = 10; % detector height offset with respect to the source
% 
% % Detector surface orientation
% ndet = [sin(phi), -cos(phi), 0];
% 
% % Source surface orientation (in the 2D plane)
% geom.target_angle = 30/180*pi; % target angle, rad
% geom.focal_spot = 1; % focal spot size in mm
% nsrc = [ ...
%     -sin(phi)*sin(geom.target_angle), ...
%     cos(phi)*sin(geom.target_angle), ...
%     -cos(geom.target_angle)];

% 
% %% Scattering geometry
% x_src2vox = xvox-xsrc;
% y_src2vox = yvox-ysrc;
% 
% geom.phi = 0.5/180*pi; % wedge angle
% 
% % Voxel mean z-coordinate
% zvox = -sqrt(x_src2vox^2 + y_src2vox^2)*tan(geom.phi/2);
% 
% % Illuminated voxel height
% Hvox = sqrt(x_src2vox^2 + y_src2vox^2)*tan(geom.phi);
% 
% z_src2vox = zvox-zsrc;
% 
% x_vox2det = xdet-xvox;
% y_vox2det = ydet-yvox;
% z_vox2det = zdet-zvox;
% 
% costheta = (x_src2vox*x_vox2det + y_src2vox*y_vox2det + z_src2vox*z_vox2det)./ ...
%     sqrt(x_src2vox^2 + y_src2vox^2 + z_src2vox^2)./ ...
%     sqrt(x_vox2det^2 + y_vox2det^2 + z_vox2det^2);
% 
% sinthetahalf = sqrt((1-costheta)/2);
% 
% 
% 
% %% Additional scatter geometry for computing sigma_wvt
% avector = [x_src2vox, y_src2vox, z_src2vox];
% bvector = [x_vox2det, y_vox2det, z_vox2det];
% anorm2 = sum(avector.^2);
% bnorm2 = sum(bvector.^2);
% 
% Svector =  bvector - sum(avector.*bvector)*avector/anorm2;
% Dvector = -avector + sum(avector.*bvector)*bvector/bnorm2;
% Vvector = -(Svector + Dvector);
% 
% % Compute the mean square product <(d*D)^2>
% geom.Adet = 1; % detector surface area, units: mm^2
% geom.dvox = 1; % voxel side length, units: mm
% dD2 = (sum(Dvector.^2) - sum(Dvector.*ndet).^2)*geom.Adet/12;
% vV2 = (geom.dvox^2*(Vvector(1)^2 + Vvector(2)^2) + Hvox^2*Vvector(3)^2)/12;
% sS2 = (sum(Svector.^2) - sum(Svector.*nsrc).^2)*geom.focal_spot^2;
% 
% phys.hbarc = 1.973269804; % units: keV*Angstrom
% 
% %% Detector response function
% % Energy scale, keV
% %E = 15:70;
% 
% % Excitation energy, keV
% %E = 50;
% 
% % Detector response energy, keV
% %U = Esrc;
% 
% % The width of the response function, where E = incoming energy in keV
% response_width = @(Esrc) 0.5*(1.61 + 0.025*Esrc);
% %response_width = @(Esrc) 0.01 + 0.001*Esrc;
% %response_width = @(Esrc) 0.05;
% 
% 
% % Detector energy channel U response to an incoming photon energy E
% response_function = @(Esrc,Edet) 1./sqrt(2*pi)/response_width(Esrc)*exp(-(Esrc-Edet)^2/2/response_width(Esrc)^2);
% 
% % Ee = 25;
% % c2 = 0.5*exp(-0.015*E);
% % c2(E<Ee) = 0;
% %
% % % background
% % bkg = 1 - (U-(E-3*sigma))./(6*sigma);
% % bkg(U<=(E-3*sigma)) = 1;
% % bkg(U>=(E+3*sigma)) = 0;
% % c3 = 0.042 - E*0.213e-3;
% %
% % % Response function
% % R = 1./sqrt(2*pi*sigma.^2).*exp(-(E-U).^2/2./sigma.^2) + ...
% %     c2./sqrt(2*pi*sigma.^2).*exp(-(E-Ee-U).^2/2./sigma.^2) + ...
% %     c3*bkg;
% %
% %
% % figure(1)
% % clf reset
% % plot(Esrc,R)
% 
% 
% %% Compute the expected number of photons
% N = zeros(NRG,1);
% 
% Edet_list = Ebin;
% Esrc_list = Ebin;
% 
% % Matrix such that N = A*I
% Amatrix = zeros(NRG,NWT);
% 
% % Loop over the source energy bins
% for nrg_src = 1:NRG
%     % Loop over the detector energy bins
%     for nrg_det = 1:NRG
%         % Loop over the cross-section q-bins
%         
%         % Emitted energy in keV
%         Esrc = Esrc_list(nrg_src);
%         
%         % Detected energy in keV
%         Edet = Edet_list(nrg_det);
%         
%         
%         RF = Nsrc(nrg_src)* ... % number of incident photons from the source
%             response_function(Esrc,Edet); % probability that a photon of nrg_src will be detected at a bin nrg_det
%         
%         % Average wavevector transfer
%         wvt = 2*Esrc/phys.hbarc*sinthetahalf;
%         
%         % Mean square width around the WVT
%         sigma_wvt2 = (2*sinthetahalf/phys.hbarc)^2*dE^2/12 + ...
%             (Esrc/phys.hbarc)^2*(sS2 + vV2 + dD2)/(2*sinthetahalf)^2/anorm2/bnorm2;
%         
%         % Resolution along the q-axis, size (1,NQ)
%         Rq = RF*exp(-(wvt-qaxis_fine).^2/(2*sigma_wvt2))/sqrt(2*pi*sigma_wvt2)*dq_fine;
%         
%         % Total number of photons
%         N(nrg_det) = N(nrg_det) + sum(Rq.*Iaxis_fine,2);
%         
%         
%         % Compute the resolution along the non-equidistant q-axis
%         Rq = RF*exp(-(wvt-qaxis).^2/(2*sigma_wvt2))/sqrt(2*pi*sigma_wvt2).*dq;
%         
%         % Matrix terms
%         Amatrix(nrg_det,:) = Amatrix(nrg_det,:) + Rq;
%         
%     end
% end
