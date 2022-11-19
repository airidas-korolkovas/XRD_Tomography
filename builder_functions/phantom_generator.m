function [a1,a2,segmentation_map] = phantom_generator()
% Fast X-ray diffraction (XRD) tomography for enhanced identification of materials
% by Airidas Korolkovas, Scientific Reports volume 12, Article number: 19097 (2022) 
% https://doi.org/10.1038/s41598-022-23396-2
% airidas.korolkovas89@gmail.com
% Last modified: 11/19/2022

load('Physics_inputs/physical_constants.mat', 'phys')
% Generate a digital phantom

% Change the physical scale of the ROI
%myscale = 2;

% Phantom geometrical size in mm
Lx = 200; % width, mm
Ly = 200; % height, mm

% Number of in-plane voxels:
NX = 100;
NY = 100;

% Voxel in-plane coordinates, units: mm
[x,y] = ndgrid(((1:NX)-0.5)*Lx/NX, ((1:NY)-0.5)*Ly/NY);

a1 = zeros(size(x));
a2 = zeros(size(x));
segmentation_map = zeros(size(x));


% physical simulation parameters
phys.K1 = 13.96; % Photoelectric absorption amplitude, units cm^2*keV^3/mol
phys.K2 = 0.30; % Compton scattering amplitude, units cm^2/mol
phys.m  = 3.00; % PE energy slope
phys.n  = 4.20; % PE Z-slope
phys.hc = 12.4; % Planck's constant times the speed of light, in units of keV*Angstrom
phys.mc2 = 510.975; % electron's rest energy mc^2, in keV


%% Material 1 (cellulose, cotton clothes)

% Physical properties
mat.density = 0.1; % in g/cm3
mat.Z1 = 1; % hydrogen
mat.Z2 = 6; % carbon
mat.Z3 = 8; % oxygen
mat.N1 = 10;
mat.N2 = 6;
mat.N3 = 5;
mat.M1 = 1.00784;
mat.M2 = 12.0107;
mat.M3 = 15.999;

mat.a1 = phys.K1/phys.mc2^phys.m*mat.density* ...
    (mat.N1*mat.Z1^phys.n + mat.N2*mat.Z2^phys.n + mat.N3*mat.Z3^phys.n)/ ...
    (mat.N1*mat.M1 + mat.N2*mat.M2 + mat.N3*mat.M3); % units of 1/cm

mat.a2 = phys.K2*mat.density* ...
    (mat.N1*mat.Z1 + mat.N2*mat.Z2 + mat.N3*mat.Z3)/ ...
    (mat.N1*mat.M1 + mat.N2*mat.M2 + mat.N3*mat.M3); % units of 1/cm

% Add Cellulose Object No. 1
% Position
x0 = 20*2;
y0 = 55*2;
tilt_angle = 0;
a = 12*2;
b = 40*2;



validPix =  ...
    (((x-x0)*cos(tilt_angle) - (y-y0)*sin(tilt_angle))/a).^2 + ...
    (((x-x0)*sin(tilt_angle) + (y-y0)*cos(tilt_angle))/b).^2 < 1;

% Add to the pixel map
a1(validPix) = mat.a1;
a2(validPix) = mat.a2;
segmentation_map(validPix) = 1; % insert material No. 1



% Add Cellulose Object No. 2
x0 = 65*2; % x-position, mm
y0 = 75*2; % y-position, mm
tilt_angle = -5/180*pi;
a = 20*2; % major semi-axis, mm
b = 20*2; % minor semi-axis, mm
validPix =  ...
    (((x-x0)*cos(tilt_angle) - (y-y0)*sin(tilt_angle))/a).^2 + ...
    (((x-x0)*sin(tilt_angle) + (y-y0)*cos(tilt_angle))/b).^2 < 1;

% Add to the pixel map
a1(validPix) = mat.a1;
a2(validPix) = mat.a2;
segmentation_map(validPix) = 1; % insert material No. 1


% Add Cellulose Object No. 3
x0 = 65*2; % x-position, mm
y0 = 30*2; % y-position, mm
tilt_angle = 5/180*pi;
a = 20*2; % major semi-axis, mm
b = 15*2; % minor semi-axis, mm
validPix =  ...
    (((x-x0)*cos(tilt_angle) - (y-y0)*sin(tilt_angle))/a).^2 + ...
    (((x-x0)*sin(tilt_angle) + (y-y0)*cos(tilt_angle))/b).^2 < 1;

% Add to the pixel map
a1(validPix) = mat.a1;
a2(validPix) = mat.a2;
segmentation_map(validPix) = 1; % insert material No. 1



%% Material 2 (aluminum alloy)
% Fe    Si    Cu   Zn    Ti   Mn   Mg  Al 
% 0.12 1.9 0.005 0.01 0.009 0.58 5.17  92.206

% Physical properties
mat.density = 2.6; % in g/cm3
mat.Z = [26, 14, 29, 30, 22, 25, 12, 13]; % list of atomic species
mat.wt = [0.12, 1.9, 0.005, 0.01, 0.009, 0.58, 5.17, 92.206]/100; % weight percentages

%mat.M = 26.981539; % molecular weight in g/mol
%mat.Z = 13; % atomic number


%mat.a1 = phys.K1/phys.mc2^phys.m* ...
%    mat.density/mat.M*mat.Z^phys.n; % units of 1/cm
mat.a1 = phys.K1/phys.mc2^phys.m*mat.density* ...
    sum(mat.wt.*mat.Z.^phys.n./phys.M(mat.Z)');

%mat.a2 = phys.K2*mat.density/mat.M*mat.Z; % units of 1/cm
mat.a2 = phys.K2*mat.density* ...
    sum(mat.wt.*mat.Z./phys.M(mat.Z)');

% Add aluminum object No. 1
x0 = 20*2; % x-position, mm
y0 = 65*2; % y-position, mm
tilt_angle = 10/180*pi;
a = 6*2; % major semi-axis, mm
b = 8*2; % minor semi-axis, mm

validPix = ...
    (((x-x0)*cos(tilt_angle) - (y-y0)*sin(tilt_angle))/a).^2 + ...
    (((x-x0)*sin(tilt_angle) + (y-y0)*cos(tilt_angle))/b).^2 < 1;

a1(validPix) = mat.a1;
a2(validPix) = mat.a2;
segmentation_map(validPix) = 2; % insert material No. 2


%% Material 3 (Ammonium nitrate)
% N2 H4 O3
% Physical properties
mat.density = 1.725; % in g/cm3
mat.Z1 = 1; % hydrogen
mat.Z2 = 7; % nitrogen
mat.Z3 = 8; % oxygen
mat.N1 = 4;
mat.N2 = 2;
mat.N3 = 3;
mat.M1 = 1.00784;
mat.M2 = 14.0067;
mat.M3 = 15.999;

mat.a1 = phys.K1/phys.mc2^phys.m*mat.density* ...
    (mat.N1*mat.Z1^phys.n + mat.N2*mat.Z2^phys.n + mat.N3*mat.Z3^phys.n)/ ...
    (mat.N1*mat.M1 + mat.N2*mat.M2 + mat.N3*mat.M3); % units of 1/cm

mat.a2 = phys.K2*mat.density* ...
    (mat.N1*mat.Z1 + mat.N2*mat.Z2 + mat.N3*mat.Z3)/ ...
    (mat.N1*mat.M1 + mat.N2*mat.M2 + mat.N3*mat.M3); % units of 1/cm


% Add TNT object No. 1
% Position
x0 = 20*2; % x-position, mm
y0 = 40*2; % y-position, mm
tilt_angle = 10/180*pi;
a = 6*2; % major semi-axis, mm
b = 6*2; % minor semi-axis, mm

validPix = ...
    (((x-x0)*cos(tilt_angle) - (y-y0)*sin(tilt_angle))/a).^2 + ...
    (((x-x0)*sin(tilt_angle) + (y-y0)*cos(tilt_angle))/b).^2 < 1;

a1(validPix) = mat.a1;
a2(validPix) = mat.a2;
segmentation_map(validPix) = 3; % insert material No. 3


% Add TNT object No. 2
x0 = 65*2; % x-position, mm
y0 = 30*2; % y-position, mm
tilt_angle = 10/180*pi;
a = 10*2; % major semi-axis, mm
b = 6*2; % minor semi-axis, mm

validPix = ...
    (((x-x0)*cos(tilt_angle) - (y-y0)*sin(tilt_angle))/a).^2 + ...
    (((x-x0)*sin(tilt_angle) + (y-y0)*cos(tilt_angle))/b).^2 < 1;

a1(validPix) = mat.a1;
a2(validPix) = mat.a2;
segmentation_map(validPix) = 3; % insert material No. 3




%% Visualize
figure(2)
clf reset
imagesc(segmentation_map')
daspect([1 1 1])
set(gca,'ydir','norm','fontsize',12,'colorscale','lin')
colormap gray
colorbar()
xlabel('x-axis, mm')
ylabel('y-axis, mm')
title('1 = cellulose, 2 = aluminum alloy, 3 = ammonium nitrate')
drawnow
