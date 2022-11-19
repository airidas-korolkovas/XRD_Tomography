// Fast X-ray diffraction (XRD) tomography for enhanced identification of materials
// by Airidas Korolkovas, Scientific Reports volume 12, Article number: 19097 (2022) 
// https://doi.org/10.1038/s41598-022-23396-2
// airidas.korolkovas89@gmail.com
// Last modified: 11/19/2022


/* Performs line integrals from voxel to each detector */
void __global__ integrate_diffraction(
        float * spectrum,
        float * Nphotons,
        float * Amatrix,
        float * Ifine,
        int * segmentation,
        int NX,
        int NY,
        float * Cphotons,
        float * S2_data,
        float const q_min,
        float const q_max,
        float const a_min,
        float const a_max) {

	// Storage for partial sums of Nphotons
    static __shared__ float Nshared[numWarps];

    // Storage for source spectrum, as seen from a detector channel
    static __shared__ float shared_spectrum[2*DNRG+1];

    // Storage for the Compton scattering cross-sections
    static __shared__ unsigned char S2[NQ*NA];

    int src = blockIdx.x;
    int det = blockIdx.y;
	int nrg = blockIdx.z; // detector energy bin index, [0,NRG=63]
    int nwt = threadIdx.x; // range [0,NWT-1]

    // Detector 2D indices
	int col = det % COL;
	int row = (det - col)/COL;

    
    // Compute the (q,dq) values of the current reconstruction bin
    float qnow, dqnow;
    wvt_value(nwt, &qnow, &dqnow);
    

    // Load the high-resolution scattering cross-section
    //float Inow[9] = {Ifine[nwt], Ifine[nwt+NWT], Ifine[nwt+2*NWT], Ifine[nwt+3*NWT], Ifine[nwt+4*NWT], Ifine[nwt+5*NWT], Ifine[nwt+6*NWT], Ifine[nwt+7*NWT], Ifine[nwt+8*NWT]};
    float Inow[MAT*NQF/NWT];
	for(int loadID = 0; loadID<(MAT*NQF/NWT); loadID++) {
        Inow[loadID] = Ifine[nwt+loadID*NWT];
    }

    // Determine the range of source energy indices (where the photon flux is not negligible)
    int nrg_src_min, nrg_src_max;
    find_src_nrg_range(nrg, &nrg_src_min, &nrg_src_max);

    // Load the source spectrum, smeared with the response function
    if (nwt<=(2*DNRG)) {
        shared_spectrum[nwt] = spectrum[nrg_src_min + nwt + nrg*NRG];
    }


    // Read the scattering cross-section tables, each of size [NQ,NA]
    int readnow = nwt; // reset counter
	int S_temp;
    while (readnow < (NQ*NA)) {
        S_temp = __float2int_rd(S2_data[readnow]);
        if (S_temp<0) { S_temp = 0; }
        else if (S_temp > 255) { S_temp = 255; }
        S2[readnow] = (unsigned char) S_temp;
        readnow += NWT;
    }


    __syncthreads();

    /* Scattering geometry */

    float mm_over_vox = __int2float_rd(NX)/100.f;

    // Source and detector
    float Rsrc = 150.f*mm_over_vox; // source trajectory radius in units of vox
    float Rdet = 170.f*mm_over_vox; // detector trajectory radius

    // ROI size in vox units
    float Lx = 200.f*mm_over_vox;
    float Ly = 200.f*mm_over_vox;


    // Source positions, counterclockwise rotation starting from SOUTH
    float phi = __int2float_rn(src)/__int2float_rn(SRC)*twopi; // source rotation angle
    float Rsrc_x = Lx/2.f + Rsrc*sinf(phi);
    float Rsrc_y = Ly/2.f - Rsrc*cosf(phi);
    float Rsrc_z = 0.f;

    //  Flat panel detector positions
    float col_pitch = 0.5*mm_over_vox; // % detector pixel pitch in mm
	float row_pitch = 1.f*mm_over_vox; // detector row pitch in mm
	float z0 = 10.f*mm_over_vox; // the z-coordinate of the row=0

    // Detector linear position along the panel
    float rdet = (__int2float_rn(col) - __int2float_rn(COL)/2.f + 0.5f)*col_pitch;

    // Detector Cartesian coordinates
    float Rdet_x = Lx/2.f - Rdet*sinf(phi) + rdet*cosf(phi);
    float Rdet_y = Ly/2.f + Rdet*cosf(phi) + rdet*sinf(phi);
    float Rdet_z = z0 + __int2float_rn(row)*row_pitch; // detector height offset with respect to the source

    // Detector surface orientation
    float ndet_x = sinf(phi);
    float ndet_y = -cosf(phi);
    float ndet_z = 0.f;

    // Source surface orientation (in the 2D plane)
	float target_angle = 0.523598775598299f; // 30/180*pi
    float focal_spot = 0.5f*mm_over_vox; //  focal spot size in units of vox
	float nsrc_x = -sinf(phi)*sinf(target_angle);
    float nsrc_y = cosf(phi)*sinf(target_angle);
    float nsrc_z = -cosf(target_angle);



    // Define the geometry variables
    float geomphi = 0.00872664625997165f; // wedge angle, 0.5deg
	float geomAdet = col_pitch*row_pitch; // detector surface area, units: vox^2
    float geomdvox = 1.f; // % voxel side length, units: vox=1

    float ax, ay, az, bx, by, bz, Hvox, costheta, sinthetahalf, ab, anorm2, bnorm2;
    float Sx, Sy, Sz, Dx, Dy, Dz, Vx, Vy, Vz;
    float Dndet, Snsrc, dD2, vV2, sS2;
    float Rdotn, cospsi, geom_src2vox, geom_vox2det, geomproduct, Rayleigh_factor;

    float wvt, sigma_wvt2;

    // Energy bin settings
    float Emin = 8.f; // Left edge of the energy spectrum range
    float Emax = 80.f; // Right edge of the energy spectrum range
    float dE = (Emax-Emin)/__int2float_rn(NRG); // Energy bin width
	float Edet1 = Emin + __int2float_rn(nrg)*dE; // Left edge of the detector energy bin
    float Edet2 = Emin + __int2float_rn(nrg+1)*dE; // Right edge of the detector energy bin
            
    float Esrc1, Esrc2, Esrc;
    float RF, Rq, qaxis_fine;

    /* Compute the expected number of photons and the model matrix Amodel */
    float Nnow = 0.f;
    float Cnow = 0.f; // Compton counter for all detector energy channels
    float Amodelnow[MAT] = { 0.f };

    float ein, Eout1, Eout2, eratio, channel_match, log2wvt, Compton_factor, a1now, a2now, log2aratio;
    int nq, na, matID;

    /* Parameters for computing attenuation */
	float2 avector;
    float Rvox_x, Rvox_y, Rvox_z;
    int numThreads = NWT; // total number of threads in this kernel
    // Maximum number of batches that can be needed to perform a full line integral
    int maxBatch = __float2int_ru(
             sqrtf( __int2float_rn(NX*NX+NY*NY) )/(DQ*__int2float_rn(numThreads))  );
    int lane, wid;
	float Pout, Pin, f1, f2, eout;
    float2 A_vox2det, A_src2vox;
            
    // The widths of log(WVT) and log2(aratio) bins
    float da = (a_max - a_min)/__int2float_rn(NA);
    float dq = (q_max - q_min)/__int2float_rn(NQ);

    int nrg_src, nqf_batch;

    // Add the scattering contributions from all voxels
	for(int nx=0; nx<NX; nx++) {
        for(int ny=0; ny<NY; ny++) {
            
            // Read the material segment ID
            matID = segmentation[nx + ny*NX]-1;
            if (matID>=0) {
                
               // Fetch the (a1,a2) values at the current voxel. This is needed to determine the scattering cross-sections
               avector = tex2D(texRef, Rvox_x, Rvox_y);
               a1now = avector.x;
               a2now = avector.y;
                        
                // Take the ratio of a2/a1, log of base 2, then cap the range
                log2aratio = __log2f(fabsf(a1now/a2now));
                na = __float2int_rd((log2aratio-a_min)/da); // the ID of the nearest bin
                if (na < 0) { na = 0; }
                else if (na >= NA) { na = NA-1; }

                // Pixel coordinates
                Rvox_x = (__int2float_rn(nx)+0.5f)*geomdvox;
                Rvox_y = (__int2float_rn(ny)+0.5f)*geomdvox;   
                        
                // Scattering geometry
                ax = Rvox_x - Rsrc_x;
                ay = Rvox_y - Rsrc_y;

                // Voxel mean z-coordinate
                //Rvox_z = -sqrtf(ax*ax + ay*ay)*tanf(geomphi/2.f);
                Rvox_z = -(ax*nsrc_x + ay*nsrc_y)/sqrtf(nsrc_x*nsrc_x + nsrc_y*nsrc_y)*tanf(geomphi/2.f);

                // Illuminated voxel height
                //Hvox = sqrtf(ax*ax + ay*ay)*tanf(geomphi);
                Hvox = (ax*nsrc_x + ay*nsrc_y)/sqrtf(nsrc_x*nsrc_x + nsrc_y*nsrc_y)*tanf(geomphi);

                az = Rvox_z - Rsrc_z;

                bx = Rdet_x - Rvox_x;
                by = Rdet_y - Rvox_y;
                bz = Rdet_z - Rvox_z;

                ab = (ax*bx + ay*by + az*bz);
                anorm2 = ax*ax + ay*ay + az*az;
                bnorm2 = bx*bx + by*by + bz*bz;

                costheta = ab*rsqrtf(anorm2)*rsqrtf(bnorm2);
                sinthetahalf = sqrtf(fabsf(1.f-costheta)/2.f);

                Sx = bx - ab*ax/anorm2;
                Sy = by - ab*ay/anorm2;
                Sz = bz - ab*az/anorm2;

                Dx = -ax + ab*bx/bnorm2;
                Dy = -ay + ab*by/bnorm2;
                Dz = -az + ab*bz/bnorm2;

                Vx = -(Sx+Dx);
                Vy = -(Sy+Dy);
                Vz = -(Sz+Dz);

                // Compute the mean square product <(d*D)^2>
                Dndet = Dx*ndet_x + Dy*ndet_y + Dz*ndet_z;
                Snsrc = Sx*nsrc_x + Sy*nsrc_y + Sz*nsrc_z;
                dD2 = (Dx*Dx + Dy*Dy + Dz*Dz - Dndet*Dndet)*geomAdet/12.f;
                vV2 = (geomdvox*geomdvox*(Vx*Vx + Vy*Vy) + Hvox*Hvox*Vz*Vz)/12.f;
                sS2 = (Sx*Sx + Sy*Sy + Sz*Sz - Snsrc*Snsrc)*focal_spot*focal_spot/12.f;


                // Compute cos(psi), where psi is the in-plane angle between the source orientation and the source-to-voxel vector
                Rdotn = -sinf(phi)*ax + cosf(phi)*ay;
                cospsi = Rdotn*__frsqrt_rn(ax*ax + ay*ay); // the cos(psi) angle between the source direction and the emitted x-ray
                
                // The geometrical weight factor that goes into the scattering formula
                geom_src2vox = cospsi*__frsqrt_rn(anorm2)*tanf(geomphi);

                // Voxel-to-detector geometrical factor, size (NX*NY,SRC)
                geom_vox2det = -(ndet_x*bx + ndet_y*by) *
                          __frsqrt_rn(bnorm2*bnorm2*bnorm2);

                // Rayleigh angular factor for unpolarized photons
                Rayleigh_factor = (1.f + costheta*costheta)/2.f;
                
                // Evaluate the product of all geometrical and attenuation quantities
                geomproduct = geom_src2vox*geom_vox2det;


                /*************** BEGIN ATTENUATION COMPUTATION ***************/
                A_src2vox = attenuator(Rdet_x, Rdet_y, Rvox_x, Rvox_y, NX, NY, maxBatch);
                A_vox2det = attenuator(Rsrc_x, Rsrc_y, Rvox_x, Rvox_y, NX, NY, maxBatch);
                /**************** END ATTENUATION COMPUTATION ****************/



                // Loop over the source energy bins
                for(nrg_src = nrg_src_min; nrg_src<=nrg_src_max; nrg_src++) {
                      
                    // Left edge of the source energy bin
                    Esrc1 = Emin + __int2float_rn(nrg_src)*dE;

                    // The right edge of the source energy bin
                    Esrc2 = Emin + __int2float_rn(nrg_src+1)*dE;

                    // Average source energy
                    Esrc = 0.5f*(Esrc1+Esrc2);
                    
                    // Compton scattering
                    ein = Esrc/510.99895f;
     
                    // Number of photons detected at the nrg bin
                    RF = shared_spectrum[nrg_src - nrg_src_min];

                    // The average wavevector transfer
                    wvt = 2.f*Esrc/hbarc*sinthetahalf;

                    // Mean square width around the WVT
                    sigma_wvt2 = (2.f*sinthetahalf/hbarc)*(2.f*sinthetahalf/hbarc)*dE*dE/12.f +
                        (Esrc/hbarc)*(Esrc/hbarc)*(sS2 + vV2 + dD2)/(2.f*sinthetahalf)/(2.f*sinthetahalf)/anorm2/bnorm2;
                                
                    // Photoelectic and Compton energy functions
                    f1 = ein*ein*ein;
                    f2 = KN0 + KN1*ein + KN2*ein*ein;

                    // Incoming and outgoing photon survival probabilities
                    Pin  = __expf(-(A_src2vox.x/f1 + A_src2vox.y/f2));
                    Pout = __expf(-(A_vox2det.x/f1 + A_vox2det.y/f2));       
                            
                    // Resolution along the fine q-axis
                    #pragma unroll
                    for (nqf_batch = 0; nqf_batch < NQBatch; nqf_batch++) {
                        // The position along the fine q-axis
                        qaxis_fine = qmin + dq_fine*(0.5f + __int2float_rn(nwt + nqf_batch*NWT));

                        Rq = RF*expf(-(wvt-qaxis_fine)*(wvt-qaxis_fine)/(2.f*sigma_wvt2))*rsqrt2pi/sqrtf(sigma_wvt2)*dq_fine;

                        // Total number of photons
                        Nnow = Nnow + Pin*Pout*geomproduct*Rq*Inow[nqf_batch + matID*NQBatch]*Rayleigh_factor;
                    }
                    

                    // Compute the resolution along the non-equidistant q-axis
                    Rq = RF*expf(-(wvt-qnow)*(wvt-qnow)/(2.f*sigma_wvt2))*rsqrt2pi/sqrtf(sigma_wvt2)*dqnow;

                    // Matrix terms
                    Amodelnow[matID] = Amodelnow[matID] + Pin*Pout*geomproduct*Rq*Rayleigh_factor;
                    
                    // Incoming energy divided by outgoing energy
                    eratio = 1.f + ein*(1.f-costheta);

                    // Outgoing Compton energy in keV
                    eout = ein/eratio;
                            
                    // The left and right edges of the outgoing energy range
                    Eout1 = Esrc1/eratio;
                    Eout2 = Esrc2/eratio;

                    // The overlap between the detector energy channel and the outgoing x-ray photon energy range
                    channel_match = fmaxf(0.f, fminf(Eout2, Edet2) - fmaxf(Eout1, Edet1))/(Eout2-Eout1);
                    
                    // Dimensionless WVT (Rayleigh)
                    wvt = ein*sinthetahalf;

                    // Wavevector transfer (on log2 scale) in case of Compton scattering:
                    log2wvt = __log2f(wvt*__fsqrt_rn(eratio + wvt*wvt)/eratio);

                    // The ID of the nearest WVT bin
                    nq = __float2int_rd((log2wvt - q_min)/dq);
                    if (nq<0) { nq = 0; }
                    else if (nq >= NQ) { nq = NQ-1; }


                    // photoelectic and Compton energy functions
                    f1 = eout*eout*eout;
                    f2 = KN0 + KN1*eout + KN2*eout*eout;

                    // Update the outgoing photon survival probability
                    Pout = __expf(-(A_vox2det.x/f1 + A_vox2det.y/f2));

                    // Compton angular factor
                    Compton_factor = eratio*eratio*(eratio + 1.f/eratio - 1.f + costheta*costheta)/2.f;

                    // Compute the photon transmission probability at a given x-ray energy
                    Cnow = Cnow + channel_match* // only add if the photon energy falls in the current detector channel
                            Pin*Pout* // outgoing photon survival probability
                            RF*geomproduct*Compton_factor* // spectrum, geometry, and physics
                            a2now*__int2float_rn((int) S2[nq + na*NQ]); // differential scattering cross-section
                    
                }
            }
        }
    }

    // Write the output to the global memory
	for(matID=0; matID<MAT; matID++) {
        Amatrix[nwt + matID*NWT + src*NWT*MAT + det*NWT*MAT*SRC + nrg*NWT*MAT*SRC*DET] = Amodelnow[matID];
    }

    // Sum the number of photons across all threads, i.e. over all q-values within a window sigma_q
    // Based on this documentation: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
    // and this: https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/

    // The lane of each thread in the warp (0,31)
    lane = threadIdx.x % warpSize;

    // The warp ID in the thread block (0,31)
    wid = threadIdx.x / warpSize;

    // First, sum all lanes within each warp
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        Nnow += __shfl_down_sync(FULL_MASK, Nnow, offset);
    }

    // The first thread in each warp holds the reduced value and writes it to shared memory
    if (lane==0) {
        Nshared[wid] = Nnow;
    }

    __syncthreads(); // Wait for all partial reductions

    // 256 threads, 32 lanes, 8 warps

    // Reduce within the first warp
    if (wid == 0) {

        // The first numWarps threads read from the shared memory. Others default to zero
        Nnow = (lane < numWarps) ? Nshared[lane] : 0.f;

        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            Nnow += __shfl_down_sync(FULL_MASK, Nnow, offset);
        }

        // The first lane has the full sum. Write it to global memory
        if (lane == 0) {
            Nphotons[src + det*SRC + nrg*SRC*DET] = Nnow;   
        }
        else if(lane == 1) {
            Cphotons[src + det*SRC + nrg*SRC*DET] = Cnow;
        }
    }


//     // Shared memory for the 2x32 partial sums of (A1,A2)
//     static __shared__ float shared1[numWarps];
//     static __shared__ float shared2[numWarps];
//
//     // Shared memory for the scattering cross-section information
//     static __shared__ unsigned char S1[NQ*NA];
//     static __shared__ unsigned char S2[NQ*NA];
//
//     // Shared memory for the source spectrum information
// 	static __shared__ float spectrum[NRG*(NPSI+1)]; // number of photons per bin width
//
//     int const threadID = threadIdx.x; // thread ID
//     int const col = blockIdx.x; // detector column index, range [0,COL-1]
//
//     // Some hard-coded geometry information
//     //float Rdet_z = 1.4375f; // Detector z-coordinate with respect to the source, in cm
//     float tanhalfphi = tanf(phiwedge/2.f);
//     float tanphi = tanf(phiwedge);
//
//
//     // Maximum number of batches that can be needed to perform a full line integral
//     int maxBatch = __float2int_ru(
//             sqrtf( __int2float_rn(NX*NX+NY*NY) )/(DQ*__int2float_rn(numThreads))  );
//
//     /* Compute the 3D position of the current query point, in normalized coordinates */
//
//
//     float Rvox_x, Rvox_y, Rvox_z, Lx, Ly, Lr, rLnorm, qr, qx, qy, a1, a2, A1_vox2det, A2_vox2det, A1_src2vox, A2_src2vox;
//     float Pin, Pout, Rsrc2vox_x, Rsrc2vox_y, Rsrc2vox_z, Rvox2det_x, Rvox2det_y, Rvox2det_z, geom_src2vox, geom_vox2det, costheta, geomproduct, Rvox2det2, Rsrc2vox_rnorm;
//     float a1now, a2now, sinhalftheta, Rdotn;
//     float ein, eout, eratio, Rayleigh_factor, Compton_factor, numPhotons, f1, f2, cospsi;
//     float Ptotal1[NsrcBatch][CHL] = { 0.f }; // Coherent total number of photons, summed over all voxels. Size [2048x256x4]
//     float Ptotal2[NsrcBatch][CHL] = { 0.f }; // Incoherent total number of photons, summed over all voxels. Size [2048x256x4]
//     float Rsrc_x[NsrcBatch], Rsrc_y[NsrcBatch], nsrc_x[NsrcBatch], nsrc_y[NsrcBatch];
//     float2 r_int, avector;
//     float log2aratio, log2wvt, greater_than_20keV, wvt;
//
//     bool end_of_line;
//     int batchID, q, wid, lane, offset, src, src_batch, chl, nrg, cospsiID, nq, na, nx, ny, readnow, warpID;
//
//     // The widths of log(WVT) and log2(aratio) bins
//     float da = (a_max - a_min)/__int2float_rn(NA);
//     float dq = (q_max - q_min)/__int2float_rn(NQ);
//
//
//     /*** ---------------------------------------------------------------------------------- ***/
//     /*** READ INPUT DATA FROM GLOBAL MEMORY (SPECTRUM, GEOMETRY, SCATTERING CROSS-SECTIONS) ***/
//     /*** ---------------------------------------------------------------------------------- ***/
//
//     // Read the detector position (in vox units), as well as orientation (dimensionless unit vector)
//     float Rdet_x = Rdet[col];
//     float Rdet_y = Rdet[col + COL];
//     float ndet_x = ndet[col];
//     float ndet_y = ndet[col + COL];
//
//
//     // Read the source geometry info. Each GPU thread is assigned NsrcBatch=256/128=2 number of x-ray sources
//     #pragma unroll
//     for(src_batch=0; src_batch<NsrcBatch; src_batch++) {
//         src = threadID + src_batch*numThreads;
//         Rsrc_x[src_batch] = Rsrc[src];
//         Rsrc_y[src_batch] = Rsrc[src+SRC];
//         nsrc_x[src_batch] = nsrc[src];
//         nsrc_y[src_batch] = nsrc[src+SRC];
//     }
//
//
//     // Read the scattering cross-section tables, each of size [NQ,NA]
//     readnow = threadID; // reset counter
// 	int S_temp;
//     while (readnow < (NQ*NA)) {
//         S_temp = __float2int_rd(S1_data[readnow]);
//         if (S_temp<0) { S_temp = 0; }
//         else if (S_temp > 255) { S_temp = 255; }
//         S1[readnow] = (unsigned char) S_temp;
//
//         S_temp = __float2int_rd(S2_data[readnow]);
//         if (S_temp<0) { S_temp = 0; }
//         else if (S_temp > 255) { S_temp = 255; }
//         S2[readnow] = (unsigned char) S_temp;
//
//         readnow += numThreads;
//     }
//
//     // Read the photon spectrum info. Use multiple batches if there are not enough threads for each bin
//     readnow = threadID; // reset counter
//     while (readnow < (NRG*(NPSI+1))) {
//         spectrum[readnow] = spectrum_data[readnow];
//         readnow += numThreads;
//     }
//     /*** END OF READ ***/
//
//
//
//     // ------------------------------------------------------------------------- //
//
//
//
//     /*** ------------------------------------------ ***/
//     /*** LOOP OVER ALL VOXELS AND ADD PHOTON COUNTS ***/
//     /*** ------------------------------------------ ***/
//     for(nx=0; nx<NX; nx++) {
//         // Pixel center x-position, in absolute coordinates, using vox units
//         Rvox_x = __int2float_rn(nx) + 0.5f;
//         for(ny=0; ny<NY; ny++) {
//             // Pixel center y-position, in absolute coordinates, using vox units
//             Rvox_y = __int2float_rn(ny) + 0.5f;
//
//             // Fetch the (a1,a2) values at the current voxel. This is needed to determine the scattering cross-sections
//             avector = tex2D(texRef, Rvox_x, Rvox_y);
//             a1now = avector.x;
//             a2now = avector.y;
//
//             // Proceed only if the pixel is sufficiently above zero
//             if (fabsf(a1now)>1E-9f && fabsf(a2now)>0.f) {
//
//                 /*** PART ONE: fetch and pre-sum the values of (a1,a2) along the voxel-detector ray, at the level of a single thread ***/
//
//                 // Take the ratio of a2/a1, log of base 2, then cap the range
//                 log2aratio = __log2f(fabsf(a1now/a2now));
//                 na = __float2int_rd((log2aratio-a_min)/da); // the ID of the nearest bin
//                 if (na < 0) { na = 0; }
//                 else if (na >= NA) { na = NA-1; }
//
//                 // Find the intersection coordinates of the voxel-detector line with the ROI perimeter
//                 r_int = GetIntersection(Rdet_x, Rdet_y, Rvox_x, Rvox_y, NX, NY);
//
//                 // Vector from the intersection point to the pixel center
//                 Lx = Rvox_x - r_int.x; // x-component
//                 Ly = Rvox_y - r_int.y; // y-component
//                 Lr = __fsqrt_rn(Lx*Lx + Ly*Ly); // amplitude
//
//                 // The reciprocal of the norm, 1/sqrt(Lx^2 + Ly^2)
//                 rLnorm = __frsqrt_rn(Lx*Lx + Ly*Ly);
//
//                 // Normalize, to obtain a unit x-ray vector
//                 Lx = Lx*rLnorm;
//                 Ly = Ly*rLnorm;
//
//                 // Reset the starting values for the attenuation sum
//                 a1 = 0.f;
//                 a2 = 0.f;
//                 end_of_line = false;
//                 batchID = 0;
//
//                 // Fetch the (a1,a2) values along the ray, batch-by-batch
//                 while (end_of_line == false && batchID<maxBatch) {
//                     // Query point index
//                     q = threadID + batchID*numThreads;
//
//                     // Query point distance from the source, voxel units
//                     qr = DQ*__int2float_rn(q);
//
//                     // Proceed if the query point does fall beyond the voxel position
//                     if (qr<=Lr) {
//                         // Query point position, in pixel units
//                         qx = (r_int.x + Lx*qr);
//                         qy = (r_int.y + Ly*qr);
//
//                         // Fetch the (a1,a2) values from the texture
//                         avector = tex2D(texRef, qx, qy);
//
//                         // Convert to individual 32bit floats and add to the tallies
//                         a1 = a1 + avector.x;
//                         a2 = a2 + avector.y;
//                     }
//                     else {
//                         end_of_line = true;
//                     }
//                     // Increment the batch counter
//                     batchID = batchID + 1;
//                 }
//
//
//                 /*** PART DEUX: sum the fetched values across the whole thread block ***/
//                 // Based on this documentation: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
//
//                 // The lane of each thread in the warp (0,31)
//                 lane = threadIdx.x % warpSize;
//
//                 // The warp ID in the thread block (0,31)
//                 wid = threadIdx.x / warpSize;
//
//                 // First, sum all lanes within each warp
//                 for (offset = warpSize/2; offset > 0; offset /= 2) {
//                     a1 += __shfl_down_sync(FULL_MASK, a1, offset);
//                     a2 += __shfl_down_sync(FULL_MASK, a2, offset);
//                 }
//
//                 // The first thread in the warp holds the reduced value
//                 // and writes it to shared memory
//                 if (lane==0) {
//                     shared1[wid] = a1;
//                     shared2[wid] = a2;
//                 }
//
//                 __syncthreads(); // Wait for all partial reductions
//
//                 // Secondly, sum across all warps
//                 A1_vox2det = shared1[0];
//                 A2_vox2det = shared2[0];
//                 for(warpID = 1; warpID<numWarps; warpID++) {
//                     A1_vox2det += shared1[warpID];
//                     A2_vox2det += shared2[warpID];
//                 }
//                 // Multiply by the step size. All threads have access to this:
//                 A1_vox2det = A1_vox2det*DQ;
//                 A2_vox2det = A2_vox2det*DQ;
//
//
//                 /*** PART TROIS: the scattering triangle source-voxel-detector ***/
//
//
//                 // Loop over all sources (each of the 128 threads works on 2 sources)
//                 #pragma unroll
//                 for(src_batch=0; src_batch<NsrcBatch; src_batch++) {
//
//                     src = threadID + src_batch*numThreads; // source index, range [0,255]
//
//                     // Each thread now reads the line integrals on the src2vox path, stored in global memory
//                     A1_src2vox =  A1[src + nx*SRC + ny*SRC*NX];
//                     A2_src2vox =  A2[src + nx*SRC + ny*SRC*NX];
//
//                     // Source-to-voxel geometrical factor, size (NX*NY,SRC)
//                     Rsrc2vox_x = Rvox_x - Rsrc_x[src_batch];
//                     Rsrc2vox_y = Rvox_y - Rsrc_y[src_batch];
//
//                     // Horizontal distance from the source to the voxel
//                     Rdotn = nsrc_x[src_batch]*Rsrc2vox_x + nsrc_y[src_batch]*Rsrc2vox_y;
//                     Rvox_z = Rdotn*tanhalfphi; // voxel position along the z-axis;
//                     Rsrc2vox_z = Rvox_z; // source assumed to be at z=0
//
//
//                     // Compute cos(psi), where psi is the in-plane angle between the source orientation and the source-to-voxel vector
//                     Rsrc2vox_rnorm = __frsqrt_rn(Rsrc2vox_x*Rsrc2vox_x + Rsrc2vox_y*Rsrc2vox_y); // the reciprocal square root of the source-to-voxel distance
//                     cospsi = Rdotn*Rsrc2vox_rnorm; // the cos(psi) angle between the source direction and the emitted x-ray
//                     // Make sure that cos(psi) is in the range [0,1], just in case some roundoff error occurs
//                     if (cospsi<0.f) { cospsi = 0.f; }
//                     else if (cospsi>1.f) { cospsi = 1.f; }
//
//                     // Update the Rsrc2vox_rnorm to full 3D
//                     Rsrc2vox_rnorm = __frsqrt_rn(Rsrc2vox_x*Rsrc2vox_x + Rsrc2vox_y*Rsrc2vox_y + Rsrc2vox_z*Rsrc2vox_z);
//
//                     // The geometrical weight factor that goes into the scattering formula
//                     geom_src2vox = cospsi*Rsrc2vox_rnorm;
//
//
//                     // Voxel-to-detector geometrical factor, size (NX*NY,SRC)
//                     Rvox2det_x = Rdet_x - Rvox_x;
//                     Rvox2det_y = Rdet_y - Rvox_y;
//                     Rvox2det_z = Rdet_z - Rvox_z;
//
//                     Rvox2det2 = Rvox2det_x*Rvox2det_x + Rvox2det_y*Rvox2det_y + Rvox2det_z*Rvox2det_z;
//                     geom_vox2det = -(ndet_x*Rvox2det_x + ndet_y*Rvox2det_y) *
//                          __frsqrt_rn(Rvox2det2*Rvox2det2*Rvox2det2);
//
//
//                     // Scattering angle theta in terms of cos(theta), range [-1,1]
//                     costheta = (Rsrc2vox_x*Rvox2det_x + Rsrc2vox_y*Rvox2det_y + Rsrc2vox_z*Rvox2det_z)
//                             * __frsqrt_rn(Rvox2det2)*Rsrc2vox_rnorm;
//
//                     // The scattering angle theta in terms of sin(theta/2). The range is [0,1]. The negative portion of the range is irrelevant, since scattering is symmetric around zero angle.
//                     sinhalftheta = __fsqrt_rn(fmaxf(1.f-costheta, 0.f)/2.f);
//
//                     // Rayleigh angular factor for unpolarized photons
//                     Rayleigh_factor = (1.f + costheta*costheta)/2.f;
//
//                     // Evaluate the product of all geometrical and attenuation quantities
//                     geomproduct = geom_src2vox*geom_vox2det;
//
//                     // Select the nearest tabulated emission angle, in terms of cos(psi)
//                     cospsiID = __float2int_rn(cospsi*__int2float_rn(NPSI-1)); // range shoud be [0,NPSI-1]
//
//                     /*** PART QUATRE: loop over all energy bins and sum into detector channels ***/
//                     for (nrg = 0; nrg<NRG; nrg++) {
//
//                         // Select the energ bin edges
//                         ein = spectrum[nrg]; // average energy of the bin, keV
//
//                         // Determine the Rayleigh energy channel
//                         chl = energy_channel(ein);
//
//                         // Energy basis function^{-1} values
//                         f1 = ein*ein*ein;
//                         f2 = KN0 + KN1*ein + KN2*ein*ein;
//
//                         // Compute the wavevector transfer, dimensionless. Must multiply by (2*E)/(hbar*c) to obtain WVT in units of 1/Angstrom
//                         // The log(wavevector transfer in 1/Angstrom)
//                         // The ID of the nearest WVT bin
//                         wvt = ein*sinhalftheta;
//                         nq = __float2int_rd((__log2f(wvt)-q_min)/dq);
//                         if (nq<0) { nq = 0; }
//                         else if (nq >= NQ) { nq = NQ-1; }
//
//                         // Photon survival probabilities
//                         Pin  = __expf(-(A1_src2vox/f1 + A2_src2vox/f2)); // incoming
//                         Pout = __expf(-(A1_vox2det/f1 + A2_vox2det/f2)); // outgoing Rayleight
//                         if (Pin  > 1.f) { Pin = 1.f;  }
//                         if (Pout > 1.f) { Pout = 1.f; }
//
//                         // The number of photons in this bin (spectrum multiplied by the bin width in keV)
//                         numPhotons = spectrum[nrg + cospsiID*NRG + NRG]*geomproduct*a2now*Pin;
//
//                         // Compute the photon transmission probability at a given x-ray energy
//                         Ptotal1[src_batch][chl] = Ptotal1[src_batch][chl] +
//                                 Pout*
//                                 numPhotons*Rayleigh_factor*
//                                 __int2float_rn((int) S1[nq + na*NQ]); // differential scattering cross-section
//
//                         // Incoming energy divided by outgoing energy
//                         eratio = 1.f + ein*(1.f-costheta);
//
//                         // Outgoing Compton energy in keV
//                         eout = ein/eratio;
//
//                         // If the photon energy is less than 20 keV, it will be assigned to chl=0, which corresponds to the range [20,50].
//                         // However, we apply a weight of 0.f, so that this photon does not add to the tally
//                         greater_than_20keV = (eout >= 0.0391390236711837f) ? 1.f : 0.f;
//
//                         // Detector energy channel, range 0-3
//                         chl = energy_channel(eout);
//
//                         // Wavevector transfer (on log2 scale) in case of Compton scattering:
//                         log2wvt = __log2f(wvt*__fsqrt_rn(eratio + wvt*wvt)/eratio);
//
//                          // The ID of the nearest WVT bin
//                          nq = __float2int_rd((log2wvt - q_min)/dq);
//                          if (nq<0) { nq = 0; }
//                          else if (nq >= NQ) { nq = NQ-1; }
//
//
//                          // photoelectic and Compton energy functions
//                          f1 = eout*eout*eout;
//                          f2 = KN0 + KN1*eout + KN2*eout*eout;
//
//                          // Outgoing photon survival probability
//                          Pout = __expf(-(A1_vox2det/f1 + A2_vox2det/f2));
//                          if (Pout > 1.f) { Pout = 1.f; }
//
//                          // Compton angular factor
//                          Compton_factor = eratio*eratio*(eratio + 1.f/eratio - 1.f + costheta*costheta)/2.f;
//
//                          // Compute the photon transmission probability at a given x-ray energy
//                          Ptotal2[src_batch][chl] = Ptotal2[src_batch][chl] +
//                                  greater_than_20keV* // do not count photons of less than 20 kV energy
//                                  Pout* // survival probability
//                                  numPhotons*Compton_factor* // spectrum, geometry, and physics
//                                  __int2float_rn((int) S2[nq + na*NQ]); // differential scattering cross-section
//                     }
//                 }
//             }
//         }
//     }
//
//     // Write the output to the global memory
//     #pragma unroll
//     for(src_batch=0; src_batch<NsrcBatch; src_batch++) {
//         src = threadID + src_batch*numThreads;
//         for(chl=0; chl<CHL; chl++) {
//             Nphotons1[src + col*SRC + chl*SRC*COL] = Ptotal1[src_batch][chl]*tanphi;
//             Nphotons2[src + col*SRC + chl*SRC*COL] = Ptotal2[src_batch][chl]*tanphi;
//         }
//     }
}

