// Fast X-ray diffraction (XRD) tomography for enhanced identification of materials
// by Airidas Korolkovas, Scientific Reports volume 12, Article number: 19097 (2022) 
// https://doi.org/10.1038/s41598-022-23396-2
// airidas.korolkovas89@gmail.com
// Last modified: 11/19/2022


// To compile the forward projector, run this line in the Matlab Command Window
// mexcuda Forward_XRD.cu

/* Include headers */
#include "mex.h"
#include "gpu/mxGPUArray.h"

        
/* Define the problem dimensions */
#define SRC 32 // number of x-ray sources
#define COL 1024 // number of detector columns
#define ROW 1 // number of detector rows
#define DET COL*ROW // total number of detector pixels
#define NRG 64 // number of energy channels of the detector
#define DNRG 5 // the half-width of the energy bin range that is considered non-zero, i.e. use 11 bins around the central line of the spectrum table
#define NWT 256 // number of wavevector transfer values to be reconstructed
#define NQF 768 // number of q-bins at high resolution (for ground truth forward projection)
#define MAT 3 // number of distinct materials to be reconstructed
        
/* Technical constants */
#define FULL_MASK 0xffffffff // needed for shuffling warps
#define warpSize 32 // number of threads in a warp = 32, fixed
#define maxNumThreads 1024 // maximum number of threads per block, fixed by hardware
// #define numThreads 128 // number of threads used for source-to-voxel line integrals
#define numWarps NWT/warpSize // number of warps in one thread block
#define NQBatch NQF/NWT // number of batches that NWT threads go through to process NQF high-resolution bins
#define rsqrt2pi 0.398942280401433f // = 1/sqrt(2*pi)
#define twopi 6.28318530717959f // = 2*pi
#define hbarc 1.973269804f // Planck's constant in keV*Angstrom

/* Define the q-axis limits for the reconstruction grid */
#define qmin 0.5f // starting edge
#define qmax 6.0f // ending edge
#define dq0 0.01f // bin width at the first edge
#define dq_fine (qmax-qmin)/__int2float_rn(NQF) // Compute the fine q-axis for forward projection

#define DQ 0.25f // Line integral step size in voxel units
     
/* The size of scattering cross-section table */
#define NQ 64 // number of q-bin (wavevector transfer values)
#define NA 64 // number of a-bins (a2/a1/1000)
         
/* Klein-Nishina function^{-1} parameters, fitted to a 2nd degree polynomial */
#define KN0 0.751691286166726f
#define KN1 1.4599683207703f
#define KN2 -0.64597532690936f


// Set up a 2D float texture reference
texture<float2, cudaTextureType2D, cudaReadModeElementType> texRef;

/*
 * Device code
 */

// /* Include the kernels */
#include "Forward_projector_kernels\Copy2Vector.cu"
#include "Forward_projector_kernels\GetIntersection.cu"
#include "Forward_projector_kernels\attenuator.cu"
#include "Forward_projector_kernels\find_src_nrg_range.cu"
#include "Forward_projector_kernels\wvt_value.cu"
#include "Forward_projector_kernels\integrate_diffraction.cu"

/*
 * Host code
 */


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    // Input 1: Source spectrum, size [NRG,1]
    mxGPUArray *spectrum = mxGPUCopyFromMxArray(prhs[0]);
    float *d_spectrum = (float *)(mxGPUGetData(spectrum));

    // Input 2: High-resolution XRD data for forward projection, size [NQF,1]
    mxGPUArray *Ifine = mxGPUCopyFromMxArray(prhs[1]);
    float *d_Ifine = (float *)(mxGPUGetData(Ifine));

    // Input 3: The segmentation map of the phantom, size [NX,NY]
    mxGPUArray *segmentation = mxGPUCopyFromMxArray(prhs[2]);
    int *d_segmentation = (int *)(mxGPUGetData(segmentation));

    // Input 4: photoelectric coefficient, size [NX,NY]
    mxGPUArray *a1 = mxGPUCopyFromMxArray(prhs[3]);
    float *d_a1 = (float *)(mxGPUGetData(a1));
 
    // Input 5: Compton coefficient, size [NX,NY]
    mxGPUArray *a2 = mxGPUCopyFromMxArray(prhs[4]);
    float *d_a2 = (float *)(mxGPUGetData(a2));
    
    // Input 6: Coherent scattering cross-sections
 	mxGPUArray *S2 = mxGPUCopyFromMxArray(prhs[5]);
    float *d_S2 = (float *)(mxGPUGetData(S2));
 
    // Scattering cross-section tables' limits
    float const q_min = static_cast<float>(mxGetScalar(prhs[6]));
    float const q_max = static_cast<float>(mxGetScalar(prhs[7]));
    float const a_min = static_cast<float>(mxGetScalar(prhs[8]));
    float const a_max = static_cast<float>(mxGetScalar(prhs[9]));
// 
//     // Additional geometry inputs
//     float const zdet = static_cast<float>(mxGetScalar(prhs[13])); // detector z-coordinate in units of vox
//     float const phiwedge = static_cast<float>(mxGetScalar(prhs[14])); // source wedge opening angle in units of rad
// 
     // Extract the grid size of the phantom
     const mwSize * dims_phantom = mxGPUGetDimensions(segmentation);
     int const NX = dims_phantom[0];
     int const NY = dims_phantom[1];
 
     // Intermediate arrays A1 and A2 - the summed line integrals, size (SRC, NX*NY) x 2
     mxGPUArray *Nphotons, *Cphotons, *Amatrix;
     float *d_Nphotons, *d_Cphotons, *d_Amatrix;

//     mxGPUArray *A1, *A2, *Nphotons1, *Nphotons2, *Tphotons, *gain0;
//     float *d_A1, *d_A2, *d_Nphotons1, *d_Nphotons2, *d_Tphotons, *d_gain0;
//     const mwSize dims1[] = { SRC, NX, NY };
//     A1 = mxGPUCreateGPUArray(
//             3, // number of dimensions
//             dims1, // array listing the size of each dimension
//             mxSINGLE_CLASS, // data class
//             mxREAL, // data complexity, real vs complex
//             MX_GPU_INITIALIZE_VALUES); // initialize values to 0
// 
//     A2 = mxGPUCreateGPUArray(
//             3, // number of dimensions
//             dims1, // array listing the size of each dimension
//             mxSINGLE_CLASS, // data class
//             mxREAL, // data complexity, real vs complex
//             MX_GPU_INITIALIZE_VALUES); // initialize values to 0
	
    // Output 1: The forward projection for the number of XRD photons
    const mwSize dims1[] = { SRC, COL, ROW, NRG };
    Nphotons = mxGPUCreateGPUArray(
            4, // number of dimensions
            dims1, // array listing the size of each dimension
            mxSINGLE_CLASS, // data class
            mxREAL, // data complexity, real vs complex
            MX_GPU_INITIALIZE_VALUES); // initialize values to 0

	// Output 2: The forward projection for the number of Compton photons
    Cphotons = mxGPUCreateGPUArray(
            4, // number of dimensions
            dims1, // array listing the size of each dimension
            mxSINGLE_CLASS, // data class
            mxREAL, // data complexity, real vs complex
            MX_GPU_INITIALIZE_VALUES); // initialize values to 0

    // Output 3: The model matrix
    const mwSize dims2[] = { NWT, MAT, SRC, COL, ROW, NRG };
    Amatrix = mxGPUCreateGPUArray(
            6, // number of dimensions
            dims2, // array listing the size of each dimension
            mxSINGLE_CLASS, // data class
            mxREAL, // data complexity, real vs complex
            MX_GPU_INITIALIZE_VALUES); // initialize values to 0
                    
//                     
// 	// Output 1 and 2 - the number of scattered photons (Rayleigh and Compton)
//     const mwSize dims2[] = { SRC, COL, CHL };
//     Nphotons1 = mxGPUCreateGPUArray(
//             3, // number of dimensions
//             dims2, // array listing the size of each dimension
//             mxSINGLE_CLASS, // data class
//             mxREAL, // data complexity, real vs complex
//             MX_GPU_INITIALIZE_VALUES); // initialize values to 0
//                     
//     Nphotons2 = mxGPUCreateGPUArray(
//             3, // number of dimensions
//             dims2, // array listing the size of each dimension
//             mxSINGLE_CLASS, // data class
//             mxREAL, // data complexity, real vs complex
//             MX_GPU_INITIALIZE_VALUES); // initialize values to 0
// 
//     // Output 3 - Transmitted photons for the four energy channels
//     Tphotons = mxGPUCreateGPUArray(
//             3, // number of dimensions
//             dims2, // array listing the size of each dimension
//             mxSINGLE_CLASS, // data class
//             mxREAL, // data complexity, real vs complex
//             MX_GPU_INITIALIZE_VALUES); // initialize values to 0
//                    
//     // Output 4 - gain0 for the four energy channels (transmission when the phantom is absent)
//     gain0 = mxGPUCreateGPUArray(
//             3, // number of dimensions
//             dims2, // array listing the size of each dimension
//             mxSINGLE_CLASS, // data class
//             mxREAL, // data complexity, real vs complex
//             MX_GPU_INITIALIZE_VALUES); // initialize values to 0
    
//     // Fetch the pointers
    d_Nphotons       = (float *)(mxGPUGetData(Nphotons));
    d_Cphotons       = (float *)(mxGPUGetData(Cphotons));
    d_Amatrix        = (float *)(mxGPUGetData(Amatrix));
    
    
 
     /* Convert the two scalar (a1,a2) density fields to a single 2D vector field */
     float2 *d_avector;
     cudaMalloc((void **)&d_avector, NX*NY*sizeof(float2));
     int const numCopyBlocks = 1 + (NX*NY-1)/maxNumThreads; // integer division with rounding up
     Copy2Vector<<<numCopyBlocks, maxNumThreads>>>(d_a1, d_a2, d_avector, NX, NY);
 
 
 	/* Set up a 2x32 bit memory channel for (a1,a2) and allocate a corresponding 2D CUDA Array */
     cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
     cudaArray* aCUDA;
     cudaMallocArray(&aCUDA, &channelDesc, NX, NY);
     cudaMemcpy2DToArray(aCUDA, // Destination memory address
                     0, // Destination starting X offset
                     0, // Destination starting Y offset
                     (void *) d_avector, // Source memory address
                     sizeof(float2)*NX, //  Pitch of source memory
                     sizeof(float2)*NX, // Width of matrix transfer (columns in bytes)
                     NY, // Height of matrix transfer (rows)
                     cudaMemcpyDeviceToDevice); // Type of transfer
 
 
      /* Set up texture reference parameters */
      texRef.addressMode[0] = cudaAddressModeBorder; // returns zero for queries outside the recon volume
      texRef.addressMode[1] = cudaAddressModeBorder; // returns zero for queries outside the recon volume
      texRef.filterMode     = cudaFilterModeLinear; // bi-linear interpolation
      texRef.normalized     = false; // use absolute coordinates
 
 
     /* Bind the texture to the 2D CUDA array */
     cudaBindTextureToArray(texRef, aCUDA);

        
    /*** Launch a 3D set of blocks ***/
    // x - x-ray source ID
	// y - phantom pixel along x-axis
	// z - phantom pixel along y-axis
    dim3 blockdims(SRC, DET, NRG);
	//dim4 blockdims(SRC, COL, ROW, NRG);
    integrate_diffraction<<<blockdims, NWT>>>(
            d_spectrum,
            d_Nphotons,
            d_Amatrix,
            d_Ifine,
            d_segmentation,
            NX,
            NY,
            d_Cphotons,
            d_S2,
            q_min,
            q_max,
            a_min,
            a_max);

    
    
// 	// Compute the attenuation line integrals (A1,A2) from each source to each voxel
//     dim3 blockdims(SRC,NX,NY);
//     integrate_src2vox<<<blockdims, numThreads>>>(d_A1, d_A2, d_Rsrc, NX, NY);
//  
//     // Compute the attenuation line integrals from each voxel to each detector
// 	// Multiply with the source-voxel survival probabilities, determine the scattering cross-sections, and sum over all voxels
//     integrate_vox2det<<<COL, numThreads>>>(d_A1, d_A2, d_Rsrc, d_nsrc, d_Rdet, d_ndet, d_Nphotons1, d_Nphotons2, d_S1, d_S2, NX, NY, d_spectrum_data, q_min, q_max, a_min, a_max, zdet, phiwedge);
// 	 
//     // Compute the line integrals (A1,A2) from each source to each detector, and store them temporarily in Tphotons[SRC,COL,0] and Tphotons[SRC,COL,1]
//     dim3 Tdims(SRC,COL);
//     integrate_src2det<<<Tdims, numThreads>>>(d_Rsrc, d_Rdet, d_Tphotons, NX, NY);
// 
//     // Read the (A1,A2) line integrals stored in Tphotons, and add the energy dependence. Thread = source ID, block = detector column. Each thread loops over all energy channels and writes the four answers.
//     polychromatic_src2det<<<COL, SRC>>>(d_Rsrc, d_nsrc, d_Rdet, d_ndet, d_Tphotons, d_gain0, d_spectrum_data);



    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(Nphotons);
    plhs[1] = mxGPUCreateMxArrayOnGPU(Cphotons);
    plhs[2] = mxGPUCreateMxArrayOnGPU(Amatrix);
    
    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(spectrum);
    mxGPUDestroyGPUArray(Ifine);
    mxGPUDestroyGPUArray(Nphotons);
    mxGPUDestroyGPUArray(Amatrix);
    mxGPUDestroyGPUArray(segmentation);
    mxGPUDestroyGPUArray(a1);
    mxGPUDestroyGPUArray(a2);
    mxGPUDestroyGPUArray(Cphotons);
    mxGPUDestroyGPUArray(S2);
    cudaUnbindTexture(texRef);
    cudaFreeArray(aCUDA);
    cudaFree(d_avector);
    
}