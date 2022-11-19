// Fast X-ray diffraction (XRD) tomography for enhanced identification of materials
// by Airidas Korolkovas, Scientific Reports volume 12, Article number: 19097 (2022) 
// https://doi.org/10.1038/s41598-022-23396-2
// airidas.korolkovas89@gmail.com
// Last modified: 11/19/2022


/* This kernel transforms the two (water,bone) linear arrays inherited from Matlab, to a single 2-vector field for faster CUDA processing. */
__global__ void Copy2Vector(float *a1, float *a2, float2 *avector, int const NX, int const NY) {
    // Pixel index
    int const nxny = threadIdx.x + blockIdx.x*maxNumThreads;
    if (nxny<(NX*NY)) {
        // Define the output
        float2 afull;

        // Read the two values
        afull.x = a1[nxny]; // photoelectric
        afull.y = a2[nxny]; // Compton

        // Write to global memory
        avector[nxny] = afull;
    }
}