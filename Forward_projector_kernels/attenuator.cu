// Fast X-ray diffraction (XRD) tomography for enhanced identification of materials
// by Airidas Korolkovas, Scientific Reports volume 12, Article number: 19097 (2022) 
// https://doi.org/10.1038/s41598-022-23396-2
// airidas.korolkovas89@gmail.com
// Last modified: 11/19/2022


/*** Compute the attenuation line integrals ***/
__device__ float2 attenuator(float Rdet_x, float Rdet_y, float Rvox_x, float Rvox_y, int const NX, int const NY, int const maxBatch)
{

    // Shared memory for the 2x32 partial sums of (A1,A2)
    static __shared__ float shared1[numWarps];
    static __shared__ float shared2[numWarps];

    float2 A_vox2det;

    // Find the intersection coordinates of the voxel-detector line with the ROI perimeter
    float2 r_int = GetIntersection(Rdet_x, Rdet_y, Rvox_x, Rvox_y, NX, NY);

    // Vector from the intersection point to the pixel center
    float Lx = Rvox_x - r_int.x; // x-component
    float Ly = Rvox_y - r_int.y; // y-component
    float Lr = __fsqrt_rn(Lx*Lx + Ly*Ly); // amplitude

    // The reciprocal of the norm, 1/sqrt(Lx^2 + Ly^2)
    float rLnorm = __frsqrt_rn(Lx*Lx + Ly*Ly);

    // Normalize, to obtain a unit x-ray vector
    Lx = Lx*rLnorm;
    Ly = Ly*rLnorm;

    // Reset the starting values for the attenuation sum
	float a1 = 0.f;
    float a2 = 0.f;
    bool end_of_line = false;
	int batchID = 0, numThreads = NWT;
    int q;
    float qr, qx, qy;
    float2 avector;
    

    // Fetch the (a1,a2) values along the ray, batch-by-batch
    while (end_of_line == false && batchID<maxBatch) {
        // Query point index
        q = threadIdx.x + batchID*numThreads;

        // Query point distance from the source, voxel units
        qr = DQ*__int2float_rn(q);

        // Proceed if the query point does fall beyond the voxel position
        if (qr<=Lr) {
            // Query point position, in pixel units
            qx = (r_int.x + Lx*qr);
            qy = (r_int.y + Ly*qr);

            // Fetch the (a1,a2) values from the texture
            avector = tex2D(texRef, qx, qy);

            // Convert to individual 32bit floats and add to the tallies
            a1 = a1 + avector.x;
            a2 = a2 + avector.y;
        }
        else {
            end_of_line = true;
        }
        // Increment the batch counter
        batchID = batchID + 1;
    }

    /*** PART DEUX: sum the fetched values across the whole thread block ***/
    // Based on this documentation: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/

    // The lane of each thread in the warp (0,31)
    int lane = threadIdx.x % warpSize;

    // The warp ID in the thread block (0,31)
    int wid = threadIdx.x / warpSize;

    // First, sum all lanes within each warp
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        a1 += __shfl_down_sync(FULL_MASK, a1, offset);
        a2 += __shfl_down_sync(FULL_MASK, a2, offset);
    }

    // The first thread in the warp holds the reduced value
    // and writes it to shared memory
    if (lane==0) {
        shared1[wid] = a1;
        shared2[wid] = a2;
    }

    __syncthreads(); // Wait for all partial reductions

    // Secondly, sum across all warps
    A_vox2det.x = shared1[0];
    A_vox2det.y = shared2[0];
    for(wid = 1; wid<numWarps; wid++) {
        A_vox2det.x += shared1[wid];
        A_vox2det.y += shared2[wid];
	}
    
    // Multiply by the step size. All threads have access to this:
    A_vox2det.x = A_vox2det.x*DQ;
    A_vox2det.y = A_vox2det.y*DQ;


    return A_vox2det;
}
