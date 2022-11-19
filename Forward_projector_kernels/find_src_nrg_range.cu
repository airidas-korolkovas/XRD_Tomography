// Fast X-ray diffraction (XRD) tomography for enhanced identification of materials
// by Airidas Korolkovas, Scientific Reports volume 12, Article number: 19097 (2022) 
// https://doi.org/10.1038/s41598-022-23396-2
// airidas.korolkovas89@gmail.com
// Last modified: 11/19/2022

/*** Finds the min and max indices of the pre-smeared source spectrum ***/
__device__ void find_src_nrg_range(int nrg, int* nrg_src_min, int* nrg_src_max)
{
               
    int nrg_min = nrg-DNRG;
    if (nrg_min<0) { nrg_min = 0; }

    int nrg_max = nrg_min + 2*DNRG;
    if (nrg_max>=NRG) { 
        nrg_max = NRG-1;
        nrg_min = nrg_max - 2*DNRG;
    }

    
    // store the output in the addresses given by the pointer variables
    *nrg_src_min = nrg_min;
    *nrg_src_max = nrg_max;

}
