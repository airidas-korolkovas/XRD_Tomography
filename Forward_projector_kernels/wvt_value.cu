// Fast X-ray diffraction (XRD) tomography for enhanced identification of materials
// by Airidas Korolkovas, Scientific Reports volume 12, Article number: 19097 (2022) 
// https://doi.org/10.1038/s41598-022-23396-2
// airidas.korolkovas89@gmail.com
// Last modified: 11/19/2022

/*** Writes the WVT values given the thread index nwt ***/
__device__ void wvt_value(int nwt, float* qnow, float* dqnow)
{
    // Mean q-value of the current bin
    float qedge1 = qmin + nwt*dq0 + 
            (qmax-qmin-dq0*__int2float_rn(NWT))*
            __int2float_rn(nwt*nwt)/__int2float_rn(NWT*NWT);

    float qedge2 = qmin + __int2float_rn(nwt+1)*dq0 + 
            (qmax-qmin-dq0*__int2float_rn(NWT))*
            __int2float_rn((nwt+1)*(nwt+1))/__int2float_rn(NWT*NWT);

    // store the output in the addresses given by the pointer variables
    *qnow = 0.5f*(qedge1+qedge2);
    *dqnow = qedge2 - qedge1;

}