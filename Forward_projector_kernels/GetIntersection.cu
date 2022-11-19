// Fast X-ray diffraction (XRD) tomography for enhanced identification of materials
// by Airidas Korolkovas, Scientific Reports volume 12, Article number: 19097 (2022) 
// https://doi.org/10.1038/s41598-022-23396-2
// airidas.korolkovas89@gmail.com
// Last modified: 11/19/2022


/*** Compute the intersection location between a line (x1,y1)->(x2,y2), and the perimeter rectangle ***/
__device__ float2 GetIntersection(float x1, float y1, float x2, float y2, int const NX, int const NY)
{
    // Define the four sides of the rectangle, as line segments (x3,y3)->(x4,y4)
    float x3[4] = {0.f, __int2float_rn(NX), __int2float_rn(NX), 0.f};
    float x4[4] = {__int2float_rn(NX), __int2float_rn(NX), 0.f, 0.f};
    float y3[4] = {0.f, 0.f, __int2float_rn(NY), __int2float_rn(NY)};
    float y4[4] = {0.f, __int2float_rn(NY), __int2float_rn(NY), 0.f};

    
    float x2_x1 = x2 - x1;
    float y2_y1 = y2 - y1;
    float eps = 0.00001f;
    float x4_x3, y3_y1, y4_y3, x3_x1, denominator, u_a, u_b;

    // Start assuming that the intersection position is NaN
    float2 r_int;
    r_int.x = __int_as_float(0x7fffffff);
    r_int.y = __int_as_float(0x7fffffff);
    
    #pragma unroll
    for(int i=0; i<4; i++) {
            
        // The 1D differences between the segment endpoints
        x4_x3 = x4[i] - x3[i];
        y3_y1 = y3[i] - y1;
        y4_y3 = y4[i] - y3[i];
        x3_x1 = x3[i] - x1;        
        denominator = x4_x3*y2_y1 - x2_x1*y4_y3;

        // Check that the lines are not parallel
        if (denominator > eps || denominator < -eps) {
            u_a = (x4_x3*y3_y1 - x3_x1*y4_y3)/denominator;

            // Check that the intersection is within the 1st segment range
            if ( (u_a > -eps) && (u_a < (1.f+eps)) ) {
                u_b = (x2_x1*y3_y1 - x3_x1*y2_y1)/denominator;

                // Check that the intersection is within the 2nd segment range
                if ( (u_b > -eps) && (u_b < (1.f+eps)) ) {
                    r_int.x = x1 + x2_x1*u_a;
                    r_int.y = y1 + y2_y1*u_a;
                    break;
                }
            }
        }
    }

    return r_int;
}