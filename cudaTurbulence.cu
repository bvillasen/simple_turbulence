#include <pycuda-complex.hpp>
#include <pycuda-helpers.hpp>
#define pi 3.14159265f
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
typedef   pycuda::complex<cudaP> pyComplex;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void initVortex
	    (  cudaP xMin, cudaP yMin, cudaP zMin, 
	    cudaP dx, cudaP dy, cudaP dz, pyComplex *psi_d ){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y; 

  const pyComplex iComplex( 0, 1.0 );
  const cudaP x = t_j*dx + xMin;
  const cudaP y = t_i*dy + yMin;
//   const cudaP z = t_k*dz + zMin;
  
  cudaP radius, x0, y0, Q, vtxR, vtxDens ;
  pyComplex vtxPhase;
  radius = 0.0796842105263; 
  pyComplex psiVal = psi_d[ tid ];
  
  //Z vortex
  x0 = -1.3625;
  y0 = 0;
  Q  = 1;
  vtxR = sqrt( ( ( x - x0 )*( x - x0 ) + ( y - y0 )*( y - y0 ) )/radius );
  vtxDens = tanh( pow( vtxR,  2 * abs(Q) ) );
  vtxPhase =  exp( iComplex * Q * atan2( x-x0, y-y0 ) );
  psiVal *= ( vtxPhase * vtxDens );
  
  //Z vortex
  x0 = 1.3625;
  y0 = 0;
  Q  = -1;
  vtxR = sqrt( ( ( x - x0 )*( x - x0 ) + ( y - y0 )*( y - y0 ) )/radius );
  vtxDens = tanh( pow( vtxR,  2 * abs(Q) ) );
  vtxPhase = exp( iComplex * Q * atan2( x-x0, y-y0 ) );
  psiVal *= ( vtxPhase * vtxDens );  
  
  psi_d[ tid ] = psiVal;
  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_psiReal;
texture< fp_tex_cudaP, cudaTextureType3D, cudaReadModeElementType> tex_psiImag;
surface< void, cudaSurfaceType3D> surf_psiReal;
surface< void, cudaSurfaceType3D> surf_psiImag;
__device__ pyComplex vortexCore_tex
	    ( cudaP xMin, cudaP yMin, cudaP zMin, 
	      cudaP dx, cudaP dy, cudaP dz, int t_i, int t_j, int t_k, 
	      cudaP gammaX, cudaP gammaY, cudaP gammaZ ){
  

  const cudaP dxInv = 1.0f/dx;
//   cudaP dyInv = 1.0f/dy;
//   cudaP dzInv = 1.0f/dz;
  const cudaP x = t_j*dx + xMin;
  const cudaP y = t_i*dy + yMin;
  const cudaP z = t_k*dz + zMin;
  
  const pyComplex iComplex( 0, 1.0f );
  pyComplex center, psiPlus, psiMinus, laplacian;
  center._M_re =    fp_tex3D(tex_psiReal, t_j, t_i, t_k);
  psiPlus._M_re  =  fp_tex3D(tex_psiReal, t_j+1, t_i, t_k);
  psiMinus._M_re =  fp_tex3D(tex_psiReal, t_j-1, t_i, t_k);
  
  center._M_im =    fp_tex3D(tex_psiImag, t_j, t_i, t_k);
  psiPlus._M_im  =  fp_tex3D(tex_psiImag, t_j+1, t_i, t_k);
  psiMinus._M_im =  fp_tex3D(tex_psiImag, t_j-1, t_i, t_k);

  laplacian = (psiPlus + psiMinus - cudaP(2)*center )*dxInv*dxInv;
  
  psiPlus._M_re  =  fp_tex3D(tex_psiReal, t_j, t_i+1, t_k);
  psiMinus._M_re =  fp_tex3D(tex_psiReal, t_j, t_i-1, t_k);
  psiPlus._M_im  =  fp_tex3D(tex_psiImag, t_j, t_i+1, t_k);
  psiMinus._M_im =  fp_tex3D(tex_psiImag, t_j, t_i-1, t_k);
  laplacian += (psiPlus + psiMinus - cudaP(2)*center )*dxInv*dxInv;

  psiPlus._M_re =   fp_tex3D(tex_psiReal, t_j, t_i, t_k+1);
  psiMinus._M_re =  fp_tex3D(tex_psiReal, t_j, t_i, t_k-1);
  psiPlus._M_im =   fp_tex3D(tex_psiImag, t_j, t_i, t_k+1);
  psiMinus._M_im =  fp_tex3D(tex_psiImag, t_j, t_i, t_k-1);
  laplacian +=  (psiPlus + psiMinus - cudaP(2)*center )*dxInv*dxInv;

  const cudaP Vtrap_GP = 8000*norm(center) + (gammaX*x*x + gammaY*y*y + gammaZ*z*z)*cudaP(0.5); 

  return iComplex*(laplacian*cudaP(0.5) - (Vtrap_GP)*center );

}
__global__ void getVelocity_tex_kernel(  cudaP dx, cudaP dy, cudaP dz, unsigned char *activity, cudaP *psiOther){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  int tid_b = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
  int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
  
  //Border blocks are skiped
  if ( ( blockIdx.x == 0 or blockDim.x == gridDim.x -1 ) or ( blockIdx.y == 0 or blockDim.y == gridDim.y -1 ) or ( blockIdx.z == 0 or blockDim.z ==  %(gridDim.z)s  -1 ) ) return; 
 
  __shared__ unsigned char activeBlock;
  if (tid_b == 0 ) activeBlock = activity[bid];
  __syncthreads();
  if ( !activeBlock ) return; 
  
  cudaP dxInv = cudaP(1.0)/dx;
  cudaP dyInv = cudaP(1.0)/dy;
  cudaP dzInv = cudaP(1.0)/dz;
  pyComplex gradient_x, gradient_y, gradient_z, center, psiPlus, psiMinus;

  center._M_re =    fp_tex3D(tex_psiReal, t_j, t_i, t_k);
  center._M_im =    fp_tex3D(tex_psiImag, t_j, t_i, t_k);

  psiPlus._M_re  =  fp_tex3D(tex_psiReal, t_j+1, t_i, t_k);
  psiPlus._M_im  =  fp_tex3D(tex_psiImag, t_j+1, t_i, t_k);
  psiMinus._M_re =  fp_tex3D(tex_psiReal, t_j-1, t_i, t_k);
  psiMinus._M_im =  fp_tex3D(tex_psiImag, t_j-1, t_i, t_k);
  gradient_x = ( psiPlus - psiMinus)*dxInv*cudaP(0.5);
  
  psiPlus._M_re  =  fp_tex3D(tex_psiReal, t_j, t_i+1, t_k);
  psiPlus._M_im  =  fp_tex3D(tex_psiImag, t_j, t_i+1, t_k);
  psiMinus._M_re =  fp_tex3D(tex_psiReal, t_j, t_i-1, t_k);
  psiMinus._M_im =  fp_tex3D(tex_psiImag, t_j, t_i-1, t_k);
  gradient_y = ( psiPlus - psiMinus)*dyInv*cudaP(0.5);
  
  psiPlus._M_re =   fp_tex3D(tex_psiReal, t_j, t_i, t_k+1);
  psiPlus._M_im =   fp_tex3D(tex_psiImag, t_j, t_i, t_k+1);
  psiMinus._M_re =  fp_tex3D(tex_psiReal, t_j, t_i, t_k-1);
  psiMinus._M_im =  fp_tex3D(tex_psiImag, t_j, t_i, t_k-1);
  gradient_z = ( psiPlus - psiMinus)*dzInv*cudaP(0.5);

  cudaP rho = norm(center) + cudaP(5e-6);
  cudaP velX = (center._M_re*gradient_x._M_im - center._M_im*gradient_x._M_re)/rho;
  cudaP velY = (center._M_re*gradient_y._M_im - center._M_im*gradient_y._M_re)/rho;
  cudaP velZ = (center._M_re*gradient_z._M_im - center._M_im*gradient_z._M_re)/rho; 

  psiOther[tid] =  sqrt( velX*velX + velY*velY + velZ*velZ ) ;
}
////////////////////////////////////////////////////////////////////////////////
//////////////////////           EULER                //////////////////////////
////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void eulerStep_texture_kernel(  const cudaP slopeCoef, const cudaP weight, 
				  const cudaP xMin, const cudaP yMin, const cudaP zMin, 
				  const cudaP dx, const cudaP dy, const cudaP dz, const cudaP dt, 
				  const cudaP gammaX, const cudaP gammaY, const cudaP gammaZ, 
				      pyComplex *psi_d, pyComplex *psiRunge,
				      unsigned char lastRK4Step  ){
  const int t_j = blockIdx.x* %(blockDim.x)s  + threadIdx.x;
  const int t_i = blockIdx.y* %(blockDim.y)s  + threadIdx.y;
  const int t_k = blockIdx.z* %(blockDim.z)s  + threadIdx.z;
  const int tid = t_j + t_i* %(blockDim.x)s * %(gridDim.x)s  + t_k* %(blockDim.x)s * %(gridDim.x)s * %(blockDim.y)s * %(gridDim.y)s ;
 
  pyComplex value;
  value = vortexCore_tex( xMin, yMin, zMin, 
		      dx, dy, dz, t_i, t_j, t_k,
		      gammaX, gammaY, gammaZ  );
  value = dt*value;
  
  if (lastRK4Step ){
    value = psiRunge[tid] + slopeCoef*value/cudaP(6.); 
    psiRunge[tid] = value;
    psi_d[tid] = value;
    surf3Dwrite(  value._M_re, surf_psiReal,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
    surf3Dwrite(  value._M_im, surf_psiImag,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);    
  }  
  else{
    //add to rk4 final value
    psiRunge[tid] = psiRunge[tid] + slopeCoef*value/cudaP(6.);
    value = psi_d[tid] + weight*value;
    surf3Dwrite(  value._M_re, surf_psiReal,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
    surf3Dwrite(  value._M_im, surf_psiImag,  t_j*sizeof(cudaP), t_i, t_k,  cudaBoundaryModeClamp);
   
    
  }
}