import sys, time, os
import numpy as np
#import pylab as plt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
#import pycuda.curandom as curandom
from pycuda.reduction import ReductionKernel
import h5py as h5

#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
volumeRenderDirectory = parentDirectory + "/volumeRender"
dataDir = "/home/bruno/Desktop/data/qTurbulence/"
sys.path.extend( [toolsDirectory, volumeRenderDirectory] )


from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray, kernelMemoryInfo
from tools import ensureDirectory, printProgressTime


cudaP = "float"
nPoints = 128
useDevice = None
usingAnimation = False
plotVelocity = False
showKernelMemInfo = False

for option in sys.argv:
  if option == "float": cudaP = "float"
  if option == "anim": usingAnimation = True
  if option == "mem": showKernelMemInfo = True
  if option == "128" or option == "256": nPoints = int(option)
  if option.find("dev=") != -1: useDevice = int(option[-1]) 
  if option == "vel": plotVelocity = True
precision  = {"float":(np.float32, np.complex64), "double":(np.float64,np.complex128) } 
cudaPre, cudaPreComplex = precision[cudaP]

#set simulation volume dimentions 
nWidth = nPoints
nHeight = nPoints
nDepth = nPoints
nData = nWidth*nHeight*nDepth

#Simulation Parameters
dtReal = 0.0025
endTime = 10
nSnapshots = 10
timeDirection = 'fwd'
nPartialIter = int(endTime/dtReal)/nSnapshots + 1
simulationTime = 0


Lx = 30.0
Ly = 30.0
Lz = 30.0
xMax, xMin = Lx/2, -Lx/2
yMax, yMin = Ly/2, -Ly/2
zMax, zMin = Lz/2, -Lz/2
dx, dy, dz = Lx/(nWidth-1), Ly/(nHeight-1), Lz/(nDepth-1 )
Z, Y, X = np.mgrid[ zMin:zMax:nDepth*1j, yMin:yMax:nHeight*1j, xMin:xMax:nWidth*1j ]
gammaX = 1
gammaY = 1
gammaZ = 1


#Load inital state
inFileName = dataDir + 'initial/interpol/psi_{0}.h5'.format( 256 )
inFile = h5.File( inFileName, 'r')
psi_h = inFile['psi'][...].astype( cudaPreComplex )
inFile.close()
if nPoints == 128: psi_h = psi_h[::2,::2,::2].copy()


nIteratiosPerFrame = 50
#Change precision of the parameters
dx, dy, dz = cudaPre(dx), cudaPre(dy), cudaPre(dz)
Lx, Ly, Lz = cudaPre(Lx), cudaPre(Ly), cudaPre(Lz)
xMin, yMin, zMin = cudaPre(xMin), cudaPre(yMin), cudaPre(zMin)
dtReal = cudaPre(dtReal)
gammaX, gammaY, gammaZ = cudaPre(gammaX), cudaPre(gammaY), cudaPre(gammaZ), 
#Initialize openGL
if usingAnimation:
  import volumeRender
  volumeRender.nWidth = nWidth
  volumeRender.nHeight = nHeight
  volumeRender.nDepth = nDepth
  volumeRender.windowTitle = "Quantum Turbulence  nPoints={0}".format(nPoints)
  volumeRender.nTextures = 1 + 1*plotVelocity
  #volumeRender.viewXmin, volumeRender.viewXmax = -1., 1.
  #volumeRender.viewYmin, volumeRender.viewYmax = -1., 1.
  #volumeRender.viewZmin, volumeRender.viewZmax = -1., 1.
  volumeRender.initGL()
  
#initialize pyCUDA context 
cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=usingAnimation)

#set thread grid for CUDA kernels
block_size_x, block_size_y, block_size_z = 8,8,4   #hardcoded, tune to your needs
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
gridz = nDepth // block_size_z + 1 * ( nDepth % block_size_z != 0 )
block3D = (block_size_x, block_size_y, block_size_z)
grid3D = (gridx, gridy, gridz)
nBlocks3D = grid3D[0]*grid3D[1]*grid3D[2]

print "\nCompiling CUDA code"
cudaCodeFile = open("cudaTurbulence.cu","r")
cudaCodeString_raw = cudaCodeFile.read().replace( "cudaP", cudaP ) 
cudaCodeString = cudaCodeString_raw % { 
  "THREADS_PER_BLOCK":block3D[0]*block3D[1]*block3D[2], 
  "B_WIDTH":block3D[0], "B_HEIGHT":block3D[1], "B_DEPTH":block3D[2],
  'blockDim.x': block3D[0], 'blockDim.y': block3D[1], 'blockDim.z': block3D[2],
  'gridDim.x': grid3D[0], 'gridDim.y': grid3D[1], 'gridDim.z': grid3D[2] }
cudaCode = SourceModule(cudaCodeString)
#TEXTURE version
initVortex_kernel = cudaCode.get_function( "initVortex" )
eulerStep_texKernel = cudaCode.get_function( "eulerStep_texture_kernel" )
getVelocity_texKernel = cudaCode.get_function( "getVelocity_tex_kernel" )
tex_psiReal = cudaCode.get_texref("tex_psiReal")
tex_psiImag = cudaCode.get_texref("tex_psiImag")
surf_psiReal = cudaCode.get_surfref("surf_psiReal")
surf_psiImag = cudaCode.get_surfref("surf_psiImag")
if showKernelMemInfo: 
  kernelMemoryInfo(eulerStep_texKernel, 'eulerStepKernel_texture')
  print ""
########################################################################
from pycuda.elementwise import ElementwiseKernel
########################################################################
multiplyByScalarReal = ElementwiseKernel(arguments="cudaP a, cudaP *realArray".replace("cudaP", cudaP),
				operation = "realArray[i] = a*realArray[i] ",
				name = "multiplyByScalarReal_kernel")
########################################################################
multiplyByScalarComplex = ElementwiseKernel(arguments="cudaP a, pycuda::complex<cudaP> *psi".replace("cudaP", cudaP),
				operation = "psi[i] = a*psi[i] ",
				name = "multiplyByScalarComplex_kernel",
				preamble="#include <pycuda-complex.hpp>")
########################################################################
getModulo = ElementwiseKernel(arguments="pycuda::complex<cudaP> *psi, cudaP *psiMod".replace("cudaP", cudaP),
			      operation = "cudaP mod = abs(psi[i]);\
					    psiMod[i] = mod*mod;".replace("cudaP", cudaP),	
			      name = "getModulo_kernel",
			      preamble="#include <pycuda-complex.hpp>")
########################################################################
sendModuloToUCHAR = ElementwiseKernel(arguments="cudaP *psiMod, unsigned char *psiUCHAR".replace("cudaP", cudaP),
			      operation = "psiUCHAR[i] = (unsigned char) ( -255*(psiMod[i]-1));",
			      name = "sendModuloToUCHAR_kernel")
########################################################################
getNorm = ReductionKernel( np.dtype(cudaPre),
			    neutral = "0",
			    arguments=" cudaP dx, cudaP dy, cudaP dz, pycuda::complex<cudaP> * psi ".replace("cudaP", cudaP),
			    map_expr = "( conj(psi[i])* psi[i] )._M_re*dx*dy*dz",
			    reduce_expr = "a+b",
			    name = "getNorm_kernel",
			    preamble="#include <pycuda-complex.hpp>")
########################################################################
def gaussian3D(x, y, z, gammaX=1, gammaY=1, gammaZ=1, random=False):    
  values =  np.exp( -gammaX*x*x - gammaY*y*y - gammaZ*z*z ).astype( cudaPre )
  if random:
    values += ( 100*np.random.random(values.shape) - 50 ) * values
  return values
########################################################################
def normalize( dx, dy, dz, complexArray ):
  factor = cudaPre( 1./(np.sqrt(getNorm(  dx, dy, dz, complexArray ).get())) )  #OPTIMIZATION
  multiplyByScalarComplex( factor, complexArray )
########################################################################
########################################################################
def rk4_texture_iteration():
  #Step 1
  slopeCoef = cudaPre( 1.0 )
  weight    = cudaPre( 0.5 )
  tex_psiReal.set_array( psiK2Real_array )
  tex_psiImag.set_array( psiK2Imag_array )
  surf_psiReal.set_array( psiK1Real_array )
  surf_psiImag.set_array( psiK1Imag_array )
  eulerStep_texKernel( slopeCoef, weight,
		  xMin, yMin, zMin, dx, dy, dz, dtReal, gammaX, gammaY, gammaZ,
		  psi_d, psiRunge_d, np.uint8(0), grid=grid3D, block=block3D )
  #Step 2
  slopeCoef = cudaPre( 2.0 )
  weight    = cudaPre( 0.5 )
  tex_psiReal.set_array( psiK1Real_array )
  tex_psiImag.set_array( psiK1Imag_array )
  surf_psiReal.set_array( psiK2Real_array )
  surf_psiImag.set_array( psiK2Imag_array )
  eulerStep_texKernel(  slopeCoef, weight,
		  xMin, yMin, zMin, dx, dy, dz, dtReal, gammaX, gammaY, gammaZ,
		  psi_d, psiRunge_d, np.uint8(0),  grid=grid3D, block=block3D )  
  #Step 3
  slopeCoef = cudaPre( 2.0 )
  weight    = cudaPre( 1. )
  tex_psiReal.set_array( psiK2Real_array )
  tex_psiImag.set_array( psiK2Imag_array )
  surf_psiReal.set_array( psiK1Real_array )
  surf_psiImag.set_array( psiK1Imag_array )
  eulerStep_texKernel(  slopeCoef, weight,
		  xMin, yMin, zMin, dx, dy, dz, dtReal, gammaX, gammaY, gammaZ,
		  psi_d, psiRunge_d, np.uint8(0),  grid=grid3D, block=block3D )    
  #Step 4
  slopeCoef = cudaPre( 1.0 )
  weight    = cudaPre( 1. )
  tex_psiReal.set_array( psiK1Real_array )
  tex_psiImag.set_array( psiK1Imag_array )
  surf_psiReal.set_array( psiK2Real_array )
  surf_psiImag.set_array( psiK2Imag_array )
  eulerStep_texKernel(  slopeCoef, weight,
		  xMin, yMin, zMin, dx, dy, dz, dtReal, gammaX, gammaY, gammaZ,
		  psi_d, psiRunge_d, np.uint8(1),grid=grid3D, block=block3D ) 
########################################################################
def realStep():
  global simulationTime
  [rk4_texture_iteration() for i in range(nIteratiosPerFrame)]
  simulationTime += dtReal*nIteratiosPerFrame
  #print simulationTime
  ########################################################################
def stepFuntion():
  getModulo( psi_d, psiMod_d )
  maxVal = (gpuarray.max(psiMod_d)).get()
  multiplyByScalarReal( cudaPre(0.95/(maxVal)), psiMod_d )
  sendModuloToUCHAR( psiMod_d, plotData_d)
  copyToScreenArray()

  #if volumeRender.nTextures == 2:
    #if not realDynamics:
      #cuda.memset_d8(activity_d.ptr, 0, nBlocks3D )
      #findActivityKernel( cudaPre(0.001), psi_d, activity_d, grid=grid3D, block=block3D )
    #if plotVar == 1: getActivityKernel( psiOther_d, activity_d, grid=grid3D, block=block3D )
    #if plotVar == 0:
      #if realTEXTURE:
	#tex_psiReal.set_array( psiK2Real_array )
	#tex_psiImag.set_array( psiK2Imag_array )
	#getVelocity_texKernel( dx, dy, dz, psi_d, activity_d, psiOther_d, grid=grid3D, block=block3D )
      #else: getVelocityKernel( np.int32(neighbors), dx, dy, dz, psi_d, activity_d, psiOther_d, grid=grid3D, block=block3D )
      #maxVal = (gpuarray.max(psiOther_d)).get()
      #if maxVal > 0: multiplyByScalarReal( cudaPre(1./maxVal), psiOther_d )
    #sendModuloToUCHAR( psiOther_d, plotData_d_1)
    #copyToScreenArray_1()  
  realStep()
#######################################################################   
def saveSnapshot( n, outFile, time, psi ):
  key = 'snap_{0:03d}'.format(n)
  snap = outFile.create_group( key )
  snap.attrs['t'] = time
  snap.create_dataset( "psi", data=psi, compression='lzf')
#######################################################################   
print "\nInitializing Data"  
initialMemory = getFreeMemory( show=True ) 
psi_d = gpuarray.to_gpu(psi_h)
initVortex_kernel( xMin, yMin, zMin, dx, dy, dz, psi_d, grid=grid3D, block=block3D )
normalize( dx, dy, dz, psi_d )
psi_h = psi_d.get()
psiMod_d = gpuarray.to_gpu(  np.zeros_like(psi_h.real) )
k1tempReal = gpuarray.to_gpu(  psi_h.real.copy() )
k1tempImag = gpuarray.to_gpu(  psi_h.imag.copy() )
psiRunge_d = gpuarray.to_gpu( psi_h )
psiK1Real_array, copy3DpsiK1Real = gpuArray3DtocudaArray( k1tempReal, allowSurfaceBind=True, precision=cudaP )
psiK1Imag_array, copy3DpsiK1Imag = gpuArray3DtocudaArray( k1tempImag, allowSurfaceBind=True, precision=cudaP )
psiK2Real_array, copy3DpsiK2Real = gpuArray3DtocudaArray( k1tempReal, allowSurfaceBind=True, precision=cudaP )
psiK2Imag_array, copy3DpsiK2Imag = gpuArray3DtocudaArray( k1tempImag, allowSurfaceBind=True, precision=cudaP ) 
del k1tempImag, k1tempReal
#memory for plotting
if usingAnimation:
  plotData_d = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
  volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( plotData_d )
  if volumeRender.nTextures == 2:
    plotData_d_1 = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
    volumeRender.plotData_dArray_1, copyToScreenArray_1 = gpuArray3DtocudaArray( plotData_d_1 )
print "Total Global Memory Used: {0:.2f} MB\n".format(float(initialMemory-getFreeMemory( show=False ))/1e6) 
######################################################################################
######################################################################################


#configure volumeRender functions  
if usingAnimation: 
  volumeRender.viewTranslation[2] = -2
  #volumeRender.keyboard = keyboard
  #volumeRender.specialKeys = specialKeyboardFunc
  volumeRender.stepFunc = stepFuntion
  #run volumeRender animation
  volumeRender.animate()

#Open outFile
snapsDir = dataDir + 'snapshots/'
ensureDirectory( snapsDir )
n = len( [ f for f in os.listdir( snapsDir ) if f.find(timeDirection)>0 ] )
snapsFileName = snapsDir + 'psi_{0}_{1}.h5'.format( timeDirection, n )
snapsFile = h5.File( snapsFileName, 'w')
  
print "\nnPoints: {0}x{1}x{2}".format(nWidth, nHeight, nWidth )
print 'Snapshots File: ', snapsFileName
print "Starting Real Dynamics: {0} timeUnits, dt: {1:1.2e} \n".format( endTime, dtReal )  

printCounter = 0
start, end = cuda.Event(), cuda.Event()
start.record()
for n in range( nSnapshots ):
  end.record().synchronize()
  printProgressTime( n, nSnapshots,  start.time_till(end)*1e-3 )
  saveSnapshot( n, snapsFile, simulationTime, psi_d.get() )
  [rk4_texture_iteration() for i in range(nPartialIter)]
  simulationTime += dtReal*nPartialIter
printProgressTime( nSnapshots, nSnapshots,  start.time_till(end)*1e-3 )
saveSnapshot( nSnapshots, snapsFile, simulationTime, psi_d.get() )
end.record().synchronize()


print "\n\nEnd Real Dynamics "  
print 'Time: {0:.0f}\n'.format( start.time_till(end)*1e-3 )

