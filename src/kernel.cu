#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime.h>
#include "utilityCore.hpp"
#include "kernel.h"

#define DEBUGOUT 0

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?


int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

int numSteps = 0;

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;

__constant__ int dgridCellCount = -1;
__constant__ int dgridSideCount = -1;
__constant__ float dgridCellWidth = -1.0f;
__constant__ float dgridInverseCellWidth = -1.0f;

glm::vec3 gridMinimum;

/********************************
* Forward Function declarations
*********************************/

/**
* Given a location in our world, gives us back which grid cell contains that point
* Additionally, takes in an optional pointer to a spot into which to throw
* the octant of our point within that grid cube
* If passed NULL, the function will not determine the octant.
*/
__device__ int locToGridIndex(glm::vec3* pos, int* octant);

/**
* Converts a linear grid index to a trio of grid coordinates
*/
__device__ glm::ivec3 gridIndexToGridCoords(int gridIndex);
/**
* Converts a trio of grid coordinates to a linear grid index
*/
__device__ int gridCoordsToGridIndex(glm::ivec3 gridCoords);
/**
* These fill in an array `results` with the indexes of cells adjacent to the provided one
* This list does include the provided cell as well
* Notably, this requires the octant of the "center" as well
* This NEEDS to be passed an int array with space for 8 result numbers
*/
__device__ void getNeighboringCellIndexes(int gridIndex, int octant, int* results);
__device__ void getNeighboringCellIndexesTrio(glm::ivec3 gridCoords, int octant, int* results);

/**
* Function for finding the beginning/end of cellNumber in the sorted list gridIndices
* Returns a 2d vector with the (indexBeginning, indexEnd) values (the latter exclusive), or (-1, -1) if not found
* So, given a `cellNumber`, the array `gridIndices` contains that number from `indexBeginning` through `indexEnd -1`
*/
__device__ glm::ivec2 findMyGridIndices(int cellNumber, int* gridIndices, int N);

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

bool testDevPosTransfer() {
	glm::vec3 tempPos[100];
	cudaMemcpy(tempPos, dev_pos, 100 * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("\nDevPosTransfer did not work here!\n");
	return true;
}//testDevPosTransfer

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  //Copy some of these grid constants to device memory, to abuse later
  cudaMemcpyToSymbol(dgridCellCount, &gridCellCount, sizeof(int));
  cudaMemcpyToSymbol(dgridSideCount, &gridSideCount, sizeof(int));
  cudaMemcpyToSymbol(dgridCellWidth, &gridCellWidth, sizeof(float));
  cudaMemcpyToSymbol(dgridInverseCellWidth, &gridInverseCellWidth, sizeof(float));

  printf("halfSideCount %d, gridSideCount %d, gridCellCount %d, gridInverseCellWidth %0.1f, halfGridWidth %0.1f, gridmin (x, y, z) (%.1f, %.1f, %.1f)\n",
	  halfSideCount, gridSideCount, gridCellCount, gridInverseCellWidth, halfGridWidth, gridMinimum.x, gridMinimum.y, gridMinimum.z);

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.

  cudaMalloc((void**)& dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");
  cudaMalloc((void**)& dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");
  cudaMalloc((void**)& dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");
  cudaMalloc((void**)& dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");


  testDevPosTransfer();
  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO <<< fullBlocksPerGrid, blockSize >>>(numObjects, dev_pos, vbodptr_positions, scene_scale);
  checkCUDAErrorWithLine("copyBoidsToVBO failed on Pos!");
  kernCopyVelocitiesToVBO <<< fullBlocksPerGrid, blockSize >>>(numObjects, dev_vel1, vbodptr_velocities, scene_scale);
  checkCUDAErrorWithLine("copyBoidsToVBO failed on Vel!");

  cudaDeviceSynchronize();
}

/******************
* math helpers    *
******************/

/**
* Helper function to compute distance between two boids
* Added as the first steps of "can I modify this program sensibly"
*/
__device__ float computeDistance(const glm::vec3* pos1, const glm::vec3* pos2){
	double result = sqrt((pos2->x - pos1->x) * (pos2->x - pos1->x) 
			+ (pos2->y - pos1->y) * (pos2->y - pos1->y)
			+ (pos2->z - pos1->z) * (pos2->z - pos1->z));
	//double result = glm::distance(*pos1, *pos2);
	return result;

}//kernComputeDistance

/**
* Clamps the speed down to maxSpeed
*
* Does so in-place (DOES modify data)
*/
__device__ void clampSpeed(glm::vec3* vel){
    glm::vec3 zeroPoint = glm::vec3(0.0f, 0.0f, 0.0f);
    double curSpeed = computeDistance(&zeroPoint, vel);
    if (curSpeed > maxSpeed){
        double scaleFactor = maxSpeed / curSpeed; 
        *vel *= scaleFactor;
    }//if

}//clampSpeed

/********************
* grid math helpers *
*********************/

__device__ int locToGridIndex(glm::vec3* pos, int* octant) {
	//positions range between -scene_scale and scene_scale
	float gridInverseCellWidth = dgridInverseCellWidth;
	float xGridPos = pos->x * gridInverseCellWidth;//range between, for our example, -10 and 10
	float yGridPos = pos->y * gridInverseCellWidth;
	float zGridPos = pos->z * gridInverseCellWidth;

	glm::ivec3 gridTrio = glm::ivec3((int)(glm::floor(xGridPos)), (int)(glm::floor(yGridPos)), (int)(glm::floor(zGridPos)));
	int retval = gridCoordsToGridIndex(gridTrio);

	if (octant != NULL) {
		uint8_t topx = (((int)xGridPos) == ((int)(xGridPos - 0.5)));
		uint8_t topy = (((int)yGridPos) == ((int)(yGridPos - 0.5)));
		uint8_t topz = (((int)zGridPos) == ((int)(zGridPos - 0.5)));

		*octant = (int)(topz << 2 | topy < 1 | topx);
	}//if
	return retval;
}//locToGridIndex

__device__ glm::ivec3 gridIndexToGridCoords(int gridIndex) {
	int gridSideCount = dgridSideCount;
	int coordz = gridIndex / (gridSideCount * gridSideCount);
	int coordy = (gridIndex / gridSideCount) % gridSideCount;
	int coordx = gridIndex % gridSideCount;

	glm::ivec3 retval = glm::ivec3(coordx, coordy, coordz);
	retval -= (gridSideCount / 2);

	return retval;

}//gridIndexToGridCoords

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridCoordsToGridIndex(glm::ivec3 gridCoords) {
	int gridSideCount = dgridSideCount;
	int halfSideCount = gridSideCount / 2;
	int coordx = gridCoords.x + halfSideCount;
	int coordy = gridCoords.y + halfSideCount;
	int coordz = gridCoords.z + halfSideCount;
	int retval = coordx + (coordy * gridSideCount) + (coordz * gridSideCount * gridSideCount);
	return retval;
}//gridCoordsToGridIndex

__device__ void getNeighboringCellIndexes(int gridIndex, int octant, int* results) {
	glm::ivec3 gridCoord = gridIndexToGridCoords(gridIndex);
	getNeighboringCellIndexesTrio(gridCoord, octant, results);
}//getNeighboringCellIndexes

__device__ void getNeighboringCellIndexesTrio(glm::ivec3 gridCoords, int octant, int* results) {
	results[0] = gridCoordsToGridIndex(gridCoords);//my index
	//bool xtop = (octant & 0x01) > 0;
	//bool ytop = (octant & 0x02) > 0;
	//bool ztop = (octant & 0x04) > 0;
	uint8_t xtop = (uint8_t)(octant & 0x01);
	uint8_t ytop = (uint8_t)(octant & 0x02);
	uint8_t ztop = (uint8_t)(octant & 0x04);
	int x = gridCoords.x;
	int y = gridCoords.y;
	int z = gridCoords.z;

	results[1]    = gridCoordsToGridIndex(glm::ivec3(xtop ? x + 1 : x - 1, 
													y, 
													z));
	results[2]    = gridCoordsToGridIndex(glm::ivec3(x,
													ytop ? y + 1 : y - 1,
													z));
	results[3]    = gridCoordsToGridIndex(glm::ivec3(x,
													y,
													ztop ? z + 1 : z - 1));
	results[4]    = gridCoordsToGridIndex(glm::ivec3(xtop ? x + 1 : x - 1,
													ytop ? y + 1 : y - 1,
													z));
	results[5]    = gridCoordsToGridIndex(glm::ivec3(xtop ? x + 1 : x - 1,
													y,
													ztop ? z + 1 : z - 1));
	results[6]    = gridCoordsToGridIndex(glm::ivec3(x,
													ytop ? y + 1 : y - 1,
													ztop ? z + 1 : z - 1));
	results[7]    = gridCoordsToGridIndex(glm::ivec3(xtop ? x + 1 : x - 1,
													ytop ? y + 1 : y - 1,
													ztop ? z + 1 : z - 1));


}//getNeighboringCellIndexesTrio

__device__ glm::ivec2 findMyGridIndices(int cellNumber, int* gridIndices, int N){
	int i = 0;
	int beginIndex	= -1;
	int endIndex	= -1;
	while (i < N) {
		if (beginIndex == -1 && gridIndices[i] == cellNumber) {
			beginIndex = i;
		}//if
		else if (beginIndex != -1 && endIndex == -1 && gridIndices[i] != cellNumber) {
			endIndex = i;
			break;//for what good it does us, efficiency-wise, to break the loop early in the land of warps
		}//else
		else if (beginIndex != -1 && endIndex == -1 && i == N) {
			endIndex = N;
		}//I don't think it ever hits here?
		i++;
	}//while
	if (beginIndex > endIndex) {
		endIndex = N;//make sure we don't run off the end
	}//if

	return glm::ivec2(beginIndex, endIndex);

}//findMyGridIndices

/******************
* stepSimulation *
******************/

/**
* Given a self-boid and another single boid, returns the velocity contribution of the pair for rule 2
* Will call this method for multiple boids
*/
__device__ glm::vec3 computeRule2VelContributionSingle(const glm::vec3* myPos, const glm::vec3* theirPos,
                                                       const glm::vec3* myVel, const glm::vec3* theirVel){
    float distBetween = computeDistance(myPos, theirPos);
    if (distBetween > rule2Distance){
        return glm::vec3(0.0f, 0.0f, 0.0f);
    }//if

    glm::vec3 distVec = *myPos - *theirPos;

    return distVec; 

}//computeRule2VelContributionSingle

/**
* Given a self-boid and another single boid, returns the velocity contribution of the pair for rule 3
* Will call this method for multiple boids
*/
__device__ glm::vec3 computeRule3VelContributionSingle(const glm::vec3* myPos, const glm::vec3* theirPos,
                                                       const glm::vec3* myVel, const glm::vec3* theirVel){
    float distBetween = computeDistance(myPos, theirPos);
    if (distBetween > rule3Distance){
        return glm::vec3(0.0f, 0.0f, 0.0f);
    }//if

	return *theirVel;

}//computeRule3VelContributionSingle

/**
* Calculates the velocity contribution from rule 1 for all boids
*/
__device__ glm::vec3 computeRule1VelContribution(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel){

    glm::vec3 perceivedCenter = glm::vec3(0.0f, 0.0f, 0.0f);
    int numNeighbors = 0;

    glm::vec3 myPos = pos[iSelf];
    //glm::vec3 myVel = vel[iSelf];

    for (int i = 0; i < N; i++){
        if (i == iSelf) continue;
        if (computeDistance(&myPos, &pos[i]) < rule1Distance){
            numNeighbors++;
            perceivedCenter += pos[i];
        }//if a neighbor

    }//for each boid

    if (numNeighbors < 1) return glm::vec3(0.0f, 0.0f, 0.0f);

    perceivedCenter /= numNeighbors;
    glm::vec3 resultVector = perceivedCenter - myPos;
    resultVector *= rule1Scale;

    return resultVector;

}//computeRule1VelContribution

/**
* Calculates the velocity contribution from rule 2 for all boids
*/
__device__ glm::vec3 computeRule2VelContribution(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel){

    glm::vec3 velChange = glm::vec3(0.0f, 0.0f, 0.0f);

    glm::vec3 myPos = pos[iSelf];
    glm::vec3 myVel = vel[iSelf];

    for (int i = 0; i < N; i++){
        if (i == iSelf) continue;
        velChange += computeRule2VelContributionSingle(&myPos, &pos[i], &myVel, &vel[i]);
    }//for each boid

	return velChange * rule2Scale;

}//computeRule2VelContribution
    
/**
* Calculates the velocity contribution from rule 3 for all boids
*/
__device__ glm::vec3 computeRule3VelContribution(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel){

    glm::vec3 velChange = glm::vec3(0.0f, 0.0f, 0.0f);

    glm::vec3 myPos = pos[iSelf];
    glm::vec3 myVel = vel[iSelf];

    for (int i = 0; i < N; i++){
        if (i == iSelf) continue;
        velChange += computeRule3VelContributionSingle(&myPos, &pos[i], &myVel, &vel[i]);
    }//for each boid

	return velChange * rule3Scale;

}//computeRule3VelContribution
    
/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
    // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
    glm::vec3 rule1VelChange = computeRule1VelContribution(N, iSelf, pos, vel);
    // Rule 2: boids try to stay a distance d away from each other
    glm::vec3 rule2VelChange = computeRule2VelContribution(N, iSelf, pos, vel);
    // Rule 3: boids try to match the speed of surrounding boids
    glm::vec3 rule3VelChange = computeRule3VelContribution(N, iSelf, pos, vel);

    return rule1VelChange + rule2VelChange + rule3VelChange;

}//computeVelocityChange

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
    glm::vec3 *vel1, glm::vec3 *vel2) {

    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    // Compute a new velocity based on pos and vel1
    glm::vec3 velChange = computeVelocityChange(N, index, pos, vel1);

    // Clamp the speed
    glm::vec3 newVel = vel1[index] + velChange;
    clampSpeed(&newVel);

    // Record the new velocity into vel2. Question: why NOT vel1?
    vel2[index] = newVel;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {

	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}

	//dev_particleGridIndices
	gridIndices[index] = locToGridIndex(&pos[index], NULL);
	//dev_particleArrayIndices
	indices[index] = index;

    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
}

void Boids::sortGridIndices(int N) {
#if DEBUGOUT
	int* debugGridIndices = (int*)malloc(N * sizeof(int));
	int* debugArrayIndices = (int*)malloc(N * sizeof(int));
#endif


	// needed for use with thrust
	thrust::device_ptr<int> dev_thrust_particleArrayIndices(dev_particleArrayIndices);
	thrust::device_ptr<int> dev_thrust_particleGridIndices(dev_particleGridIndices);

	numSteps++;
	printf("On sort #%d\n", numSteps);

#if DEBUGOUT
	cudaMemcpy(debugGridIndices, dev_particleGridIndices, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(debugArrayIndices, dev_particleArrayIndices, N * sizeof(int), cudaMemcpyDeviceToHost);
#endif

	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + N, dev_thrust_particleArrayIndices);

}//sortGridIndices

/**
* Fills in the dev_gridCellStartIndices and dev_gridCellEndIndices arrays
* This is happening once for each cell, not once for each boid
*/
__global__ void kernFindGridStartEnds(int N, int* particleGridIndices, int* gridStarts, int* gridEnds) {
	// TODO-2.1
	// Identify the start point of each cell in the gridIndices array.
	// This is basically a parallel unrolling of a loop that goes
	// "this index doesn't match the one before it, must be a new cell!"
	int gridCellCount = dgridCellCount;
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= gridCellCount) return;

	glm::ivec2 startEnd = findMyGridIndices(index, particleGridIndices, N);
	gridStarts[index] = startEnd.x;
	gridEnds[index] = startEnd.y;

}//kernFindGridStartEnds

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

/* OLD HEADER
__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
*/

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}//if

	//find our grid cell and octant
	glm::vec3 myPos = pos[index];
	glm::vec3 myVel = vel1[index];
	int octant = -1;//will get overwritten
	int myGridIndex = locToGridIndex(&myPos, &octant);
	
	//get the grid indices for our neighbors
	int neighborGridIndices[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
	getNeighboringCellIndexes(myGridIndex, octant, neighborGridIndices);
	
	//Accumulators for velocity contributions
	glm::vec3 rule1VelChange = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 rule2VelChange = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 rule3VelChange = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 rule1Center	 = glm::vec3(0.0f, 0.0f, 0.0f);
	int rule1NumNeighbors	 = 0;

	//for each cell that could be close to us
	for (int i = 0; i < 8; i++) {
		int neighborCellIndex = neighborGridIndices[i];
		int cellStart = gridCellStartIndices[neighborCellIndex];
		int cellEnd = gridCellEndIndices[neighborCellIndex];
		//for each boid in that cell
		int j = cellStart;
		while (j != -1 && j != cellEnd) {
			int neighborBoidIndex = particleArrayIndices[j];
			if (neighborBoidIndex != index) {
				glm::vec3 theirPos = pos[neighborBoidIndex];
				glm::vec3 theirVel = vel1[neighborBoidIndex];
				//do the contribution math
				//Rule 1
				if (computeDistance(&myPos, &theirPos) < rule1Distance) {
					rule1NumNeighbors++;
					rule1Center += theirPos;
				}//if
				//Rule 2
				rule2VelChange += computeRule2VelContributionSingle(&myPos, &theirPos,
					&myVel, &theirVel);
				//Rule 3
				rule3VelChange += computeRule3VelContributionSingle(&myPos, &theirPos,
					&myVel, &theirVel);
				//increment j, to find other boids in that cell
			}//if

			j++;
		}//while
	}//for
	if (rule1NumNeighbors > 0) {
		rule1Center /= rule1NumNeighbors;
		rule1VelChange = rule1Center - myPos;
	}//if we have neighbors for rule 1


	//scale the velocity contributions
	rule1VelChange *= rule1Scale;
	rule2VelChange *= rule2Scale;
	rule3VelChange *= rule3Scale;

	//Add up vel contributions
	glm::vec3 newVel = myVel + rule1VelChange + rule2VelChange + rule3VelChange;
	clampSpeed(&newVel);

	vel2[index] = newVel;
	
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {

    int N = numObjects;
    dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

    kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(N, dev_pos, dev_vel1, dev_vel2);
    kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(N, dt, dev_pos, dev_vel2);
    
    cudaMemcpy(dev_vel1, dev_vel2, N * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
}

void Boids::stepSimulationScatteredGrid(float dt) {

	int N = numObjects;
	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);
	dim3 fullBlocksPerCellGrid((gridCellCount + blockSize - 1) / blockSize);

	kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (N, gridSideCount/*garbage*/, gridMinimum,
		gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
	checkCUDAErrorWithLine("Error on kernComputeIndices\n");

	cudaDeviceSynchronize();
	checkCUDAErrorWithLine("Error on sync after computing indices!");

	sortGridIndices(N);

	//TODO: look into making this just a serial operation, rather than parallelizing it inefficiently?
	kernFindGridStartEnds <<<fullBlocksPerCellGrid, blockSize >>>
		(N, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
	checkCUDAErrorWithLine("Error on kernFindGridStartEnds\n");


	kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> > (N, dev_gridCellStartIndices, dev_gridCellEndIndices,
		dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("Error on kernUpdateVelNeighborSearchScattered!\n");
	cudaDeviceSynchronize();
	checkCUDAErrorWithLine("Error on sync after updating vel's\n");

	kernUpdatePos <<<fullBlocksPerGrid, blockSize >>> (N, dt, dev_pos, dev_vel2);
	checkCUDAErrorWithLine("Error on kernUpdatePos!\n");

	cudaMemcpy(dev_vel1, dev_vel2, N * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	checkCUDAErrorWithLine("cudaMemCpy for ping-pong vel1 and vel2 failed!\n");
	cudaDeviceSynchronize();
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}

