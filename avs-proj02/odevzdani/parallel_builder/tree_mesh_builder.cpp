/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Vojtech Bubela <xbubel08stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    18.12.2022
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.
    unsigned totalTriangles = 0;

    // create dummy vector for fisr iteration so it wont pass surface_in_cube
    const Vec3_t<float> dummyVect;

    // start the paralellization
    // everything can be shared, since none of the threads will try to rewrite the values
    // only read them
    // since this first call of octreeDecompose will generate all the tasks recursively it can be
    // called only one time, also there is no need to wait on a lock because when program pointer
    // returns there the calculations will be all done
    #pragma omp parallel default(none) shared(field, totalTriangles, dummyVect)
    #pragma omp single nowait
    totalTriangles = octreeDecompose(field, mGridSize, dummyVect);
    
    
    return totalTriangles;
}

unsigned TreeMeshBuilder::octreeDecompose(const ParametricScalarField &field, unsigned currentGridSize, const Vec3_t<float> &cubeOffset)
{
    unsigned trianglesCount = 0;

    // convert gridside size to float
    float gridSideF = (float) currentGridSize;

    
    // check if surface lies in this cube
    if(surfaceNotInCube(gridSideF, cubeOffset, field))
    {
        // there will be no triangles here
        return 0;
    }

    // check if lowet level was reached
    if(currentGridSize <= GRID_SIDE_MIN_LEN)
    {
        return buildCube(cubeOffset, field);
    }


    // recursively call octreeDecompose
    // gridside size will now be halved
    const unsigned newGridSize = currentGridSize / 2;
    float halfSideSize = (float)newGridSize;

    for (int i = 0; i < 8; ++i) {

        // split cube into 8 new cubes
        #pragma omp task default(none) shared(field, cubeOffset, newGridSize, halfSideSize, trianglesCount) \
        firstprivate(i)
        {
            // calculate new cube coordinates
            const Vec3_t<float> newCube(
                cubeOffset.x + halfSideSize * sc_vertexNormPos[i].x,
                cubeOffset.y + halfSideSize * sc_vertexNormPos[i].y,
                cubeOffset.z + halfSideSize * sc_vertexNormPos[i].z
            );

            unsigned Triangles = octreeDecompose(field, newGridSize, newCube);

            // only one thread can write value at the same time
            #pragma omp atomic
            trianglesCount += Triangles;
        }
    }

    // wait until all task are finished to return the cummulated count of triangles
    #pragma omp taskwait
    return trianglesCount;
}

// use evaluate field to find out if the surface of bunny goes through actuall cube
bool TreeMeshBuilder::surfaceNotInCube(float gridSideF, const Vec3_t<float> cubeOffset, const ParametricScalarField &field)
{

    // this expression will always be the same so it can be static
    static float partial_exp = sqrt(3.0f) / 2.0f;

    // evaluateFieldAt need the point in "continuous space"
    // get half of lenght of current cube
    float maxSideSize = gridSideF * mGridResolution;
    float halfSideSize = maxSideSize / 2.0f;

    // get the intex in the middle of cube side
    const Vec3_t<float> compareVec(
        cubeOffset.x * mGridResolution + halfSideSize,
        cubeOffset.y * mGridResolution + halfSideSize,
        cubeOffset.z * mGridResolution + halfSideSize
    );

    // return bool
    return evaluateFieldAt(compareVec, field) > mIsoLevel + partial_exp * maxSideSize;
}


// taken from ref_mesh_builder.cpp
float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    // make sure only one thread pushes triangles at the same time
    #pragma omp critical
    mTriangles.push_back(triangle);
}
