/**
 * @file    tree_mesh_builder.h
 *
 * @author  Vojtech Bubela <xbubel08stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    18.12.2022
 **/

#ifndef TREE_MESH_BUILDER_H
#define TREE_MESH_BUILDER_H

#include "base_mesh_builder.h"

class TreeMeshBuilder : public BaseMeshBuilder
{
public:
    TreeMeshBuilder(unsigned gridEdgeSize);

protected:
static const unsigned GRID_SIDE_MIN_LEN = 1;
    unsigned marchCubes(const ParametricScalarField &field);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
    void emitTriangle(const Triangle_t &triangle);
    unsigned octreeDecompose(const ParametricScalarField &field, unsigned currentGridSize, const Vec3_t<float> &cubeOffset);
    bool surfaceNotInCube(float gridSideF, const Vec3_t<float> cubeOffset, const ParametricScalarField &field);
    const Triangle_t *getTrianglesArray() const { return mTriangles.data(); }

    std::vector<Triangle_t> mTriangles;
};

#endif // TREE_MESH_BUILDER_H
