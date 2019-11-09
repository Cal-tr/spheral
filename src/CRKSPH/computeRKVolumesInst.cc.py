text = """
//------------------------------------------------------------------------------
// Explicit instantiation.
//------------------------------------------------------------------------------
#include "Geometry/Dimension.hh"

#include "computeRKVolumes.cc"

namespace Spheral {

template void computeRKVolumes(const ConnectivityMap<Dim<%(ndim)s>>& connectivityMap,
                               const TableKernel<Dim<%(ndim)s>>& W,
                               const FieldList<Dim<%(ndim)s>, Dim<%(ndim)s>::Vector>& position,
                               const FieldList<Dim<%(ndim)s>, Dim<%(ndim)s>::Scalar>& mass,
                               const FieldList<Dim<%(ndim)s>, Dim<%(ndim)s>::Scalar>& massDensity,
                               const FieldList<Dim<%(ndim)s>, Dim<%(ndim)s>::SymTensor>& H,
                               const FieldList<Dim<%(ndim)s>, Dim<%(ndim)s>::SymTensor>& damage,
                               const CRKVolumeType volumeType,
                               FieldList<Dim<%(ndim)s>, int>& surfacePoint,
                               FieldList<Dim<%(ndim)s>, Dim<%(ndim)s>::Vector>& deltaCentroid,
                               FieldList<Dim<%(ndim)s>, std::vector<Dim<%(ndim)s>::Vector>>& etaVoidPoints,
                               FieldList<Dim<%(ndim)s>, Dim<%(ndim)s>::FacetedVolume>& cells,
                               FieldList<Dim<%(ndim)s>, std::vector<CellFaceFlag>>& cellFaceFlags,
                               FieldList<Dim<%(ndim)s>, Dim<%(ndim)s>::Scalar>& volume);

}

"""
