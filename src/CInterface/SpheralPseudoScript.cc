//---------------------------------Spheral++----------------------------------//
// A fake main script to construct the pieces of a Spheral simulation for 
// calling from assorted host codes that don't want to use python.
//
// An instance of this class essentially takes the place of writing Spheral
// python script.
//
// This is implemented as a singleton so the host can create one out of the
// ether and access it whenever needed.
// 
// This is the base class for SpheralPsuedoScript2D and SpheralPsuedoScript3D,
// the idea being to put the common code here to be shared by the two
// concrete dimensional types.
// 
// Created by JMO, Thu Feb 28 2013
//----------------------------------------------------------------------------//
#include "SpheralPseudoScript.hh"
#include "SolidMaterial/LinearPolynomialEquationOfState.hh"
#include "SolidMaterial/NullStrength.hh"
#include "Kernel/NBSplineKernel.hh"
#include "Kernel/GaussianKernel.hh"
#include "Kernel/PiGaussianKernel.hh"
#include "NodeGenerators/fillFacetedVolume.hh"
#include "NodeGenerators/generateCylDistributionFromRZ.hh"
#include "NodeList/SPHSmoothingScale.hh"
#include "NodeList/ASPHSmoothingScale.hh"
#include "SPH/SolidSPHHydroBase.hh"
#include "SPH/SolidSPHHydroBaseRZ.hh"
#include "CRKSPH/SolidCRKSPHHydroBase.hh"
#include "CRKSPH/SolidCRKSPHHydroBaseRZ.hh"
#include "ArtificialViscosity/MonaghanGingoldViscosity.hh"
#include "ArtificialViscosity/TensorMonaghanGingoldViscosity.hh"
#include "ArtificialViscosity/CRKSPHMonaghanGingoldViscosity.hh"
#include "Hydro/HydroFieldNames.hh"
#include "Strength/SolidFieldNames.hh"
#include "Damage/computeFragmentField.hh"
#include "DataBase/ReplaceFieldList.hh"
#include "DataBase/IncrementFieldList.hh"
#include "DataBase/ReplaceBoundedFieldList.hh"
#include "Utilities/DataTypeTraits.hh"
#include "Utilities/iterateIdealH.hh"
#include "Utilities/globalNodeIDsInline.hh"
#if USE_MPI
#include "Distributed/NestedGridDistributedBoundary.hh"
#include "Distributed/TreeDistributedBoundary.hh"
#endif
#include "Boundary/PeriodicBoundary.hh"
#include "Boundary/ReflectingBoundary.hh"
#include "Boundary/AxisBoundaryRZ.hh"
#include "Field/Field.hh"
#include "Field/FieldListSet.hh"
#include "FieldOperations/sampleMultipleFields2Lattice.hh"

namespace Spheral {

using namespace std;

//------------------------------------------------------------------------------
// Local utility functions.
//------------------------------------------------------------------------------
namespace {

//------------------------------------------------------------------------------
// Copy a set of C arrays to a FieldList.  Assume all elements are doubles, and
// the sizing between the array and the number of elements in the FieldList is
// correct.
//------------------------------------------------------------------------------
template<typename Dimension>
void
copyArrayToScalarFieldList(const double* array,
                           FieldList<Dimension, typename Dimension::Scalar>& fieldList) {
  const unsigned nfields = fieldList.numFields();
  unsigned k = 0;
  for (unsigned i = 0; i != nfields; ++i) {
    const unsigned n = fieldList[i]->numInternalElements();
    std::copy(&array[k], &array[k] + n, &(*fieldList[i]->begin()));
    k += n;
  }
}

// Same as above for ints.
template<typename Dimension>
void
copyArrayToIntFieldList(const int* array,
                        FieldList<Dimension, int>& fieldList) {
  const unsigned nfields = fieldList.numFields();
  unsigned k = 0;
  for (unsigned i = 0; i != nfields; ++i) {
    const unsigned n = fieldList[i]->numInternalElements();
    std::copy(&array[k], &array[k] + n, &(*fieldList[i]->begin()));
    k += n;
  }
}

//------------------------------------------------------------------------------
// Copy a set of C arrays to a VectorFieldList.  Assume all elements are doubles, and
// the sizing between the array and the number of elements in the FieldList is
// correct.
// The argument passed in should be a vector<double*>, with each double* array
// arranged as
//      arrays[0] = x_array
//      arrays[1] = y_array
//      ...
//------------------------------------------------------------------------------
template<typename Dimension>
void
copyArrayToVectorFieldList(const double** arrays,
                           FieldList<Dimension, typename Dimension::Vector>& fieldList) {
  const unsigned nfields = fieldList.numFields();
  unsigned offset = 0;
  for (unsigned i = 0; i != nfields; ++i) {
    const unsigned n = fieldList[i]->numInternalElements();
    for (unsigned j = 0; j != n; ++j) {
      for (unsigned k = 0; k != Dimension::nDim; ++k) {
        fieldList(i,j)[k] = arrays[k][offset + j];
      }
    }
    offset += n;
  }
}

//------------------------------------------------------------------------------
// Copy a set of C arrays to a TensorFieldList.  Assume all elements are doubles, and
// the sizing between the array and the number of elements in the FieldList is
// correct.
// The argument passed in should be a vector<double*>, with each double* array
// arranged as
//                2D                        3D
//      arrays[0] = xx_array      arrays[0] = xx_array
//      arrays[1] = xy_array      arrays[1] = xy_array
//      arrays[2] = yx_array      arrays[2] = xz_array
//      arrays[3] = yy_array      arrays[3] = yx_array
//                                arrays[4] = yy_array
//                                arrays[5] = yz_array
//                                arrays[6] = zx_array
//                                arrays[7] = zy_array
//                                arrays[8] = zz_array
//------------------------------------------------------------------------------
template<typename Dimension>
void
copyArrayToTensorFieldList(const double** arrays,
                           FieldList<Dimension, typename Dimension::Tensor>& fieldList) {
  const unsigned nelems = Dimension::nDim*Dimension::nDim;
  const unsigned nfields = fieldList.numFields();
  unsigned offset = 0;
  for (unsigned i = 0; i != nfields; ++i) {
    const unsigned n = fieldList[i]->numInternalElements();
    for (unsigned j = 0; j != n; ++j) {
      for (unsigned k = 0; k != nelems; ++k) {
        fieldList(i,j)[k] = arrays[k][offset + j];
      }
    }
    offset += n;
  }
}

//------------------------------------------------------------------------------
// Copy a set of C arrays to a SymTensorFieldList.  Assume all elements are doubles, and
// the sizing between the array and the number of elements in the FieldList is
// correct.
// The argument passed in should be a vector<double*>, with each double* array
// arranged as
//                2D                        3D
//      arrays[0] = xx_array      arrays[0] = xx_array
//      arrays[1] = xy_array      arrays[1] = xy_array
//      arrays[2] = yy_array      arrays[2] = xz_array
//                                arrays[3] = yy_array
//                                arrays[4] = yz_array
//                                arrays[5] = zz_array
//------------------------------------------------------------------------------
template<typename Dimension>
void
copyArrayToSymTensorFieldList(const double** arrays,
                              FieldList<Dimension, typename Dimension::SymTensor>& fieldList) {
  const unsigned nelems = (Dimension::nDim == 2 ? 3 : 6);
  const unsigned nfields = fieldList.numFields();
  unsigned offset = 0;
  for (unsigned i = 0; i != nfields; ++i) {
    const unsigned n = fieldList[i]->numInternalElements();
    for (unsigned j = 0; j != n; ++j) {
      for (unsigned k = 0; k != nelems; ++k) {
        fieldList(i,j)[k] = arrays[k][offset + j];
      }
    }
    offset += n;
  }
}

// Same as above but for double** vs. const double**.  Yeesh.
template<typename Dimension>
void
copyArrayToSymTensorFieldList(double** arrays,
                              FieldList<Dimension, typename Dimension::SymTensor>& fieldList) {
  copyArrayToSymTensorFieldList(const_cast<const double**>(arrays), fieldList);
}

// Similar to above, but copy scalar to diagonal elements.
template<typename Dimension>
void
copyArrayToSymTensorFieldList(const double* diag_array,
                              FieldList<Dimension, typename Dimension::SymTensor>& fieldList) {
  typedef typename Dimension::SymTensor SymTensor;
  const unsigned nfields = fieldList.numFields();
  unsigned offset = 0;
  for (unsigned i = 0; i != nfields; ++i) {
    const unsigned n = fieldList[i]->numInternalElements();
    for (unsigned j = 0; j != n; ++j) {
      fieldList(i,j) = SymTensor::one * diag_array[offset + j];
    }
    offset += n;
  }
}

//------------------------------------------------------------------------------
// Copy a FieldList to a set of C arrays.  In this case we only copy internal
// values back to the C arrays.
//------------------------------------------------------------------------------
template<typename Dimension>
void
copyScalarFieldListToArray(const FieldList<Dimension, typename Dimension::Scalar>& fieldList,
                           double* array) {
  const unsigned nfields = fieldList.numFields();
  unsigned k = 0;
  for (unsigned i = 0; i != nfields; ++i) {
    const unsigned n = fieldList[i]->numInternalElements();
    std::copy(fieldList[i]->begin(), fieldList[i]->begin() + n, &array[k]);
    k += n;
  }
}

// Same as above for ints.
template<typename Dimension>
void
copyIntFieldListToArray(const FieldList<Dimension, int>& fieldList,
                        int* array) {
  const unsigned nfields = fieldList.numFields();
  unsigned k = 0;
  for (unsigned i = 0; i != nfields; ++i) {
    const unsigned n = fieldList[i]->numInternalElements();
    std::copy(fieldList[i]->begin(), fieldList[i]->begin() + n, &array[k]);
    k += n;
  }
}

//------------------------------------------------------------------------------
// Copy a VectorFieldList to a set of C arrays.  Assume all elements are doubles, and
// the sizing between the array and the number of elements in the FieldList is
// correct.
// The argument passed in should be a vector<double*>, with each double* array
// arranged as
//      arrays[0] = x_array
//      arrays[1] = y_array
//      ...
//------------------------------------------------------------------------------
template<typename Dimension>
void
copyVectorFieldListToArray(const FieldList<Dimension, typename Dimension::Vector>& fieldList,
                           double** arrays) {
  const unsigned nfields = fieldList.numFields();
  unsigned offset = 0;
  for (unsigned i = 0; i != nfields; ++i) {
    const unsigned n = fieldList[i]->numInternalElements();
    for (unsigned j = 0; j != n; ++j) {
      for (unsigned k = 0; k != Dimension::nDim; ++k) {
        arrays[k][offset + j] = fieldList(i,j)[k];
      }
    }
    offset += n;
  }
}

//------------------------------------------------------------------------------
// Copy a TensorFieldList to a set of C arrays.  Assume all elements are doubles, and
// the sizing between the array and the number of elements in the FieldList is
// correct.
// The argument passed in should be a vector<double*>, with each double* array
// arranged as
//                2D                        3D
//      arrays[0] = xx_array      arrays[0] = xx_array
//      arrays[1] = xy_array      arrays[1] = xy_array
//      arrays[2] = yx_array      arrays[2] = xz_array
//      arrays[3] = yy_array      arrays[3] = yx_array
//                                arrays[4] = yy_array
//                                arrays[5] = yz_array
//                                arrays[6] = zx_array
//                                arrays[7] = zy_array
//                                arrays[8] = zz_array
//------------------------------------------------------------------------------
template<typename Dimension>
void
copyTensorFieldListToArray(const FieldList<Dimension, typename Dimension::Tensor>& fieldList,
                           double** arrays) {
  const unsigned nelems = Dimension::nDim*Dimension::nDim;
  const unsigned nfields = fieldList.numFields();
  unsigned offset = 0;
  for (unsigned i = 0; i != nfields; ++i) {
    const unsigned n = fieldList[i]->numInternalElements();
    for (unsigned j = 0; j != n; ++j) {
      for (unsigned k = 0; k != nelems; ++k) {
        arrays[k][offset + j] = fieldList(i,j)[k];
      }
    }
    offset += n;
  }
}

//------------------------------------------------------------------------------
// Copy a SymTensorFieldList to a set of C arrays.  Assume all elements are doubles, and
// the sizing between the array and the number of elements in the FieldList is
// correct.
// The argument passed in should be a vector<double*>, with each double* array
// arranged as
//                2D                        3D
//      arrays[0] = xx_array      arrays[0] = xx_array
//      arrays[1] = xy_array      arrays[1] = xy_array
//      arrays[2] = yy_array      arrays[2] = xz_array
//                                arrays[3] = yy_array
//                                arrays[4] = yz_array
//                                arrays[5] = zz_array
//------------------------------------------------------------------------------
template<typename Dimension>
void
copySymTensorFieldListToArray(const FieldList<Dimension, typename Dimension::SymTensor>& fieldList,
                              double** arrays) {
  const unsigned nelems = (Dimension::nDim == 2 ? 3 : 6);
  const unsigned nfields = fieldList.numFields();
  unsigned offset = 0;
  for (unsigned i = 0; i != nfields; ++i) {
    const unsigned n = fieldList[i]->numInternalElements();
    for (unsigned j = 0; j != n; ++j) {
      for (unsigned k = 0; k != nelems; ++k) {
        arrays[k][offset + j] = fieldList(i,j)[k];
      }
    }
    offset += n;
  }
}

//------------------------------------------------------------------------------
// Struct to determine the correct hydro object to build, based on
// Dimension as a template and axisymmetry as an argument.
//------------------------------------------------------------------------------
template<typename Dimension> struct HydroConstructor;

// 3D 
template<> struct HydroConstructor<Dim<3>> {
  static std::shared_ptr<Physics<Dim<3>>> newinstance(const bool CRK,
                                                      const SmoothingScaleBase<Dim<3>>& smoothingScaleMethod,
                                                      ArtificialViscosity<Dim<3>>& Q,
                                                      const TableKernel<Dim<3>>& W,
                                                      const TableKernel<Dim<3>>& WPi,
                                                      const TableKernel<Dim<3>>& WGrad,
                                                      const double filter,
                                                      const double cfl,
                                                      const bool useVelocityMagnitudeForDt,
                                                      const bool compatibleEnergyEvolution,
                                                      const bool evolveTotalEnergy,
                                                      const bool gradhCorrection,
                                                      const bool XSPH,
                                                      const bool correctVelocityGradient,
                                                      const bool sumMassDensityOverAllNodeLists,
                                                      const MassDensityType densityUpdate,
                                                      const HEvolutionType HUpdate,
                                                      const CRKOrder correctionOrder,
                                                      const CRKVolumeType volumeType,
                                                      const double epsTensile,
                                                      const double nTensile,
                                                      const bool limitMultimaterialTopology,
                                                      const bool damageRelieveRubble,
                                                      const bool negativePressureInDamage,
                                                      const Dim<3>::Vector& xmin,
                                                      const Dim<3>::Vector& xmax,
                                                      const bool RZ) {
    if (CRK) {
      return std::shared_ptr<Physics<Dim<3>>>(new SolidCRKSPHHydroBase<Dim<3>>(smoothingScaleMethod,
                                                                               Q,
                                                                               W,
                                                                               WPi,
                                                                               filter,
                                                                               cfl,
                                                                               useVelocityMagnitudeForDt,
                                                                               compatibleEnergyEvolution,
                                                                               evolveTotalEnergy,
                                                                               XSPH,
                                                                               densityUpdate,
                                                                               HUpdate,
                                                                               correctionOrder,
                                                                               volumeType,
                                                                               epsTensile,
                                                                               nTensile,
                                                                               limitMultimaterialTopology,
                                                                               damageRelieveRubble,
                                                                               negativePressureInDamage));
    }
    else {
      return std::shared_ptr<Physics<Dim<3>>>(new SolidSPHHydroBase<Dim<3>>(smoothingScaleMethod,
                                                                            Q,
                                                                            W,
                                                                            WPi,
                                                                            WGrad,
                                                                            filter,
                                                                            cfl,
                                                                            useVelocityMagnitudeForDt,
                                                                            compatibleEnergyEvolution,
                                                                            evolveTotalEnergy,
                                                                            gradhCorrection,
                                                                            XSPH,
                                                                            correctVelocityGradient,
                                                                            sumMassDensityOverAllNodeLists,
                                                                            densityUpdate,
                                                                            HUpdate,
                                                                            epsTensile,
                                                                            nTensile,
                                                                            damageRelieveRubble,
                                                                            negativePressureInDamage,
                                                                            xmin,
                                                                            xmax));
    }
  }

  static void addBoundaries(const bool RZ,
                            vector<std::shared_ptr<Boundary<Dim<3>>>>& bounds) {}
};

// 2D 
template<> struct HydroConstructor<Dim<2>> {
  static std::shared_ptr<Physics<Dim<2>>> newinstance(const bool CRK,
                                                      const SmoothingScaleBase<Dim<2>>& smoothingScaleMethod,
                                                      ArtificialViscosity<Dim<2>>& Q,
                                                      const TableKernel<Dim<2>>& W,
                                                      const TableKernel<Dim<2>>& WPi,
                                                      const TableKernel<Dim<2>>& WGrad,
                                                      const double filter,
                                                      const double cfl,
                                                      const bool useVelocityMagnitudeForDt,
                                                      const bool compatibleEnergyEvolution,
                                                      const bool evolveTotalEnergy,
                                                      const bool gradhCorrection,
                                                      const bool XSPH,
                                                      const bool correctVelocityGradient,
                                                      const bool sumMassDensityOverAllNodeLists,
                                                      const MassDensityType densityUpdate,
                                                      const HEvolutionType HUpdate,
                                                      const CRKOrder correctionOrder,
                                                      const CRKVolumeType volumeType,
                                                      const double epsTensile,
                                                      const double nTensile,
                                                      const bool limitMultimaterialTopology,
                                                      const bool damageRelieveRubble,
                                                      const bool negativePressureInDamage,
                                                      const Dim<2>::Vector& xmin,
                                                      const Dim<2>::Vector& xmax,
                                                      const bool RZ) {
    if (RZ) {
      if (CRK) {
        return std::shared_ptr<Physics<Dim<2>>>(new SolidCRKSPHHydroBaseRZ(smoothingScaleMethod,
                                                                           Q,
                                                                           W,
                                                                           WPi,
                                                                           filter,
                                                                           cfl,
                                                                           useVelocityMagnitudeForDt,
                                                                           compatibleEnergyEvolution,
                                                                           evolveTotalEnergy,
                                                                           XSPH,
                                                                           densityUpdate,
                                                                           HUpdate,
                                                                           correctionOrder,
                                                                           volumeType,
                                                                           epsTensile,
                                                                           nTensile,
                                                                           limitMultimaterialTopology,
                                                                           damageRelieveRubble,
                                                                           negativePressureInDamage));
      }
      else {
        return std::shared_ptr<Physics<Dim<2>>>(new SolidSPHHydroBaseRZ(smoothingScaleMethod,
                                                                        Q,
                                                                        W,
                                                                        WPi,
                                                                        WGrad,
                                                                        filter,
                                                                        cfl,
                                                                        useVelocityMagnitudeForDt,
                                                                        compatibleEnergyEvolution,
                                                                        evolveTotalEnergy,
                                                                        gradhCorrection,
                                                                        XSPH,
                                                                        correctVelocityGradient,
                                                                        sumMassDensityOverAllNodeLists,
                                                                        densityUpdate,
                                                                        HUpdate,
                                                                        epsTensile,
                                                                        nTensile,
                                                                        damageRelieveRubble,
                                                                        negativePressureInDamage,
                                                                        xmin,
                                                                        xmax));
      }
    } else {
      if (CRK) {
        return std::shared_ptr<Physics<Dim<2>>>(new SolidCRKSPHHydroBase<Dim<2>>(smoothingScaleMethod,
                                                                                 Q,
                                                                                 W,
                                                                                 WPi,
                                                                                 filter,
                                                                                 cfl,
                                                                                 useVelocityMagnitudeForDt,
                                                                                 compatibleEnergyEvolution,
                                                                                 evolveTotalEnergy,
                                                                                 XSPH,
                                                                                 densityUpdate,
                                                                                 HUpdate,
                                                                                 correctionOrder,
                                                                                 volumeType,
                                                                                 epsTensile,
                                                                                 nTensile,
                                                                                 limitMultimaterialTopology,
                                                                                 damageRelieveRubble,
                                                                                 negativePressureInDamage));
      }
      else {
        return std::shared_ptr<Physics<Dim<2>>>(new SolidSPHHydroBase<Dim<2>>(smoothingScaleMethod,
                                                                              Q,
                                                                              W,
                                                                              WPi,
                                                                              WGrad,
                                                                              filter,
                                                                              cfl,
                                                                              useVelocityMagnitudeForDt,
                                                                              compatibleEnergyEvolution,
                                                                              evolveTotalEnergy,
                                                                              gradhCorrection,
                                                                              XSPH,
                                                                              correctVelocityGradient,
                                                                              sumMassDensityOverAllNodeLists,
                                                                              densityUpdate,
                                                                              HUpdate,
                                                                              epsTensile,
                                                                              nTensile,
                                                                              damageRelieveRubble,
                                                                              negativePressureInDamage,
                                                                              xmin,
                                                                              xmax));
      }
    }
  }

  static void addBoundaries(const bool RZ,
                            vector<std::shared_ptr<Boundary<Dim<2>>>>& bounds) {
    if (RZ) {
      bounds.push_back(std::shared_ptr<Boundary<Dim<2>>>(new AxisBoundaryRZ(0.25)));
    }
  }

};

}

//------------------------------------------------------------------------------
// Get the instance.
//------------------------------------------------------------------------------
template<typename Dimension>
SpheralPseudoScript<Dimension>&
SpheralPseudoScript<Dimension>::
instance() {
  if (mInstancePtr == 0) mInstancePtr = new SpheralPseudoScript<Dimension>();
  CHECK(mInstancePtr != 0);
  return *mInstancePtr;
}

//------------------------------------------------------------------------------
// initialize (called once at beginning of simulation).
//------------------------------------------------------------------------------
template<typename Dimension>
void
SpheralPseudoScript<Dimension>::
initialize(const bool     RZ,
           const bool     CRK,
           const bool     ASPH,
           const bool     XSPH,
           const bool     compatibleEnergy,
           const bool     totalEnergy,
           const bool     vGradCorrection,
           const bool     hGradCorrection,
           const bool     sumMassDensity,
           const bool     useVelocityDt,
           const bool     ScalarQ,
           const int      distributedBoundary,
           const int      kernelType,
           const int      piKernelType,
           const int      gradKernelType,
           const int      nbspline,
           const int      crkorder,
           const int      damage,
           const unsigned nmats,
           const double   CFL,
           const double   hmin,
           const double   hmax,
           const double   hminmaxratio,
           const double   nPerh,
           const double   Clinear,
           const double   Cquadratic,
           const Vector&  xmin,
           const Vector&  xmax) {

  // Pre-conditions.
  VERIFY2(not RZ or Dimension::nDim == 2,
          "SpheralPseudoScript::initialize: Axisymmetric coordinates (RZ) can only be requested in the 2D instantiation of Spheral.");

  VERIFY2(distributedBoundary >= 0 && distributedBoundary < 3,
          "SpheralPseudoScript::initialize: Distributed boundary option must be 0 (None), 1 (Nested Grid), or 2 (Tree).");

  VERIFY2(crkorder >= 0 && crkorder < 3,
          "SpheralPseudoScript::initialize: CRK correction order must be 0 (Constant), 1 (Linear), or 2 (Quadratic).");

  VERIFY2(kernelType >= 0 && kernelType < 3 && piKernelType >= 0 && piKernelType < 3 && gradKernelType >= 0 && gradKernelType < 3,
          "SpheralPseudoScript::initialize: SPH kernel type must be 0 (NBSpline), 1 (Gaussian), or 2 (PiGaussian).");

  // Get our instance.
  SpheralPseudoScript<Dimension>& me = SpheralPseudoScript<Dimension>::instance();

  // Create internal units (cm, gm, usec).
  me.mUnitsPtr.reset(new PhysicalConstants(0.01,     // unit length (m)
                                           0.001,    // unit mass (kg)
                                           1.0e-6)); // unit time (sec)

  // Construct the stand-in fake EOS and strength model.  The host code will
  // actually fill in the state fields Spheral normally uses these for.
  me.mEOSptr.reset(new LinearPolynomialEquationOfState<Dimension>(1.0, 0.1, 10.0, 
                                                                  1.0, 0.0, 0.0, 0.0,
                                                                  1.0, 0.0, 0.0,
                                                                  100.0, 
                                                                  *me.mUnitsPtr,
                                                                  0.0, 0.0, 1e100,
                                                                  MaterialPressureMinType::ZeroPressure));
  me.mStrengthModelPtr.reset(new NullStrength<Dimension>());

  // Build the general interpolation kernel.
  if(kernelType == 0) {
    me.mKernelPtr.reset(new TableKernel<Dimension>(NBSplineKernel<Dimension>(nbspline), 1000));
  }
  else if(kernelType == 1) {
    me.mKernelPtr.reset(new TableKernel<Dimension>(GaussianKernel<Dimension>(3.0), 1000));
  }
  else if(kernelType == 2) {
    me.mKernelPtr.reset(new TableKernel<Dimension>(PiGaussianKernel<Dimension>(7.0), 1000));
  }

  // Build the interpolation kernel for artificial viscosity.
  if(piKernelType == 0) {
    me.mPiKernelPtr.reset(new TableKernel<Dimension>(NBSplineKernel<Dimension>(nbspline), 1000));
  }
  else if(piKernelType == 1) {
    me.mPiKernelPtr.reset(new TableKernel<Dimension>(GaussianKernel<Dimension>(3.0), 1000));
  }
  else if(piKernelType == 2) {
    me.mPiKernelPtr.reset(new TableKernel<Dimension>(PiGaussianKernel<Dimension>(7.0), 1000));
  }

  // Build the interpolation kernel for the velocity gradient.
  if(gradKernelType == 0) {
    me.mGradKernelPtr.reset(new TableKernel<Dimension>(NBSplineKernel<Dimension>(nbspline), 1000));
  }
  else if(gradKernelType == 1) {
    me.mGradKernelPtr.reset(new TableKernel<Dimension>(GaussianKernel<Dimension>(3.0), 1000));
  }
  else if(gradKernelType == 2) {
    me.mGradKernelPtr.reset(new TableKernel<Dimension>(PiGaussianKernel<Dimension>(7.0), 1000));
  }

  // Construct the NodeLists for our materials.
  // me.mNodeLists.reserve(nmats);
  // me.mNeighbors.reserve(nmats);
  me.mNodeLists.clear();
  me.mNeighbors.clear();
  me.mHostCodeBoundaries.clear();
  for (unsigned imat = 0; imat != nmats; ++imat) {
    const string name = "NodeList " + std::to_string(imat);
    me.mNodeLists.push_back(std::shared_ptr< SolidNodeList<Dimension>>(new SolidNodeList<Dimension>(name,
                                                                                                    *me.mEOSptr,
                                                                                                    *me.mStrengthModelPtr,
                                                                                                    0,
                                                                                                    0,
                                                                                                    hmin,
                                                                                                    hmax,
                                                                                                    hminmaxratio,
                                                                                                    nPerh,
                                                                                                    500,      // maxNumNeighbors -- not currently used
                                                                                                    0.0,      // rhoMin -- we depend on the host code
                                                                                                    1e100))); // rhoMax -- we depend on the host code
    if (distributedBoundary == 2) {
      me.mNeighbors.push_back(std::shared_ptr< TreeNeighbor<Dimension>>(new TreeNeighbor<Dimension>(*me.mNodeLists.back(),
                                                                                                    NeighborSearchType::GatherScatter,
                                                                                                    me.mKernelPtr->kernelExtent(),
                                                                                                    xmin,
                                                                                                    xmax)));
    }
    else if (distributedBoundary == 1) {
      me.mNeighbors.push_back(std::shared_ptr< NestedGridNeighbor<Dimension>>(new NestedGridNeighbor<Dimension>(*me.mNodeLists.back(),
                                                                                                                NeighborSearchType::GatherScatter,
                                                                                                                31,
                                                                                                                (xmax - xmin).maxElement(),
                                                                                                                Vector::zero,
                                                                                                                me.mKernelPtr->kernelExtent(),
                                                                                                                1)));
    }
    me.mNodeLists.back()->registerNeighbor(*me.mNeighbors.back());
  }

  // Build the database and add our NodeLists.
  me.mDataBasePtr.reset(new DataBase<Dimension>());
  for (unsigned imat = 0; imat != nmats; ++imat) {
    me.mDataBasePtr->appendNodeList(*me.mNodeLists[imat]);
  }

  CRKOrder correctionOrder;
  if (crkorder == 0) {
    correctionOrder = CRKOrder::ZerothOrder;
  }
  else if (crkorder == 1) {
    correctionOrder = CRKOrder::LinearOrder;
  }
  else if (crkorder == 2) {
    correctionOrder = CRKOrder::QuadraticOrder;
  }

  // Build the hydro physics objects.
  if (ASPH) {
    me.mSmoothingScaleMethodPtr.reset(new ASPHSmoothingScale<Dimension>());
  } else {
    me.mSmoothingScaleMethodPtr.reset(new SPHSmoothingScale<Dimension>());
  }
  if (CRK) {
    me.mQptr.reset(new CRKSPHMonaghanGingoldViscosity<Dimension>(Clinear, Cquadratic, false, false, 1.0, 0.2));
  } else {
    if (ScalarQ) {
      me.mQptr.reset(new MonaghanGingoldViscosity<Dimension>(Clinear, Cquadratic, false, false));
    } else {
      me.mQptr.reset(new TensorMonaghanGingoldViscosity<Dimension>(Clinear, Cquadratic));
    }
  }
  me.mQptr->epsilon2(0.01);
  me.mHydroPtr.reset();
  me.mHydroPtr = HydroConstructor<Dimension>::newinstance(CRK,
                                                          *me.mSmoothingScaleMethodPtr,
                                                          *me.mQptr,
                                                          *me.mKernelPtr,
                                                          *me.mPiKernelPtr,
                                                          *me.mGradKernelPtr,
                                                          0.0,                                  // filter
                                                          CFL,                                  // cfl
                                                          useVelocityDt,                        // useVelocityMagnitudeForDt
                                                          compatibleEnergy,                     // compatibleEnergyEvolution
                                                          totalEnergy,                          // evolve total energy
                                                          hGradCorrection,                      // gradhCorrection
                                                          XSPH,                                 // XSPH
                                                          vGradCorrection,                      // correctVelocityGradient
                                                          sumMassDensity,                       // sumMassDensityOverAllNodeLists
                                                          MassDensityType::RigorousSumDensity,  // densityUpdate
                                                          HEvolutionType::IdealH,               // HUpdate
                                                          correctionOrder,                      // CRK order
                                                          CRKVolumeType::CRKMassOverDensity,    // CRK volume type
                                                          0.0,                                  // epsTensile
                                                          4.0,                                  // nTensile
                                                          false,                                // limitMultimaterialTopology
                                                          false,                                // damageRelieve
                                                          false,                                // negativePressureInDamage
                                                          xmin,                                 // xmin
                                                          xmax,                                 // xmax
                                                          RZ);

  // Build a time integrator.  We're not going to use this to advance state,
  // but the other methods are useful.
  me.mIntegratorPtr.reset(new CheapSynchronousRK2<Dimension>(*me.mDataBasePtr));
  me.mIntegratorPtr->appendPhysicsPackage(*me.mHydroPtr);

  // Remember if we're feeding damage in
  me.mDamage = damage;

  // Remember the distributed boundary type.
  me.mDistributedBoundary = distributedBoundary;

  // Do the one-time initialization work for our packages.
  me.mHydroPtr->initializeProblemStartup(*me.mDataBasePtr);

  // Add the axis reflecting boundary in RZ.
  HydroConstructor<Dimension>::addBoundaries(RZ, me.mHostCodeBoundaries);
}

//------------------------------------------------------------------------------
// initializeStep -- to be called once at the beginning of a cycle.
// Returns:    the time step vote
// Arguments:  nintpermat : array indicating the number of internal nodes per material
//             npermat : array indicating the total number of nodes per material
//------------------------------------------------------------------------------
template<typename Dimension>
double
SpheralPseudoScript<Dimension>::
initializeStep(const unsigned* nintpermat,
               const unsigned* npermat,
               const double*   mass,
               const double*   massDensity,
               const double**  position,
               const double*   specificThermalEnergy,
               const double**  velocity,
               const double**  Hfield,
               const double*   pressure,
               const double**  deviatoricStress,
               const double*   deviatoricStressTT,
               const double*   soundSpeed,
               const double*   bulkModulus,
               const double*   shearModulus,
               const double*   yieldStrength,
               const double*   plasticStrain,
               const double*   scalarDamage,
               const int*      particleType) {

  // Get our instance.
  SpheralPseudoScript<Dimension>& me = SpheralPseudoScript<Dimension>::instance();

  // Check the input and set numbers of nodes.
  const unsigned nmats = me.mNodeLists.size();
  me.mNumInternalNodes = vector<unsigned>(nmats);
  me.mNumHostGhostNodes = vector<unsigned>(nmats);
  for (unsigned imat = 0; imat != nmats; ++imat) {
    VERIFY(nintpermat[imat] <= npermat[imat]);
    me.mNumInternalNodes[imat] = nintpermat[imat];
    me.mNumHostGhostNodes[imat] = npermat[imat] - nintpermat[imat];
    me.mNodeLists[imat]->numInternalNodes(me.mNumInternalNodes[imat]);
    me.mNodeLists[imat]->numGhostNodes(me.mNumHostGhostNodes[imat]);
  }

  // Prepare the state and such.
  me.mStatePtr.reset(new State<Dimension>(*me.mDataBasePtr,
                                          me.mIntegratorPtr->physicsPackagesBegin(), 
                                          me.mIntegratorPtr->physicsPackagesEnd()));
  me.mDerivsPtr.reset(new StateDerivatives<Dimension>(*me.mDataBasePtr,
                                                      me.mIntegratorPtr->physicsPackagesBegin(), 
                                                      me.mIntegratorPtr->physicsPackagesEnd()));

  // Copy the given state into Spheral's structures.
  SpheralPseudoScript::updateState(mass, 
                                   massDensity,
                                   position,
                                   specificThermalEnergy,
                                   velocity,
                                   Hfield,
                                   pressure, 
                                   deviatoricStress,
                                   deviatoricStressTT,
                                   soundSpeed, 
                                   bulkModulus, 
                                   shearModulus, 
                                   yieldStrength,
                                   plasticStrain,
                                   scalarDamage,
                                   particleType);

  // Vote on a time step and return it.
  me.mIntegratorPtr->lastDt(1e10);
  const double dt = me.mIntegratorPtr->selectDt(1e-100, 1e100, *me.mStatePtr, *me.mDerivsPtr);
  return dt;
}

//------------------------------------------------------------------------------
// Just update the state without reinitializing everything for intermediate
// step estimates of the derivatives.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SpheralPseudoScript<Dimension>::
updateState(const double*  mass,
            const double*  massDensity,
            const double** position,
            const double*  specificThermalEnergy,
            const double** velocity,
            const double** Hfield,
            const double*  pressure,
            const double** deviatoricStress,
            const double*  deviatoricStressTT,
            const double*  soundSpeed,
            const double*  bulkModulus,
            const double*  shearModulus,
            const double*  yieldStrength,
            const double*  plasticStrain,
            const double*  scalarDamage,
            const int*     particleType) {

  // Get our instance.
  auto& me = SpheralPseudoScript<Dimension>::instance();
  const unsigned nmats = me.mNodeLists.size();

  // Size the NodeLists.
  for (unsigned imat = 0; imat != nmats; ++imat) {
    me.mNodeLists[imat]->numInternalNodes(me.mNumInternalNodes[imat]);
    me.mNodeLists[imat]->numGhostNodes(me.mNumHostGhostNodes[imat]);
  }

  // Pull the state fields.
  auto m = me.mStatePtr->fields(HydroFieldNames::mass, 0.0);
  auto pos = me.mStatePtr->fields(HydroFieldNames::position, Vector::zero);
  auto vel = me.mStatePtr->fields(HydroFieldNames::velocity, Vector::zero);
  auto rho = me.mStatePtr->fields(HydroFieldNames::massDensity, 0.0);
  auto eps = me.mStatePtr->fields(HydroFieldNames::specificThermalEnergy, 0.0);
  auto H = me.mStatePtr->fields(HydroFieldNames::H, SymTensor::zero);
  auto P = me.mStatePtr->fields(HydroFieldNames::pressure, 0.0);
  auto cs = me.mStatePtr->fields(HydroFieldNames::soundSpeed, 0.0);
  auto S = me.mStatePtr->fields(SolidFieldNames::deviatoricStress, SymTensor::zero);
  auto STT = me.mStatePtr->fields(SolidFieldNames::deviatoricStressTT, 0.0);
  auto ps = me.mStatePtr->fields(SolidFieldNames::plasticStrain, 0.0);
  auto pType = me.mStatePtr->fields(SolidFieldNames::particleTypes, 0);
  auto K = me.mStatePtr->fields(SolidFieldNames::bulkModulus, 0.0);
  auto mu = me.mStatePtr->fields(SolidFieldNames::shearModulus, 0.0);
  auto Y = me.mStatePtr->fields(SolidFieldNames::yieldStrength, 0.0);
  auto D = me.mStatePtr->fields(SolidFieldNames::effectiveTensorDamage, SymTensor::zero);

  // Fill in the material properties.
  if (mass != NULL)                  copyArrayToScalarFieldList(mass, m);
  if (position[0] != NULL)           copyArrayToVectorFieldList(position, pos);
  if (velocity[0] != NULL)           copyArrayToVectorFieldList(velocity, vel);
  if (massDensity != NULL)           copyArrayToScalarFieldList(massDensity, rho);
  if (specificThermalEnergy != NULL) copyArrayToScalarFieldList(specificThermalEnergy, eps);
  if (Hfield[0] != NULL)             copyArrayToSymTensorFieldList(Hfield, H);
  if (pressure != NULL)              copyArrayToScalarFieldList(pressure, P);
  if (deviatoricStress[0] != NULL)   copyArrayToSymTensorFieldList(deviatoricStress, S);
  if (deviatoricStressTT != NULL)    copyArrayToScalarFieldList(deviatoricStressTT, STT);
  if (soundSpeed != NULL)            copyArrayToScalarFieldList(soundSpeed, cs);
  if (bulkModulus != NULL)           copyArrayToScalarFieldList(bulkModulus, K);
  if (shearModulus != NULL)          copyArrayToScalarFieldList(shearModulus, mu);
  if (yieldStrength != NULL)         copyArrayToScalarFieldList(yieldStrength, Y);
  if (plasticStrain != NULL)         copyArrayToScalarFieldList(plasticStrain, ps);
  if (particleType != NULL)          copyArrayToIntFieldList(particleType, pType);
  if (me.mDamage) {
    if (scalarDamage != NULL)        copyArrayToSymTensorFieldList(scalarDamage, D);
  }

  // Add host code boundaries
  me.mHydroPtr->clearBoundaries();
  for (unsigned i = 0; i<me.mHostCodeBoundaries.size(); ++i) {
    me.mHydroPtr->appendBoundary(*me.mHostCodeBoundaries[i]);
  }

  // If requested, add the appropriate DistributedBoundary for Spheral to handle
  // distributed ghost nodes.
#if USE_MPI
  if (me.mDistributedBoundary == 2) {
    me.mHydroPtr->appendBoundary(TreeDistributedBoundary<Dimension>::instance());
  }
  else if (me.mDistributedBoundary == 1) {
    me.mHydroPtr->appendBoundary(NestedGridDistributedBoundary<Dimension>::instance());
  }
#endif

  // pre-step initialize
  me.mIntegratorPtr->preStepInitialize(*me.mStatePtr, *me.mDerivsPtr);
}

//------------------------------------------------------------------------------
// evaluateDerivatives
//------------------------------------------------------------------------------
template<typename Dimension>
void
SpheralPseudoScript<Dimension>::
evaluateDerivatives(double*  massDensitySum,
                    double*  DmassDensityDt,
                    double** DvelocityDt,
                    double** DxDt,
                    double*  DspecificThermalEnergyDt,
                    double** DvelocityDx,
                    double** DHfieldDt,
                    double** HfieldIdeal,
                    double** DdeviatoricStressDt,
                    double*  DdeviatoricStressDtTT,
                    double*  qpressure,
                    double*  qwork) {

  // Get our instance.
  auto& me = SpheralPseudoScript<Dimension>::instance();

  // Zero out the stored derivatives.
  me.mDerivsPtr->Zero();

  // Have Spheral evaluate the fluid derivatives.
  me.mIntegratorPtr->initializeDerivatives(0.0, 1.0, *me.mStatePtr, *me.mDerivsPtr);
  me.mIntegratorPtr->evaluateDerivatives(0.0, 1.0, *me.mDataBasePtr, *me.mStatePtr, *me.mDerivsPtr);
  me.mIntegratorPtr->finalizeDerivatives(0.0, 1.0, *me.mDataBasePtr, *me.mStatePtr, *me.mDerivsPtr);

  // Pull the individual derivative state fields.
  auto rhoSum = me.mDerivsPtr->fields(ReplaceFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::massDensity, 0.0);
  auto DrhoDt = me.mDerivsPtr->fields(IncrementFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::massDensity, 0.0);
  auto DvDt = me.mDerivsPtr->fields(HydroFieldNames::hydroAcceleration, Vector::zero);
  auto DposDt = me.mDerivsPtr->fields(IncrementFieldList<Dimension, Vector>::prefix() + HydroFieldNames::position, Vector::zero);
  auto DepsDt = me.mDerivsPtr->fields(IncrementFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::specificThermalEnergy, 0.0);
  auto DvDx = me.mDerivsPtr->fields(HydroFieldNames::velocityGradient, Tensor::zero);
  auto DHDt = me.mDerivsPtr->fields(IncrementFieldList<Dimension, SymTensor>::prefix() + HydroFieldNames::H, SymTensor::zero);
  auto Hideal = me.mDerivsPtr->fields(ReplaceBoundedFieldList<Dimension, SymTensor>::prefix() + HydroFieldNames::H, SymTensor::zero);
  auto DSDt = me.mDerivsPtr->fields(IncrementFieldList<Dimension, SymTensor>::prefix() + SolidFieldNames::deviatoricStress, SymTensor::zero);
  auto DSDtTT = me.mDerivsPtr->fields(IncrementFieldList<Dim<2>, Scalar>::prefix() + SolidFieldNames::deviatoricStressTT, 0.0);
  auto effViscousPressure = me.mDerivsPtr->fields(HydroFieldNames::effectiveViscousPressure, 0.0);
  auto viscousWork = me.mDerivsPtr->fields(HydroFieldNames::viscousWork, 0.0);

  // Copy the fluid derivative state to the arguments.
  copyScalarFieldListToArray(rhoSum, massDensitySum);
  copyScalarFieldListToArray(DrhoDt, DmassDensityDt);
  copyVectorFieldListToArray(DvDt, DvelocityDt);
  copyVectorFieldListToArray(DposDt, DxDt);
  copyScalarFieldListToArray(DepsDt, DspecificThermalEnergyDt);
  copyTensorFieldListToArray(DvDx, DvelocityDx);
  copySymTensorFieldListToArray(DHDt, DHfieldDt);
  copySymTensorFieldListToArray(Hideal, HfieldIdeal);
  copySymTensorFieldListToArray(DSDt, DdeviatoricStressDt);
  if (DdeviatoricStressDtTT != NULL) copyScalarFieldListToArray(DSDtTT, DdeviatoricStressDtTT);
  copyScalarFieldListToArray(effViscousPressure, qpressure);
  copyScalarFieldListToArray(viscousWork, qwork);
}

//------------------------------------------------------------------------------
// addBoundary
//------------------------------------------------------------------------------
template<typename Dimension>
void
SpheralPseudoScript<Dimension>::
addBoundary(const Vector& point, 
            const Vector& normal) {

  // Get our instance.
  auto& me = SpheralPseudoScript<Dimension>::instance();

  // Add reflecting boundary
  me.mHostCodeBoundaries.push_back(std::shared_ptr<ReflectingBoundary<Dimension>>(new ReflectingBoundary<Dimension>( (GeomPlane<Dimension>(point,normal)))));
}

//------------------------------------------------------------------------------
// addPeriodicBoundary
//------------------------------------------------------------------------------
template<typename Dimension>
void
SpheralPseudoScript<Dimension>::
addPeriodicBoundary(const Vector& point1,
                    const Vector& normal1,
                    const Vector& point2,
                    const Vector& normal2) {

  // Get our instance.
  auto& me = SpheralPseudoScript<Dimension>::instance();

  // Add reflecting boundary
  me.mHostCodeBoundaries.push_back(std::shared_ptr<PeriodicBoundary<Dimension>>(new PeriodicBoundary<Dimension>(GeomPlane<Dimension>(point1,normal1),
                                                                                                                GeomPlane<Dimension>(point2,normal2))));
}

//------------------------------------------------------------------------------
// Provide a method of iterating the initial H tensors to something reasonable.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SpheralPseudoScript<Dimension>::
iterateHfield(double**     Hfield,
              const int    maxIterations,
              const double tolerance) {

  // Get our instance.
  auto& me = SpheralPseudoScript<Dimension>::instance();

  // Copy the input H info to the Spheral H field.
  auto H = me.mDataBasePtr->globalHfield();
  copyArrayToSymTensorFieldList(Hfield, H);

  // We call the Utilities helper method to do this job.
  const auto& bcs = me.mIntegratorPtr->uniqueBoundaryConditions();
  iterateIdealH(*me.mDataBasePtr, bcs, *me.mKernelPtr, *me.mSmoothingScaleMethodPtr, 
                maxIterations, tolerance);

  // Copy the internal H field back to the output.
  copySymTensorFieldListToArray(H, Hfield);
}

//------------------------------------------------------------------------------
// Compute the fragment ID field given the damage.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SpheralPseudoScript<Dimension>::
computeFragmentID(double* damage,
                  double  fragRadius,
                  double  fragDensity,
                  double  fragDamage,
                  int*    fragments) {

  // Get our instance.
  auto& me = SpheralPseudoScript<Dimension>::instance();
  const unsigned nmats = me.mNodeLists.size();

  // Size the NodeLists.
  for (unsigned imat = 0; imat != nmats; ++imat) {
    me.mNodeLists[imat]->numInternalNodes(me.mNumInternalNodes[imat]);
    me.mNodeLists[imat]->numGhostNodes(me.mNumHostGhostNodes[imat]);
  }

  // Copy damage values from the array to the field list.
  auto rho = me.mStatePtr->fields(HydroFieldNames::massDensity, 0.0);
  auto D = me.mStatePtr->fields(SolidFieldNames::effectiveTensorDamage, SymTensor::zero);
  auto fragIDs = me.mStatePtr->fields(SolidFieldNames::fragmentIDs, int(1));
  if (damage != NULL) copyArrayToSymTensorFieldList(damage, D);

  // Call the fragment ID function for each node list.
  double dimFactor = double(Dimension::nDim);
  for (unsigned imat = 0; imat != nmats; ++imat) {
    *fragIDs[imat] = computeFragmentField(*me.mNodeLists[imat], fragRadius,
                                          *rho[imat], *D[imat],
                                          fragDensity, fragDamage*dimFactor, false) ;
  }

  // Copy back to the array.
  copyIntFieldListToArray(fragIDs,fragments) ;

  // Zero out fields
  D.Zero();
  fragIDs.Zero();
}

//------------------------------------------------------------------------------
// Sample the SPH state variables to a lattice mesh.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SpheralPseudoScript<Dimension>::
sampleLatticeMesh(const Vector&  xmin,
                  const Vector&  xmax,
                  const int*     nsamples,
                  double*        latticeDensity) {

  // Get our instance.
  auto& me = SpheralPseudoScript<Dimension>::instance();
  const unsigned nmats = me.mNodeLists.size();

  // Pull the state fields.
  auto m = me.mStatePtr->fields(HydroFieldNames::mass, 0.0);
  auto rho = me.mStatePtr->fields(HydroFieldNames::massDensity, 0.0);
  auto pos = me.mStatePtr->fields(HydroFieldNames::position, Vector::zero);
  auto H = me.mStatePtr->fields(HydroFieldNames::H, SymTensor::zero);
  auto pType = me.mStatePtr->fields(SolidFieldNames::particleTypes, 0);
  auto weight = m/rho;
  auto mask = pType+1;

  FieldListSet<Dimension> sphSet;
  sphSet.ScalarFieldLists.push_back(rho);

  std::vector<int> nsample;
  for (int i = 0 ; i < Dimension::nDim ; ++i) {
    nsample.push_back(nsamples[i]);
  }

  std::vector< std::vector<Scalar>> scalarValues;
  std::vector< std::vector<Vector>> vectorValues;
  std::vector< std::vector<Tensor>> tensorValues;
  std::vector< std::vector<SymTensor>> symTensorValues;

  sampleMultipleFields2LatticeMash(sphSet, pos, weight, H, mask, *me.mKernelPtr, xmin, xmax, nsample,
                                   scalarValues, vectorValues, tensorValues, symTensorValues);

  for (int i = 0 ; i < scalarValues[0].size() ; ++i) {
    latticeDensity[i] = scalarValues[0][i] ;
  }
}

//------------------------------------------------------------------------------
// Fill a volume with evenly spaced distribution of particles.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SpheralPseudoScript<Dimension>::
fillVolume(const int*     nnodes,
           const int*     nfaces,
           const double** coords,
           const int*     conn,
           const double   spacing,
           double*        volume,
           int*           nparticles,
           double**       sphcoords) {

  if (Dimension::nDim == 3) {
    std::vector< Dim<3>::Vector > nodeVec;
    std::vector< std::vector<unsigned> > faceVec;
    for (int i = 0 ; i < nnodes[0] ; ++i) {
      nodeVec.push_back(Dim<3>::Vector(coords[0][i], coords[1][i], coords[2][i]));
    }
    for (int i = 0 ; i < nfaces[0] ; ++i) {
      std::vector<unsigned> faceSet;
      int offset = 3*i;
      faceSet.push_back(conn[offset+0]);
      faceSet.push_back(conn[offset+1]);
      faceSet.push_back(conn[offset+2]);
      faceVec.push_back(faceSet);
    }
    Dim<3>::FacetedVolume mesh(nodeVec, faceVec);
    std::vector< Dim<3>::Vector > sphNodes = fillFacetedVolume2(mesh, spacing, 0, 1);
    volume[0] = mesh.volume();
    nparticles[0] = sphNodes.size();
    double * sphcoordx = new double[sphNodes.size()];
    double * sphcoordy = new double[sphNodes.size()];
    double * sphcoordz = new double[sphNodes.size()];
    for (int i = 0 ; i < sphNodes.size() ; ++i) {
      sphcoordx[i] = sphNodes[i][0] ;
      sphcoordy[i] = sphNodes[i][1] ;
      sphcoordz[i] = sphNodes[i][2] ;
    }
    sphcoords[0] = sphcoordx ;
    sphcoords[1] = sphcoordy ;
    sphcoords[2] = sphcoordz ;
  }

}

//------------------------------------------------------------------------------
// Generate 3D particles from particles in RZ space.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SpheralPseudoScript<Dimension>::
generateCylFromRZ(const int*     nnodes,
                  const double** coords,
                  const double** htensor,
                  const double** volume,
                  const double   frac,
                  int*           nparticles,
                  double**       sphcoords,
                  double**       sphhtensor,
                  double**       sphvolume) {

  std::vector< double > xvec, yvec, zvec, mvec;
  std::vector< Dim<3>::SymTensor > Hvec;
  std::vector<int> gids;
  std::vector< vector<double> > extras;
  double nodesperh(1.5), kernelext(1.0), phi(2.0*M_PI);
  int proc(0), nprocs(1), counter(0);
  for (int i = 0 ; i < nnodes[0] ; ++i) {
    xvec.push_back(coords[0][i]);
    yvec.push_back(coords[1][i]);
    zvec.push_back(0.0);
    mvec.push_back(volume[0][i]);
    gids.push_back(counter);
    double hxx = htensor[0][i];
    double hyy = htensor[3][i];
    double hzz = htensor[5][i];
    Dim<3>::SymTensor hh(hxx, 0.0, 0.0,
                         0.0, hyy, 0.0,
                         0.0, 0.0, hzz);
    Hvec.push_back(hh);
    ++counter;
  }
  generateCylDistributionFromRZ(xvec,yvec,zvec,mvec,Hvec,gids,extras,
                                nodesperh,kernelext,phi*frac,proc,nprocs);
  int n3d = xvec.size();
  VERIFY(yvec.size() == n3d && zvec.size() == n3d &&
         mvec.size() == n3d && Hvec.size() == n3d);
  nparticles[0] = n3d;
  double * sphcoordx = new double[n3d];
  double * sphcoordy = new double[n3d];
  double * sphcoordz = new double[n3d];
  double * sphhxx = new double[n3d];
  double * sphhxy = new double[n3d];
  double * sphhxz = new double[n3d];
  double * sphhyy = new double[n3d];
  double * sphhyz = new double[n3d];
  double * sphhzz = new double[n3d];
  double * sphvol = new double[n3d];
  for (int i = 0 ; i < n3d ; ++i) {
    sphcoordx[i] = xvec[i] ;
    sphcoordy[i] = yvec[i] ;
    sphcoordz[i] = zvec[i] ;
    sphhxx[i] = Hvec[i][0] ;
    sphhxy[i] = Hvec[i][1] ;
    sphhxz[i] = Hvec[i][2] ;
    sphhyy[i] = Hvec[i][3] ;
    sphhyz[i] = Hvec[i][4] ;
    sphhzz[i] = Hvec[i][5] ;
    sphvol[i] = mvec[i] ;
  }
  sphcoords[0] = sphcoordx ;
  sphcoords[1] = sphcoordy ;
  sphcoords[2] = sphcoordz ;
  sphhtensor[0] = sphhxx ;
  sphhtensor[1] = sphhxy ;
  sphhtensor[2] = sphhxz ;
  sphhtensor[3] = sphhyy ;
  sphhtensor[4] = sphhyz ;
  sphhtensor[5] = sphhzz ;
  sphvolume[0] = sphvol ;
}

//------------------------------------------------------------------------------
// Update the connectivity between nodes using Spheral's internal neighbor
// finding.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SpheralPseudoScript<Dimension>::
updateConnectivity() {

  // Get our instance.
  auto& me = SpheralPseudoScript<Dimension>::instance();

  // We use the time integrators setGhostNodes method for this.  If we're not
  // using any of Spheral's boundary conditions this will only cause the 
  // connectivity to be recomputed.  However, if Spheral's boundary conditions
  // are being used this will also throw away and recreate the ghost nodes
  // from scratch.
  me.mIntegratorPtr->setGhostNodes();
}

//------------------------------------------------------------------------------
// Get the current connectivity.
//------------------------------------------------------------------------------
template<typename Dimension>
void
SpheralPseudoScript<Dimension>::
getConnectivity(int***  numNeighbors,
                int**** connectivity) {

  // Get our instance.
  auto& me = SpheralPseudoScript<Dimension>::instance();

  // Get the ConnectivityMap.
  const auto& cm = me.mDataBasePtr->connectivityMap();

  // Read the data out to the 4 deep C array (yipes!)
  const unsigned nmats = me.mNodeLists.size();
  CHECK(cm.nodeLists().size() == nmats);
  numNeighbors = new int**[nmats];
  connectivity = new int***[nmats];
  for (unsigned imat = 0; imat != nmats; ++imat) {
    const unsigned ni = me.mNodeLists[imat]->numInternalNodes();
    numNeighbors[imat] = new int*[ni];
    connectivity[imat] = new int**[ni];
    for (unsigned i = 0; i != ni; ++i) {
      numNeighbors[imat][i] = new int[nmats];
      connectivity[imat][i] = new int*[nmats];
      const vector<vector<int>>& fullConnectivity = cm.connectivityForNode(imat, i);
      CHECK(fullConnectivity.size() == nmats);
      for (unsigned jmat = 0; jmat != nmats; ++jmat) {
        const unsigned nneigh = fullConnectivity[jmat].size();
        numNeighbors[imat][i][jmat] = nneigh;
        connectivity[imat][i][jmat] = new int[nneigh];
        std::copy(fullConnectivity[jmat].begin(), fullConnectivity[jmat].end(), connectivity[imat][i][jmat]);
      }
    }
  }
}

//------------------------------------------------------------------------------
// Get the number of materials.
//------------------------------------------------------------------------------
template<typename Dimension>
int
SpheralPseudoScript<Dimension>::
getNumMaterials() {
  return SpheralPseudoScript<Dimension>::instance().mNodeLists.size();
}

//------------------------------------------------------------------------------
// Get the number of nodes per material.
//------------------------------------------------------------------------------
template<typename Dimension>
int*
SpheralPseudoScript<Dimension>::
getNumNodes() {

  // Get our instance.
  auto& me = SpheralPseudoScript<Dimension>::instance();

  const unsigned nmats = me.mNodeLists.size();
  int* result = new int[nmats];
  for (unsigned i = 0; i != nmats; ++i) {
    result[i] = me.mNodeLists[i]->numNodes();
  }
}

//------------------------------------------------------------------------------
// Get the number of internal nodes per material.
//------------------------------------------------------------------------------
template<typename Dimension>
int*
SpheralPseudoScript<Dimension>::
getNumInternalNodes() {

  // Get our instance.
  auto& me = SpheralPseudoScript<Dimension>::instance();

  const unsigned nmats = me.mNodeLists.size();
  int* result = new int[nmats];
  for (unsigned i = 0; i != nmats; ++i) {
    result[i] = me.mNodeLists[i]->numInternalNodes();
  }
}

//------------------------------------------------------------------------------
// Get the number of ghost nodes per material.
//------------------------------------------------------------------------------
template<typename Dimension>
int*
SpheralPseudoScript<Dimension>::
getNumGhostNodes() {

  // Get our instance.
  auto& me = SpheralPseudoScript<Dimension>::instance();

  const unsigned nmats = me.mNodeLists.size();
  int* result = new int[nmats];
  for (unsigned i = 0; i != nmats; ++i) {
    result[i] = me.mNodeLists[i]->numGhostNodes();
  }
}

//------------------------------------------------------------------------------
// Constructor (private).
//------------------------------------------------------------------------------
template<typename Dimension>
SpheralPseudoScript<Dimension>::
SpheralPseudoScript() {
}

//------------------------------------------------------------------------------
// Destructor (private).
//------------------------------------------------------------------------------
template<typename Dimension>
SpheralPseudoScript<Dimension>::
~SpheralPseudoScript() {
}

//------------------------------------------------------------------------------
// Instantiations.
//------------------------------------------------------------------------------
template<typename Dimension> SpheralPseudoScript<Dimension>* SpheralPseudoScript<Dimension>::mInstancePtr = 0;
template class SpheralPseudoScript<Dim<2>>;
template class SpheralPseudoScript<Dim<3>>;

}
