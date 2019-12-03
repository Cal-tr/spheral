//---------------------------------Spheral++----------------------------------//
// SuperiorRKCorrections
//
// Computes RK corrections for other physics packages
//----------------------------------------------------------------------------//
#ifndef __LLNLSpheral_SuperiorRKCorrections__
#define __LLNLSpheral_SuperiorRKCorrections__

#include "CRKSPH/CRKSPHCorrectionParams.hh"
#include "DataOutput/registerWithRestart.hh"
#include "Field/FieldList.hh"
#include "Geometry/CellFaceFlag.hh"
#include "Physics/Physics.hh"

namespace Spheral {

template<typename Dimension> class State;
template<typename Dimension> class StateDerivatives;
template<typename Dimension> class DataBase;
template<typename Dimension> class Boundary;

template<typename Dimension, CRKOrder correctionOrder>
class SuperiorRKCorrections : public Physics<Dimension> {
public:
  typedef typename Dimension::Scalar Scalar;
  typedef typename Dimension::Vector Vector;
  typedef typename Dimension::Tensor Tensor;
  typedef typename Dimension::SymTensor SymTensor;
  typedef typename Dimension::FacetedVolume FacetedVolume;
  
  typedef typename std::vector<Boundary<Dimension>*>::iterator BoundaryIterator;
  typedef typename std::vector<Boundary<Dimension>*>::const_iterator ConstBoundaryIterator;
  typedef typename std::pair<double, std::string> TimeStepType;

  // Constructor
  SuperiorRKCorrections(const DataBase<Dimension>& dataBase,
                        const TableKernel<Dimension>& W,
                        const CRKVolumeType volumeType,
                        const bool needHessian);

  // Destructor.
  virtual ~SuperiorRKCorrections();

  // Evaluate derivatives
  virtual void evaluateDerivatives(const Scalar time,
                                   const Scalar dt,
                                   const DataBase<Dimension>& dataBase,
                                   const State<Dimension>& state,
                                   StateDerivatives<Dimension>& derivatives) const override;
  
  // Vote on a time step.
  virtual TimeStepType dt(const DataBase<Dimension>& dataBase, 
                          const State<Dimension>& state,
                          const StateDerivatives<Dimension>& derivs,
                          const Scalar currentTime) const override;

  // Register the state
  virtual void registerState(DataBase<Dimension>& dataBase,
                             State<Dimension>& state) override;

  // Register the state derivatives
  virtual void registerDerivatives(DataBase<Dimension>& dataBase,
                                   StateDerivatives<Dimension>& derivs) override;

  // Label for the package
  virtual std::string label() const override { return "SuperiorRKCorrections"; }

  // Apply boundary conditions to ghost points
  virtual void applyGhostBoundaries(State<Dimension>& state,
                                    StateDerivatives<Dimension>& derivs) override;
  
  // Enforce boundary conditions for internal points
  virtual void enforceBoundaries(State<Dimension>& state,
                                 StateDerivatives<Dimension>& derivs) override;
  
  // Initialize field lists and calculate initial RK corrections
  virtual void initializeProblemStartup(DataBase<Dimension>& dataBase) override;
  
  // Compute the Voronoi volumes
  virtual void preStepInitialize(const DataBase<Dimension>& dataBase, 
                                 State<Dimension>& state,
                                 StateDerivatives<Dimension>& derivs) override;
  
  // Compute RK corrections
  virtual void initialize(const Scalar time, 
                          const Scalar dt,
                          const DataBase<Dimension>& dataBase, 
                          State<Dimension>& state,
                          StateDerivatives<Dimension>& derivs) override;

  // Finalize
  virtual void finalize(const Scalar time, 
                        const Scalar dt,
                        DataBase<Dimension>& dataBase, 
                        State<Dimension>& state,
                        StateDerivatives<Dimension>& derivs) override;
  
  // We do require the connecitivity
  virtual bool requireConnectivity() const override { return true; }
  
  // Methods required for restarting.
  virtual void dumpState(FileIO& file, const std::string& pathName) const;
  virtual void restoreState(const FileIO& file, const std::string& pathName);

  // Return data
  CRKVolumeType volumeType() const { return mVolumeType; }
  bool needHessian() const { return mNeedHessian; }
  const FieldList<Dimension, Scalar>& volume() const { return mVolume; }

  const FieldList<Dimension, std::vector<double>>& corrections() const { return mCorrections; }

  const FieldList<Dimension, int>& surfacePoint() const { return mSurfacePoint; }
  const FieldList<Dimension, std::vector<Vector>>& etaVoidPoints() const { return mEtaVoidPoints; }
  const FieldList<Dimension, FacetedVolume>& cells() const { return mCells; }
  const FieldList<Dimension, std::vector<CellFaceFlag>>& cellFaceFlags() const { return mCellFaceFlags; }

private:

  // Data
  const DataBase<Dimension>& mDataBase;
  const TableKernel<Dimension>& mW;
  const CRKVolumeType mVolumeType;
  const bool mNeedHessian;

  // State
  FieldList<Dimension, Scalar> mVolume;
  
  // Corrections
  FieldList<Dimension, std::vector<double>> mCorrections;
  
  // Voronoi stuff
  FieldList<Dimension, int> mSurfacePoint;
  FieldList<Dimension, std::vector<Vector>> mEtaVoidPoints;
  FieldList<Dimension, FacetedVolume> mCells;
  FieldList<Dimension, std::vector<CellFaceFlag>> mCellFaceFlags;
  FieldList<Dimension, Vector> mDeltaCentroid;
  
  // The restart registration.
  RestartRegistrationType mRestart;

  // No default constructor, copying, or assignment.
  SuperiorRKCorrections();
  SuperiorRKCorrections(const SuperiorRKCorrections&);
  SuperiorRKCorrections& operator=(const SuperiorRKCorrections&);
}; // end SuperiorRKCorrections

} // end namespace Spheral

#else

// Forward declaration
namespace Spheral {
template<typename Dimension> class SuperiorRKCorrections;
}

#endif
