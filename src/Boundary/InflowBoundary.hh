//---------------------------------Spheral++----------------------------------//
// InflowBoundary -- creates inflow ghost images, which become internal nodes
// as they cross the specified boundary plane.
//
// Created by JMO, Tue Oct 15 11:23:09 PDT 2019
//
// Modified by:
//----------------------------------------------------------------------------//
#ifndef __Spheral_InflowBoundary__
#define __Spheral_InflowBoundary__

#include "Boundary.hh"
#include "Physics/Physics.hh"
#include "Geometry/GeomPlane.hh"
#include "NodeList/NodeList.hh"
#include "DataBase/StateBase.hh" // For constructing Field keys.

namespace Spheral {

// Forward declarations.
template<typename Dimension> class NodeList;
template<typename Dimension> class FieldBase;
template<typename Dimension, typename DataType> class Field;
template<typename Dimension, typename DataType> class FieldList;
template<typename Dimension> class DataBase;

template<typename Dimension>
class InflowBoundary: public Boundary<Dimension>, public Physics<Dimension> {

public:
  //--------------------------- Public Interface ---------------------------//
  typedef typename Dimension::Scalar Scalar;
  typedef typename Dimension::Vector Vector;
  typedef typename Dimension::Tensor Tensor;
  typedef typename Dimension::SymTensor SymTensor;
  typedef typename Dimension::ThirdRankTensor ThirdRankTensor;
  typedef typename Dimension::FacetedVolume FacetedVolume;
  typedef typename StateBase<Dimension>::KeyType KeyType;
  typedef typename Physics<Dimension>::TimeStepType TimeStepType;

  // Constructors and destructors.
  InflowBoundary(NodeList<Dimension>& nodeList,
                 const GeomPlane<Dimension>& plane);
  virtual ~InflowBoundary();

  //**********************************************************************
  // Boundary condition methods:
  // Use the given NodeList's neighbor object to select the ghost nodes.
  virtual void setGhostNodes(NodeList<Dimension>& nodeList) override;

  // For the computed set of ghost nodes, set the positions and H's.
  virtual void updateGhostNodes(NodeList<Dimension>& nodeList) override;

  // Apply the boundary condition to the ghost node values in the given Field.
  virtual void applyGhostBoundary(Field<Dimension, int>& field) const override;
  virtual void applyGhostBoundary(Field<Dimension, Scalar>& field) const override;
  virtual void applyGhostBoundary(Field<Dimension, Vector>& field) const override;
  virtual void applyGhostBoundary(Field<Dimension, Tensor>& field) const override;
  virtual void applyGhostBoundary(Field<Dimension, SymTensor>& field) const override;
  virtual void applyGhostBoundary(Field<Dimension, ThirdRankTensor>& field) const override;
  virtual void applyGhostBoundary(Field<Dimension, FacetedVolume>& field) const override;

  // Find any internal nodes that are in violation of this Boundary.
  virtual void setViolationNodes(NodeList<Dimension>& nodeList) override;

  // For the computed set of nodes in violation of the boundary, bring them
  // back into compliance (for the positions and H's.)
  virtual void updateViolationNodes(NodeList<Dimension>& nodeList) override;

  // Apply the boundary condition to the violation node values in the given Field.
  virtual void enforceBoundary(Field<Dimension, int>& field) const override;
  virtual void enforceBoundary(Field<Dimension, Scalar>& field) const override;
  virtual void enforceBoundary(Field<Dimension, Vector>& field) const override;
  virtual void enforceBoundary(Field<Dimension, Tensor>& field) const override;
  virtual void enforceBoundary(Field<Dimension, SymTensor>& field) const override;
  virtual void enforceBoundary(Field<Dimension, ThirdRankTensor>& field) const override;
  virtual void enforceBoundary(Field<Dimension, FacetedVolume>& field) const override;

  virtual void applyGhostBoundary(Field<Dimension, std::vector<Scalar>>& field) const override;
  virtual void applyGhostBoundary(Field<Dimension, std::vector<Vector>>& field) const override;

  // After physics have been initialized we take a snapshot of the node state.
  virtual void initializeProblemStartup() override;
  //**********************************************************************

  //**********************************************************************
  // Physics methods:
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

  // Register the state you want carried around (and potentially evolved), as
  // well as the policies for such evolution.
  virtual void registerState(DataBase<Dimension>& dataBase,
                             State<Dimension>& state) override;

  // Register the derivatives/change fields for updating state.
  virtual void registerDerivatives(DataBase<Dimension>& dataBase,
                                   StateDerivatives<Dimension>& derivs) override;

  // Packages might want a hook to do some post-step finalizations.
  // Really we should rename this post-step finalize.
  virtual void finalize(const Scalar time, 
                        const Scalar dt,
                        DataBase<Dimension>& dataBase, 
                        State<Dimension>& state,
                        StateDerivatives<Dimension>& derivs) override;
  //**********************************************************************

  // Accessor methods.
  int numInflowNodes() const;
  const NodeList<Dimension>& nodeList() const;
  const GeomPlane<Dimension>& plane() const;

  // Get the stored data for generating ghost nodes.
  template<typename DataType> std::vector<DataType>& storedValues(const std::string fieldName, const DataType& dummy);
  template<typename DataType> std::vector<DataType>& storedValues(const Field<Dimension, DataType>& field);

  //****************************************************************************
  // Methods required for restarting.
  virtual std::string label() const override;
  virtual void dumpState(FileIO& file, const std::string& pathName) const;
  virtual void restoreState(const FileIO& file, const std::string& pathName);
  //****************************************************************************

private:
  //--------------------------- Private Interface ---------------------------//
  GeomPlane<Dimension> mPlane;
  NodeList<Dimension>* mNodeListPtr;
  int mBoundaryCount, mNumInflowNodes;
  Scalar mInflowVelocity, mXmin, mDT;
  bool mActive;

  typedef std::map<KeyType, std::vector<int>> IntStorageType;
  typedef std::map<KeyType, std::vector<Scalar>> ScalarStorageType;
  typedef std::map<KeyType, std::vector<Vector>> VectorStorageType;
  typedef std::map<KeyType, std::vector<Tensor>> TensorStorageType;
  typedef std::map<KeyType, std::vector<SymTensor>> SymTensorStorageType;
  typedef std::map<KeyType, std::vector<ThirdRankTensor>> ThirdRankTensorStorageType;
  typedef std::map<KeyType, std::vector<FacetedVolume>> FacetedVolumeStorageType;
  typedef std::map<KeyType, std::vector<std::vector<Scalar>>> VectorScalarStorageType;
  typedef std::map<KeyType, std::vector<std::vector<Vector>>> VectorVectorStorageType;

  IntStorageType mIntValues;
  ScalarStorageType mScalarValues;
  VectorStorageType mVectorValues;
  TensorStorageType mTensorValues;
  SymTensorStorageType mSymTensorValues;
  ThirdRankTensorStorageType mThirdRankTensorValues;
  FacetedVolumeStorageType mFacetedVolumeValues;
  VectorScalarStorageType mVectorScalarValues;
  VectorVectorStorageType mVectorVectorValues;

  // The restart registration.
  RestartRegistrationType mRestart;

  // Internal trait methods to help with looking up the correct storage.
  IntStorageType&             storageForType(const int& dummy)                 { return mIntValues; }
  ScalarStorageType&          storageForType(const Scalar& dummy)              { return mScalarValues; }
  VectorStorageType&          storageForType(const Vector& dummy)              { return mVectorValues; }
  TensorStorageType&          storageForType(const Tensor& dummy)              { return mTensorValues; }
  SymTensorStorageType&       storageForType(const SymTensor& dummy)           { return mSymTensorValues; }
  ThirdRankTensorStorageType& storageForType(const ThirdRankTensor& dummy)     { return mThirdRankTensorValues; }
  FacetedVolumeStorageType&   storageForType(const FacetedVolume& dummy)       { return mFacetedVolumeValues; }
  VectorScalarStorageType&    storageForType(const std::vector<Scalar>& dummy) { return mVectorScalarValues; }
  VectorVectorStorageType&    storageForType(const std::vector<Vector>& dummy) { return mVectorVectorValues; }

  // No default or copy constructors.
  InflowBoundary();
  InflowBoundary(InflowBoundary&);
};

}

#include "InflowBoundaryInline.hh"

#else

// Forward declaration.
namespace Spheral {
  template<typename Dimension> class InflowBoundary;
}

#endif
