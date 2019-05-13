//---------------------------------Spheral++----------------------------------//
// ConnectivityMap
//
// Stores the full set of significant neighbors for a set of NodeLists.
//
// Created by J. Michael Owen, Sun Oct 30 15:36:33 PST 2005
//----------------------------------------------------------------------------//
#ifndef _Spheral_NeighborSpace_ConnectivityMap_hh_
#define _Spheral_NeighborSpace_ConnectivityMap_hh_

#include "Utilities/KeyTraits.hh"
#include "Field/FieldList.hh"

#include <vector>
#include <map>

namespace Spheral {

template<typename Dimension> class NodeList;
template<typename Dimension> class FluidNodeList;
template<typename Dimension> class Boundary;
template<typename Dimension, typename DataType> class FieldList;

template<typename Dimension>
class ConnectivityMap {
public:
  //--------------------------- Public Interface ---------------------------//
  typedef std::vector<int>::const_iterator const_iterator;

  // Constructors, destructor.
  ConnectivityMap();
  ~ConnectivityMap();

  template<typename NodeListIterator>
  ConnectivityMap(const NodeListIterator& begin,
                  const NodeListIterator& end,
                  const bool buildGhostConnectivity,
                  const bool buildOverlapConnectivity);

  // Rebuild for a given set of NodeLists.
  template<typename NodeListIterator>
  void rebuild(const NodeListIterator& begin, 
               const NodeListIterator& end, 
               const bool computeGhostConnectivity,
               const bool buildOverlapConnectivity);

  // Patch the connectivity information:
  // flags   -- (0,1): 0 => node deleted, 1 => node preserved
  // old2new -- maps old -> new node indices.
  void patchConnectivity(const FieldList<Dimension, int>& flags,
                         const FieldList<Dimension, int>& old2new);

  // Remove connectivity between neighbors.
  // Note this method assumes neighbor info is symmetric, and removes the pair connectivity for each
  // member of a pair (maintaining symmetry).
  void removeConnectivity(const FieldList<Dimension, std::vector<std::vector<int>>>& neighborsToCut);

  // Remove non-surface multimaterial toplogical connections
  void removeMultimaterialConnectivity(const FieldList<Dimension, int>& surfacePoint);

  // Are we computing neighbors for ghosts?
  bool buildGhostConnectivity() const;

  // Do we compute overlap connectivity?
  bool buildOverlapConnectivity() const;

  // Get the set of NodeLists.
  const std::vector<const NodeList<Dimension>*>& nodeLists() const;

  //............................................................................
  // Get the set of neighbors for the given (internal!) node in the given NodeList.
  const std::vector< std::vector<int> >&
  connectivityForNode(const NodeList<Dimension>* nodeListPtr,
                      const int nodeID) const;

  // Same as above, just referencing the NodeList by an integer index.
  const std::vector< std::vector<int> >&
  connectivityForNode(const int nodeListID,
                      const int nodeID) const;

  //............................................................................
  // Note the following two methods return the points we have neighbors in common with,
  // not the common neighbors.  You need to query ConnectivityMap::connectivityIntersectionForNodes
  // to get the overlapping set of points.
  // Get the set of neighbors we have overlap with (common neighbors).
  const std::vector< std::vector<int> >&
  overlapConnectivityForNode(const NodeList<Dimension>* nodeListPtr,
                             const int nodeID) const;

  // Same as above, just referencing the NodeList by an integer index.
  const std::vector< std::vector<int> >&
  overlapConnectivityForNode(const int nodeListID,
                             const int nodeID) const;


  //............................................................................
  // Compute the common neighbors for a pair of nodes.  Note this method 
  // returns by value since this information is not stored by ConnectivityMap.
  std::vector< std::vector<int> >
  connectivityIntersectionForNodes(const int nodeListi, const int i,
                                   const int nodeListj, const int j) const;

  // Compute the union of neighbors for a pair of nodes.  Note this method 
  // returns by value since this information is not stored by ConnectivityMap.
  std::vector< std::vector<int> >
  connectivityUnionForNodes(const int nodeListi, const int i,
                            const int nodeListj, const int j) const;

  // Compute the number of neighbors for the given node.
  int numNeighborsForNode(const NodeList<Dimension>* nodeListPtr,
                          const int nodeID) const;

  int numNeighborsForNode(const int nodeListID,
                          const int nodeID) const;

  // Return the connectivity in terms of global node IDs.
  std::map<int, std::vector<int> > globalConnectivity(std::vector<Boundary<Dimension>*>& boundaries) const;

  // Function to determine if given node information (i and j), if the 
  // pair should already have been calculated by iterating over each
  // others neighbors.
  bool calculatePairInteraction(const int nodeListi, const int i, 
                                const int nodeListj, const int j,
                                const int firstGhostNodej) const;

  // Provide iterator interface for walking the nodes in a NodeList
  // in order to maintain domain decomposition independence when 
  // desired.
  const_iterator begin(const int nodeList) const;
  const_iterator end(const int nodeList) const;

  // Return the number of nodes we should walk for the given NodeList.
  int numNodes(const int nodeList) const;

  // The ith node (ordered) in the given NodeList.
  int ithNode(const int nodeList, const int index) const;

  // Get the ith NodeList or FluidNodeList.
  const NodeList<Dimension>& nodeList(const int index) const;

  // Return which NodeList index in order the given one would be in our connectivity.
  unsigned nodeListIndex(const NodeList<Dimension>* nodeListPtr) const;

  // Check that the internal data structure is valid.
  bool valid() const;

private:
  //--------------------------- Private Interface ---------------------------//
  // The set of NodeLists.
  std::vector<const NodeList<Dimension>*> mNodeLists;

  // Are we building ghost and/or overlap connectivity?
  bool mBuildGhostConnectivity, mBuildOverlapConnectivity;

  // The full connectivity map.  This might be quite large!
  // [offset[NodeList] + nodeID] [NodeListID] [neighborIndex]
  typedef std::vector<std::vector<std::vector<int>>> ConnectivityStorageType;
  std::vector<int> mOffsets;
  ConnectivityStorageType mConnectivity;

  // Same for overlap connectivity.
  ConnectivityStorageType mOverlapConnectivity;

  // The set of node indices per Nodelist in order for traversal.
  std::vector<std::vector<int>> mNodeTraversalIndices;

  // The set of keys we may compute for each node.
  typedef typename KeyTraits::Key Key;
  FieldList<Dimension, Key> mKeys;

  // Internal method to fill in the connectivity, once the set of NodeLists 
  // is determined.
  void computeConnectivity();

  // No default constructor, copying, or assignment.
  ConnectivityMap(const ConnectivityMap&);
  ConnectivityMap& operator=(const ConnectivityMap&);
};

}

#include "ConnectivityMapInline.hh"

#endif

