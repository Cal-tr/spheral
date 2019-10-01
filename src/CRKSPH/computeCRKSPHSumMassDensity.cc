//------------------------------------------------------------------------------
// Compute the CRKSPH mass density summation.
//------------------------------------------------------------------------------
#include "computeCRKSPHSumMassDensity.hh"
#include "computeCRKSPHMoments.hh"
#include "computeCRKSPHCorrections.hh"
#include "CRKSPHCorrectionParams.hh"
#include "NodeList/NodeList.hh"
#include "Hydro/HydroFieldNames.hh"

using std::vector;
using std::min;
using std::max;
using std::abs;

namespace Spheral {

template<typename Dimension>
void
computeCRKSPHSumMassDensity(const ConnectivityMap<Dimension>& connectivityMap,
                            const TableKernel<Dimension>& W,
                            const FieldList<Dimension, typename Dimension::Vector>& position,
                            const FieldList<Dimension, typename Dimension::Scalar>& mass,
                            const FieldList<Dimension, typename Dimension::Scalar>& vol,
                            const FieldList<Dimension, typename Dimension::SymTensor>& H,
                            FieldList<Dimension, typename Dimension::Scalar>& massDensity) {

  // Pre-conditions.
  const size_t numNodeLists = massDensity.size();
  REQUIRE(position.size() == numNodeLists);
  REQUIRE(mass.size() == numNodeLists);
  REQUIRE(vol.size() == numNodeLists);
  REQUIRE(H.size() == numNodeLists);

  typedef typename Dimension::Scalar Scalar;
  typedef typename Dimension::Vector Vector;
  typedef typename Dimension::Tensor Tensor;
  typedef typename Dimension::SymTensor SymTensor;
  typedef typename Dimension::ThirdRankTensor ThirdRankTensor;
  typedef typename Dimension::FourthRankTensor FourthRankTensor;
  typedef typename Dimension::FifthRankTensor FifthRankTensor;

  // Initialize stuff to sum
  massDensity = 0.0;
  FieldList<Dimension, Scalar> wsum(FieldStorageType::CopyFields), vol1(FieldStorageType::CopyFields);
  for (size_t nodeListi = 0; nodeListi != numNodeLists; ++nodeListi) {
    wsum.appendNewField("weight sum", position[nodeListi]->nodeList(), 0.0);
    vol1.appendNewField("sampled volume", position[nodeListi]->nodeList(), 0.0);
  }

  // Some useful variables.
  const auto W0 = W.kernelValue(0.0, 1.0);

  // The set of interacting node pairs.
  const auto& pairs = connectivityMap.nodePairList();
  const auto  npairs = pairs.size();

#pragma omp parallel
  {
    // Some scratch variables.
    int i, j, nodeListi, nodeListj;
    Scalar Wi, Wj;
    Vector rij, etai, etaj;
    Vector Bi = Vector::zero, Bj = Vector::zero;
    Tensor Ci = Tensor::zero, Cj = Tensor::zero;

    auto massDensity_thread = massDensity.threadCopy();
    auto wsum_thread = wsum.threadCopy();
    auto vol1_thread = vol1.threadCopy();

#pragma omp for
    for (auto k = 0; k < npairs; ++k) {
      i = pairs[k].i_node;
      j = pairs[k].j_node;
      nodeListi = pairs[k].i_list;
      nodeListj = pairs[k].j_list;

      // Get the state for node i.
      const auto& ri = position(nodeListi, i);
      const auto  mi = mass(nodeListi, i);
      const auto  Vi = vol(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
      const auto  Hdeti = Hi.Determinant();

      // State for node j.
      const auto& rj = position(nodeListi, j);
      const auto  mj = mass(nodeListi, j);
      const auto  Vj = vol(nodeListi, j);
      const auto  rhoj = massDensity(nodeListi, j);
      const auto& Hj = H(nodeListi, j);
      const auto  Hdetj = Hj.Determinant();

      // Kernel weighting and gradient.
      rij = ri - rj;
      etai = Hi*rij;
      etaj = Hj*rij;
      Wi = W.kernelValue(etai.magnitude(), Hdeti);
      Wj = W.kernelValue(etaj.magnitude(), Hdetj);

      // Sum the pair-wise contributions.
      wsum_thread(nodeListi, i) += Vj*Wj;
      wsum_thread(nodeListi, j) += Vi*Wi;
      massDensity_thread(nodeListi, i) += mj * Vj*Wj;
      massDensity_thread(nodeListi, j) += mi * Vi*Wi;
      vol1_thread(nodeListi, i) += Vj * Vj*Wj;
      vol1_thread(nodeListi, j) += Vi * Vi*Wi;
    }

#pragma omp critical
    {
      for (nodeListi = 0; nodeListi < numNodeLists; ++nodeListi) {
        const auto ni = massDensity[nodeListi]->numInternalElements();
        for (auto i = 0; i < ni; ++i) {
          massDensity(nodeListi, i) += massDensity_thread(nodeListi, i);
          wsum(nodeListi, i) += wsum_thread(nodeListi, i);
          vol1(nodeListi, i) += vol1_thread(nodeListi, i);
        }
      }
    } // OMP critical
  }   // OMP parallel
  
  // The self contribution.
  for (auto nodeListi = 0; nodeListi < numNodeLists; ++nodeListi) {
    const auto& nodeList = dynamic_cast<const FluidNodeList<Dimension>&>(massDensity[0]->nodeList());
    const auto ni = nodeList.numInternalNodes();
    const auto rhoMin = nodeList.rhoMin();
    const auto rhoMax = nodeList.rhoMax();

#pragma omp parallel for
    for (auto i = 0; i < ni; ++i) {

      // Get the state for node i.
      const auto  mi = mass(nodeListi, i);
      const auto  Vi = vol(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
      const auto  Hdeti = Hi.Determinant();

      wsum(nodeListi, i) += Vi*Hdeti*W0;
      CHECK(wsum(nodeListi, i) > 0.0);
      vol1(nodeListi, i) = (vol1(nodeListi, i) + Vi*Vi*Hdeti*W0)/wsum(nodeListi, i);
      massDensity(nodeListi, i) = max(max(rhoMin, 0.1*mi*H(nodeListi, i).Determinant()),
                                      min(rhoMax,
                                          (massDensity(nodeListi, i) + mi*Vi*Hdeti*W0)/
                                          (wsum(nodeListi, i)*vol1(nodeListi, i))));
      ENSURE(vol1(nodeListi, i) > 0.0);
      ENSURE(massDensity(nodeListi, i) > 0.0);
    }
  }
}

}
