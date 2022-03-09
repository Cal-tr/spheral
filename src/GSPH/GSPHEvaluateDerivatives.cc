namespace Spheral {
//------------------------------------------------------------------------------
// Determine the principle derivatives.
//------------------------------------------------------------------------------
template<typename Dimension>
void
GSPHHydroBase<Dimension>::
evaluateDerivatives(const typename Dimension::Scalar time,
                    const typename Dimension::Scalar dt,
                    const DataBase<Dimension>& dataBase,
                    const State<Dimension>& state,
                          StateDerivatives<Dimension>& derivatives) const {
  TIME_GSPHevalDerivs.start();

  const auto& riemannSolver = this->riemannSolver();

  const auto& smoothingScale = this->smoothingScaleMethod();
  
  // A few useful constants we'll use in the following loop.
  const auto tiny = std::numeric_limits<Scalar>::epsilon();
  const auto xsph = this->XSPH();
  const auto epsTensile = this->epsilonTensile();
  //const auto epsDiffusionCoeff = this->specificThermalEnergyDiffusionCoefficient();
  const auto compatibleEnergy = this->compatibleEnergyEvolution();
  const auto totalEnergy = this->evolveTotalEnergy();
  const auto gradType = this->gradientType();
  //const auto correctVelocityGradient = this->correctVelocityGradient();

  // The connectivity.
  const auto& connectivityMap = dataBase.connectivityMap();
  const auto& nodeLists = connectivityMap.nodeLists();
  const auto& pairs = connectivityMap.nodePairList();
  const auto  npairs = pairs.size();
  const auto  numNodeLists = nodeLists.size();
  const auto  nPerh = nodeLists[0]->nodesPerSmoothingScale();

  // kernel
  const auto& W = this->kernel();
  const auto  WnPerh = W(1.0/nPerh, 1.0);
  const auto  W0 = W(0.0, 1.0);

  // Get the state and derivative FieldLists.
  // State FieldLists.
  const auto mass = state.fields(HydroFieldNames::mass, 0.0);
  const auto position = state.fields(HydroFieldNames::position, Vector::zero);
  const auto velocity = state.fields(HydroFieldNames::velocity, Vector::zero);
  const auto massDensity = state.fields(HydroFieldNames::massDensity, 0.0);
  const auto volume = state.fields(HydroFieldNames::volume, 0.0);
  const auto specificThermalEnergy = state.fields(HydroFieldNames::specificThermalEnergy, 0.0);
  const auto H = state.fields(HydroFieldNames::H, SymTensor::zero);
  const auto pressure = state.fields(HydroFieldNames::pressure, 0.0);
  const auto soundSpeed = state.fields(HydroFieldNames::soundSpeed, 0.0);
  const auto riemannDpDx = state.fields(GSPHFieldNames::RiemannPressureGradient,Vector::zero);
  const auto riemannDvDx = state.fields(GSPHFieldNames::RiemannVelocityGradient,Tensor::zero);
  
  CHECK(mass.size() == numNodeLists);
  CHECK(position.size() == numNodeLists);
  CHECK(velocity.size() == numNodeLists);
  CHECK(massDensity.size() == numNodeLists);
  CHECK(volume.size() == numNodeLists);
  CHECK(specificThermalEnergy.size() == numNodeLists);
  CHECK(H.size() == numNodeLists);
  CHECK(pressure.size() == numNodeLists);
  CHECK(soundSpeed.size() == numNodeLists);

  // Derivative FieldLists.
  const auto  M = derivatives.fields(HydroFieldNames::M_SPHCorrection, Tensor::zero);
  auto  normalization = derivatives.fields(HydroFieldNames::normalization, 0.0);
  auto  DxDt = derivatives.fields(IncrementFieldList<Dimension, Vector>::prefix() + HydroFieldNames::position, Vector::zero);
  auto  DrhoDt = derivatives.fields(IncrementFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::massDensity, 0.0);
  auto  DvDt = derivatives.fields(HydroFieldNames::hydroAcceleration, Vector::zero);
  auto  DepsDt = derivatives.fields(IncrementFieldList<Dimension, Scalar>::prefix() + HydroFieldNames::specificThermalEnergy, 0.0);
  auto  DvDx = derivatives.fields(HydroFieldNames::velocityGradient, Tensor::zero);
  //auto  localDvDx = derivatives.fields(HydroFieldNames::internalVelocityGradient, Tensor::zero);
  //auto  localM = derivatives.fields("local"+HydroFieldNames::M_SPHCorrection, Tensor::zero);
  auto  DHDt = derivatives.fields(IncrementFieldList<Dimension, SymTensor>::prefix() + HydroFieldNames::H, SymTensor::zero);
  auto  Hideal = derivatives.fields(ReplaceBoundedFieldList<Dimension, SymTensor>::prefix() + HydroFieldNames::H, SymTensor::zero);
  auto& pairAccelerations = derivatives.getAny(HydroFieldNames::pairAccelerations, vector<Vector>());
  auto& pairDepsDt = derivatives.getAny(HydroFieldNames::pairWork, vector<Scalar>());
  //auto  XSPHWeightSum = derivatives.fields(HydroFieldNames::XSPHWeightSum, 0.0);
  auto  XSPHDeltaV = derivatives.fields(HydroFieldNames::XSPHDeltaV, Vector::zero);
  auto  weightedNeighborSum = derivatives.fields(HydroFieldNames::weightedNeighborSum, 0.0);
  auto  massSecondMoment = derivatives.fields(HydroFieldNames::massSecondMoment, SymTensor::zero);
  //auto  DpDx = derivatives.fields(GSPHFieldNames::pressureGradient,Vector::zero);
  auto  newRiemannDpDx = derivatives.fields(ReplaceFieldList<Dimension, Scalar>::prefix() + GSPHFieldNames::RiemannPressureGradient,Vector::zero);
  auto  newRiemannDvDx = derivatives.fields(ReplaceFieldList<Dimension, Scalar>::prefix() + GSPHFieldNames::RiemannVelocityGradient,Tensor::zero);
  
  CHECK(normalization.size() == numNodeLists);
  CHECK(DxDt.size() == numNodeLists);
  CHECK(DrhoDt.size() == numNodeLists);
  CHECK(DvDt.size() == numNodeLists);
  CHECK(DepsDt.size() == numNodeLists);
  CHECK(DvDx.size() == numNodeLists);
  //CHECK(localDvDx.size() == numNodeLists);
  //CHECK(localM.size() == numNodeLists);
  CHECK(DHDt.size() == numNodeLists);
  CHECK(Hideal.size() == numNodeLists);
  //CHECK(XSPHWeightSum.size() == numNodeLists);
  CHECK(XSPHDeltaV.size() == numNodeLists);
  CHECK(weightedNeighborSum.size() == numNodeLists);
  CHECK(massSecondMoment.size() == numNodeLists);
  CHECK(DpDx.size() == numNodeLists);

  if (compatibleEnergy){
    pairAccelerations.resize(npairs);
    pairDepsDt.resize(2*npairs);
  }

  this->computeMCorrection(time,dt,dataBase,state,derivatives);

  // Walk all the interacting pairs.
#pragma omp parallel
  {
    // Thread private scratch variables
    int i, j, nodeListi, nodeListj;
    Scalar psii,psij, Wi, gWi, Wj, gWj, Pstar, rhostari, rhostarj;
    Vector gradPsii, gradPsij, Ai, Aj, vstar;

    typename SpheralThreads<Dimension>::FieldListStack threadStack;
    auto DvDt_thread = DvDt.threadCopy(threadStack);
    auto weightedNeighborSum_thread = weightedNeighborSum.threadCopy(threadStack);
    auto massSecondMoment_thread = massSecondMoment.threadCopy(threadStack);
    auto DepsDt_thread = DepsDt.threadCopy(threadStack);
    //auto DrhoDt_thread = DrhoDt.threadCopy(threadStack);
    auto DvDx_thread = DvDx.threadCopy(threadStack);
    //auto DpDx_thread = DpDx.threadCopy(threadStack);
    auto newRiemannDpDx_thread = newRiemannDpDx.threadCopy(threadStack);
    auto newRiemannDvDx_thread = newRiemannDvDx.threadCopy(threadStack);
    //auto localDvDx_thread = localDvDx.threadCopy(threadStack);
    //auto localM_thread = localDvDx.threadCopy(threadStack);
    //auto XSPHWeightSum_thread = XSPHWeightSum.threadCopy(threadStack);
    auto XSPHDeltaV_thread =  XSPHDeltaV.threadCopy(threadStack);
    auto normalization_thread = normalization.threadCopy(threadStack);
    
#pragma omp for
    for (auto kk = 0u; kk < npairs; ++kk) {
      i = pairs[kk].i_node;
      j = pairs[kk].j_node;
      nodeListi = pairs[kk].i_list;
      nodeListj = pairs[kk].j_list;

      // Get the state for node i.
      const auto& riemannDpDxi = riemannDpDx(nodeListi, i);
      const auto& riemannDvDxi = riemannDvDx(nodeListi, i);
      const auto& ri = position(nodeListi, i);
      const auto& mi = mass(nodeListi, i);
      const auto& vi = velocity(nodeListi, i);
      const auto& rhoi = massDensity(nodeListi, i);
      const auto& voli = volume(nodeListi, i);
      //const auto& epsi = specificThermalEnergy(nodeListi, i);
      const auto& Pi = pressure(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
      const auto& ci = soundSpeed(nodeListi, i);
      const auto  Hdeti = Hi.Determinant();
      CHECK(mi > 0.0);
      CHECK(rhoi > 0.0);
      CHECK(Hdeti > 0.0);

      auto& normi = normalization_thread(nodeListi,i);
      auto& DepsDti = DepsDt_thread(nodeListi, i);
      auto& DvDti = DvDt_thread(nodeListi, i);
      //auto& DpDxi = DpDx_thread(nodeListi,i);
      auto& newRiemannDpDxi = newRiemannDpDx_thread(nodeListi,i);
      auto& newRiemannDvDxi = newRiemannDvDx_thread(nodeListi,i);
      auto& DvDxi = DvDx_thread(nodeListi, i);
      //auto& localDvDxi = localDvDx_thread(nodeListi, i);
      //auto& localMi = localM_thread(nodeListi, i);
      auto& weightedNeighborSumi = weightedNeighborSum_thread(nodeListi, i);
      auto& massSecondMomenti = massSecondMoment_thread(nodeListi, i);
      //auto& XSPHWeightSumi = XSPHWeightSum_thread(nodeListi,i);
      auto& XSPHDeltaVi = XSPHDeltaV_thread(nodeListi,i);
      const auto& Mi = M(nodeListi,i);


      // Get the state for node j
      const auto& riemannDpDxj = riemannDpDx(nodeListj, j);
      const auto& riemannDvDxj = riemannDvDx(nodeListj, j);
      const auto& rj = position(nodeListj, j);
      const auto& mj = mass(nodeListj, j);
      const auto& vj = velocity(nodeListj, j);
      const auto& rhoj = massDensity(nodeListj, j);
      const auto& volj = volume(nodeListj, j);
      //const auto& epsj = specificThermalEnergy(nodeListj, j);
      const auto& Pj = pressure(nodeListj, j);
      const auto& Hj = H(nodeListj, j);
      const auto& cj = soundSpeed(nodeListj, j);
      const auto  Hdetj = Hj.Determinant();
      CHECK(mj > 0.0);
      CHECK(rhoj > 0.0);
      CHECK(Hdetj > 0.0);

      auto& normj = normalization_thread(nodeListj,j);
      auto& DvDtj = DvDt_thread(nodeListj, j);
      auto& DepsDtj = DepsDt_thread(nodeListj, j);
      auto& newRiemannDpDxj = newRiemannDpDx_thread(nodeListj,j);
      auto& newRiemannDvDxj = newRiemannDvDx_thread(nodeListj,j);
      //auto& DpDxj = DpDx_thread(nodeListj,j);
      //auto& DpDxRawj = DpDxRaw_thread(nodeListj,j);
      //auto& DvDxRawj = DvDxRaw_thread(nodeListj,j);
      auto& DvDxj = DvDx_thread(nodeListj, j);
      //auto& localDvDxj = localDvDx_thread(nodeListj, j);
      //auto& localMj = localM_thread(nodeListj, j);
      auto& weightedNeighborSumj = weightedNeighborSum_thread(nodeListj, j);
      auto& massSecondMomentj = massSecondMoment_thread(nodeListj, j);
      //auto& XSPHWeightSumj = XSPHWeightSum_thread(nodeListj,j);
      auto& XSPHDeltaVj = XSPHDeltaV_thread(nodeListj,j);
      const auto& Mj = M(nodeListj,j);

      // Flag if this is a contiguous material pair or not.
      //const bool sameMatij =  (nodeListi == nodeListj);

      // Node displacement.
      const auto rij = ri - rj;
      const auto rhatij =rij.unitVector();
      const auto vij = vi - vj;
      const auto etai = Hi*rij;
      const auto etaj = Hj*rij;
      const auto etaMagi = etai.magnitude();
      const auto etaMagj = etaj.magnitude();
      CHECK(etaMagi >= 0.0);
      CHECK(etaMagj >= 0.0);


      // Symmetrized kernel weight and gradient.
      std::tie(Wi, gWi) = W.kernelAndGradValue(etaMagi, Hdeti);
      const auto Hetai = Hi*etai.unitVector();
      const auto gradWi = gWi*Hetai;

      std::tie(Wj, gWj) = W.kernelAndGradValue(etaMagj, Hdetj);
      const auto Hetaj = Hj*etaj.unitVector();
      const auto gradWj = gWj*Hetaj;

      // Zero'th and second moment of the node distribution -- used for the
      // ideal H calculation.
      const auto rij2 = rij.magnitude2();
      const auto thpt = rij.selfdyad()*safeInvVar(rij2*rij2*rij2);
      weightedNeighborSumi += std::abs(gWi);
      weightedNeighborSumj += std::abs(gWj);
      massSecondMomenti += gradWi.magnitude2()*thpt;
      massSecondMomentj += gradWj.magnitude2()*thpt;

      // Determine an effective pressure including a term to fight the tensile instability.
      //const auto fij = epsTensile*pow(Wi/(Hdeti*WnPerh), nTensile);
      const auto fij = epsTensile*FastMath::pow4(Wi/(Hdeti*WnPerh));
      const auto Ri = fij*(Pi < 0.0 ? -Pi : 0.0);
      const auto Rj = fij*(Pj < 0.0 ? -Pj : 0.0);
      const auto Peffi = Pi + Ri;
      const auto Peffj = Pj + Rj;

      // we'll clean this up when we have a gradient 
      // implementation we're in love with
      auto gradPi = riemannDpDxi;
      auto gradPj = riemannDpDxj;
      auto gradVi = riemannDvDxi;
      auto gradVj = riemannDvDxj;
      if (gradType==GradientType::SPHSameTimeGradient){
        gradPi = newRiemannDpDxi;
        gradPj = newRiemannDpDxj;
        gradVi = newRiemannDvDxi;
        gradVj = newRiemannDvDxj;
      }
      riemannSolver.interfaceState(i,            j, 
                                   nodeListi,    nodeListj, 
                                   ri,           rj, 
                                   rhoi,         rhoj, 
                                   ci,           cj, 
                                   Peffi,        Peffj, 
                                   vi,           vj, 
                                   gradPi,       gradPj, 
                                   gradVi,       gradVj, 
                                   Pstar,
                                   vstar,
                                   rhostari,
                                   rhostarj);

      // get our basis function and interface area vectors
      //--------------------------------------------------------
      psii = volj*Wi;
      psij = voli*Wj;
      gradPsii = volj * Mi.Transpose()*gradWi;
      gradPsij = voli * Mj.Transpose()*gradWj;

      const auto Ai = voli*gradPsii;
      const auto Aj = volj*gradPsij;

      // acceleration
      //------------------------------------------------------
      const auto deltaDvDt = Pstar*(Ai+Aj);
      DvDti -= deltaDvDt;
      DvDtj += deltaDvDt;

      // energy
      //------------------------------------------------------
      const auto deltaDepsDti = 2.0*Pstar*Ai.dot(vi-vstar);
      const auto deltaDepsDtj = 2.0*Pstar*Aj.dot(vstar-vj);

      DepsDti += deltaDepsDti;
      DepsDtj += deltaDepsDtj;
     
      if(compatibleEnergy){
        const auto invmij = 1.0/(mi*mj);
        pairAccelerations[kk] = deltaDvDt*invmij; 
        pairDepsDt[2*kk]   = deltaDepsDti*invmij; 
        pairDepsDt[2*kk+1] = deltaDepsDtj*invmij; 
      }

      // gradients
      //------------------------------------------------------
      const auto deltaDvDxi = 2.0*(vi-vstar).dyad(gradPsii);
      const auto deltaDvDxj = 2.0*(vstar-vj).dyad(gradPsij);

      // based on riemann soln
      DvDxi -= deltaDvDxi;
      DvDxj -= deltaDvDxj;

      // if(sameMatij){
      //   localMi -= rij.dyad(gradPsii);
      //   localMj -= rij.dyad(gradPsij);
      //   localDvDxi -= deltaDvDxi;
      //   localDvDxj -= deltaDvDxj;
      // }

      // DpDxi -= 2.0*(Peffi-Pstar)*gradPsii;
      // DpDxj -= 2.0*(Pstar-Peffj)*gradPsij;

      // // based on nodal values
      // DvDxRawi -= (vi-vj).dyad(gradPsii);
      // DvDxRawj -= (vi-vj).dyad(gradPsij);

      // DpDxRawi -= (Pi-Pj)*gradPsii;
      // DpDxRawj -= (Pi-Pj)*gradPsij;
            // while we figure out what we want ...
      switch(gradType){ 
        case GradientType::RiemannGradient: // default grad based on riemann soln
          newRiemannDvDxi -= deltaDvDxi;
          newRiemannDvDxj -= deltaDvDxj;
          newRiemannDpDxi -= 2.0*(Pi-Pstar)*gradPsii;
          newRiemannDpDxj -= 2.0*(Pstar-Pj)*gradPsij;
          break;
        case GradientType::HydroAccelerationGradient: // based on hydro accel for DpDx
          newRiemannDvDxi -= deltaDvDxi;
          newRiemannDvDxj -= deltaDvDxj;
          newRiemannDpDxi += rhoi/mi*deltaDvDt;
          newRiemannDpDxj -= rhoj/mj*deltaDvDt;
          break;
        case GradientType::SPHGradient: // raw gradients
          newRiemannDvDxi -= (vi-vj).dyad(gradPsii);
          newRiemannDvDxj -= (vi-vj).dyad(gradPsij);
          newRiemannDpDxi -= (Pi-Pj)*gradPsii;
          newRiemannDpDxj -= (Pi-Pj)*gradPsij;
          break;
        case GradientType::MixedMethodGradient: // raw gradient for P riemann gradient for v
          newRiemannDvDxi -= deltaDvDxi;
          newRiemannDvDxj -= deltaDvDxj;
          newRiemannDpDxi -= (Pi-Pj)*gradPsii;
          newRiemannDpDxj -= (Pi-Pj)*gradPsij;
          break;       
        default:
          break;
          // do nada    
        }

      // XSPH
      //-----------------------------------------------------------
      if (xsph) {
        XSPHDeltaVi -= psii*(vi-vstar);
        XSPHDeltaVj -= psij*(vj-vstar);
      }

      normi += psii;
      normj += psij;

    } // loop over pairs
    threadReduceFieldLists<Dimension>(threadStack);
  } // OpenMP parallel region


  // Finish up the derivatives for each point.
  for (auto nodeListi = 0u; nodeListi < numNodeLists; ++nodeListi) {
    const auto& nodeList = mass[nodeListi]->nodeList();
    const auto  hmin = nodeList.hmin();
    const auto  hmax = nodeList.hmax();
    const auto  hminratio = nodeList.hminratio();
    const auto  nPerh = nodeList.nodesPerSmoothingScale();

    const auto ni = nodeList.numInternalNodes();
#pragma omp parallel for
    for (auto i = 0u; i < ni; ++i) {

      // Get the state for node i.
      const auto  numNeighborsi = connectivityMap.numNeighborsForNode(nodeListi, i);
      const auto& ri = position(nodeListi, i);
      const auto& mi = mass(nodeListi, i);
      const auto& voli = volume(nodeListi,i);
      const auto& vi = velocity(nodeListi, i);
      const auto& rhoi = massDensity(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
      const auto  Hdeti = Hi.Determinant();
      CHECK(mi > 0.0);
      CHECK(rhoi > 0.0);
      CHECK(Hdeti > 0.0);

      auto& normi = normalization(nodeListi, i);
      auto& DxDti = DxDt(nodeListi, i);
      auto& DrhoDti = DrhoDt(nodeListi, i);
      auto& DvDti = DvDt(nodeListi, i);
      auto& DepsDti = DepsDt(nodeListi, i);
      auto& DvDxi = DvDx(nodeListi, i);
      //auto& localDvDxi = localDvDx(nodeListi, i);
      //auto& localMi = localM(nodeListi, i);
      auto& DHDti = DHDt(nodeListi, i);
      auto& Hideali = Hideal(nodeListi, i);
      //auto& XSPHWeightSumi = XSPHWeightSum(nodeListi, i);
      auto& XSPHDeltaVi = XSPHDeltaV(nodeListi, i);
      auto& weightedNeighborSumi = weightedNeighborSum(nodeListi, i);
      auto& massSecondMomenti = massSecondMoment(nodeListi, i);

      DvDti /= mi;
      DepsDti /= mi;

      normi += voli*Hdeti*W0;

      DrhoDti = - rhoi * DvDxi.Trace() ;

      // If needed finish the total energy derivative.
      if (totalEnergy) DepsDti = mi*(vi.dot(DvDti) + DepsDti);

      // Complete the moments of the node distribution for use in the ideal H calculation.
      weightedNeighborSumi = Dimension::rootnu(max(0.0, weightedNeighborSumi/Hdeti));
      massSecondMomenti /= Hdeti*Hdeti;

      // Determine the position evolution, based on whether we're doing XSPH or not.
      DxDti = vi;
      if (xsph){
        DxDti += XSPHDeltaVi/max(tiny, normi);
      } 

      // in case we want to use local DvDx for Rieman Gradient
      //const auto localMdeti = localMi.Determinant();
      //const auto goodLocalM = ( localMdeti > 1.0e-2 and numNeighborsi > Dimension::pownu(2));
      //localMi =  (goodLocalM ? localMi.Inverse() : Tensor::one);
      //localDvDxi = localDvDxi*localMi;

      // The H tensor evolution.
      DHDti = smoothingScale.smoothingScaleDerivative(Hi,
                                                             ri,
                                                             DvDxi,
                                                             hmin,
                                                             hmax,
                                                             hminratio,
                                                             nPerh);
      Hideali = smoothingScale.newSmoothingScale(Hi,
                                                        ri,
                                                        weightedNeighborSumi,
                                                        massSecondMomenti,
                                                        W,
                                                        hmin,
                                                        hmax,
                                                        hminratio,
                                                        nPerh,
                                                        connectivityMap,
                                                        nodeListi,
                                                        i);
    } // nodes loop
  } // nodeLists loop

  TIME_GSPHevalDerivs.stop();
} // eval derivs method 


//------------------------------------------------------------------------------
// EvalDerivs subroutine for spatial derivs
//------------------------------------------------------------------------------
template<typename Dimension>
void
GSPHHydroBase<Dimension>::
computeMCorrection(const typename Dimension::Scalar /*time*/,
                   const typename Dimension::Scalar /*dt*/,
                   const DataBase<Dimension>& dataBase,
                   const State<Dimension>& state,
                         StateDerivatives<Dimension>& derivatives) const {

  // The kernels and such.
  const auto& W = this->kernel();
  const auto gradType = this->gradientType();

  // The connectivity.
  const auto& connectivityMap = dataBase.connectivityMap();
  const auto& nodeLists = connectivityMap.nodeLists();
  const auto numNodeLists = nodeLists.size();

  // Get the state and derivative FieldLists. 
  const auto volume = state.fields(HydroFieldNames::volume, 0.0);
  const auto velocity = state.fields(HydroFieldNames::velocity, Vector::zero);
  const auto pressure = state.fields(HydroFieldNames::pressure, 0.0);
  const auto position = state.fields(HydroFieldNames::position, Vector::zero);
  const auto H = state.fields(HydroFieldNames::H, SymTensor::zero);
  CHECK(mass.size() == numNodeLists);
  CHECK(position.size() == numNodeLists);
  CHECK(massDensity.size() == numNodeLists);
  CHECK(H.size() == numNodeLists);

  auto  M = derivatives.fields(HydroFieldNames::M_SPHCorrection, Tensor::zero);
  auto  newRiemannDpDx = derivatives.fields(ReplaceFieldList<Dimension, Scalar>::prefix() + GSPHFieldNames::RiemannPressureGradient,Vector::zero);
  auto  newRiemannDvDx = derivatives.fields(ReplaceFieldList<Dimension, Scalar>::prefix() + GSPHFieldNames::RiemannVelocityGradient,Tensor::zero);
  
  CHECK(M.size() == numNodeLists);

  // The set of interacting node pairs.
  const auto& pairs = connectivityMap.nodePairList();
  const auto  npairs = pairs.size();

#pragma omp parallel
  {
    // Thread private scratch variables
    int i, j, nodeListi, nodeListj;

    typename SpheralThreads<Dimension>::FieldListStack threadStack;
    auto M_thread = M.threadCopy(threadStack);
    auto newRiemannDpDx_thread = newRiemannDpDx.threadCopy(threadStack);
    auto newRiemannDvDx_thread = newRiemannDvDx.threadCopy(threadStack);

#pragma omp for
    for (auto kk = 0u; kk < npairs; ++kk) {
      i = pairs[kk].i_node;
      j = pairs[kk].j_node;
      nodeListi = pairs[kk].i_list;
      nodeListj = pairs[kk].j_list;
      
      // Get the state for node i.
      const auto& ri = position(nodeListi, i);
      const auto& voli = volume(nodeListi, i);
      const auto& vi = velocity(nodeListi, i);
      const auto& Pi = pressure(nodeListi, i);
      const auto& Hi = H(nodeListi, i);
      const auto  Hdeti = Hi.Determinant();
      CHECK(mi > 0.0);
      CHECK(Hdeti > 0.0);

      auto& Mi = M_thread(nodeListi, i);
      auto& newRiemannDpDxi = newRiemannDpDx_thread(nodeListi, i);
      auto& newRiemannDvDxi = newRiemannDvDx_thread(nodeListi, i);

      // Get the state for node j
      const auto& rj = position(nodeListj, j);
      const auto& volj = volume(nodeListj, j);
      const auto& vj = velocity(nodeListj, j);
      const auto& Pj = pressure(nodeListj, j);
      const auto& Hj = H(nodeListj, j);
      const auto  Hdetj = Hj.Determinant();
      CHECK(mj > 0.0);
      CHECK(Hdetj > 0.0);

      auto& Mj = M_thread(nodeListj, j);
      auto& newRiemannDpDxj = newRiemannDpDx_thread(nodeListj, j);
      auto& newRiemannDvDxj = newRiemannDvDx_thread(nodeListj, j);

      const auto rij = ri - rj;

      const auto etai = Hi*rij;
      const auto etaj = Hj*rij;
      const auto etaMagi = etai.magnitude();
      const auto etaMagj = etaj.magnitude();
      CHECK(etaMagi >= 0.0);
      CHECK(etaMagj >= 0.0);

      const auto gWi = W.gradValue(etaMagi, Hdeti);
      const auto Hetai = Hi*etai.unitVector();
      const auto gradWi = gWi*Hetai;

      const auto gWj = W.gradValue(etaMagj, Hdetj);
      const auto Hetaj = Hj*etaj.unitVector();
      const auto gradWj = gWj*Hetaj;

      const auto gradPsii = volj*gradWi;
      const auto gradPsij = voli*gradWj;

      // Linear gradient correction term.
      Mi -= rij.dyad(gradPsii);
      Mj -= rij.dyad(gradPsij);
      // // based on nodal values
      if (gradType == GradientType::SPHSameTimeGradient){
        newRiemannDpDxi -= (Pi-Pj)*gradPsii;
        newRiemannDpDxj -= (Pi-Pj)*gradPsij;

        newRiemannDvDxi -= (vi-vj).dyad(gradPsii);
        newRiemannDvDxj -= (vi-vj).dyad(gradPsij);
      }
    } // loop over pairs

    // Reduce the thread values to the master.
    threadReduceFieldLists<Dimension>(threadStack);

  }   // OpenMP parallel region
  
  // Finish up the spatial gradient calculation
  for (auto nodeListi = 0u; nodeListi < numNodeLists; ++nodeListi) {
    const auto& nodeList = M[nodeListi]->nodeList();
    const auto ni = nodeList.numInternalNodes();
#pragma omp parallel for
    for (auto i = 0u; i < ni; ++i) {
      const auto  numNeighborsi = connectivityMap.numNeighborsForNode(nodeListi, i);
      auto& Mi = M(nodeListi, i);
      auto& newRiemannDpDxi = newRiemannDpDx(nodeListi, i);
      auto& newRiemannDvDxi = newRiemannDvDx(nodeListi, i);

      const auto Mdeti = std::abs(Mi.Determinant());

      const auto enoughNeighbors =  numNeighborsi > Dimension::pownu(2);
      const auto goodM =  (Mdeti > 1e-2 and enoughNeighbors);                   

      Mi = ( goodM ? Mi.Inverse() : Tensor::one);

      if (gradType == GradientType::SPHSameTimeGradient){
        newRiemannDpDxi = Mi.Transpose()*newRiemannDpDxi;
        newRiemannDvDxi = newRiemannDvDxi*Mi;
      }
    }
    
  }
  
  for (ConstBoundaryIterator boundItr = this->boundaryBegin();
         boundItr != this->boundaryEnd();
         ++boundItr)(*boundItr)->applyFieldListGhostBoundary(M);

  if (gradType == GradientType::SPHSameTimeGradient){ 
    for (ConstBoundaryIterator boundItr = this->boundaryBegin();
          boundItr != this->boundaryEnd();
           ++boundItr){
      (*boundItr)->applyFieldListGhostBoundary(newRiemannDpDx);
      (*boundItr)->applyFieldListGhostBoundary(newRiemannDvDx);
    }
  }
  for (ConstBoundaryIterator boundaryItr = this->boundaryBegin(); 
         boundaryItr != this->boundaryEnd();
         ++boundaryItr) (*boundaryItr)->finalizeGhostBoundary();
 
}

} // spheral namespace