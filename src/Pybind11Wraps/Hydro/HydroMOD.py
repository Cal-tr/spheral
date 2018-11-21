"""
Spheral Hydro module.

Provides the support classes for hydro algorithms.
"""

from PYB11Generator import *
from SpheralCommon import *
from spheralDimensions import *
dims = spheralDimensions()

#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------
includes += ['"Geometry/GeomPlane.hh"',
             '"Hydro/HydroFieldNames.hh"']

#-------------------------------------------------------------------------------
# Namespaces
#-------------------------------------------------------------------------------
namespaces = ["Spheral"]

#-------------------------------------------------------------------------------
# HydroFieldNames
#-------------------------------------------------------------------------------
class HydroFieldNames:

    mass = PYB11readonly(static=True)
    position = PYB11readonly(static=True)
    velocity = PYB11readonly(static=True)
    H = PYB11readonly(static=True)
    work = PYB11readonly(static=True)
    velocityGradient = PYB11readonly(static=True)
    internalVelocityGradient = PYB11readonly(static=True)
    hydroAcceleration = PYB11readonly(static=True)
    massDensity = PYB11readonly(static=True)
    normalization = PYB11readonly(static=True)
    specificThermalEnergy = PYB11readonly(static=True)
    maxViscousPressure = PYB11readonly(static=True)
    effectiveViscousPressure = PYB11readonly(static=True)
    massDensityCorrection = PYB11readonly(static=True)
    viscousWork = PYB11readonly(static=True)
    XSPHDeltaV = PYB11readonly(static=True)
    XSPHWeightSum = PYB11readonly(static=True)
    Hsmooth = PYB11readonly(static=True)
    massFirstMoment = PYB11readonly(static=True)
    massSecondMoment = PYB11readonly(static=True)
    weightedNeighborSum = PYB11readonly(static=True)
    pressure = PYB11readonly(static=True)
    temperature = PYB11readonly(static=True)
    soundSpeed = PYB11readonly(static=True)
    pairAccelerations = PYB11readonly(static=True)
    pairWork = PYB11readonly(static=True)
    gamma = PYB11readonly(static=True)
    entropy = PYB11readonly(static=True)
    PSPHcorrection = PYB11readonly(static=True)
    omegaGradh = PYB11readonly(static=True)
    numberDensitySum = PYB11readonly(static=True)
    timeStepMask = PYB11readonly(static=True)
    m0_CRKSPH = PYB11readonly(static=True)
    m1_CRKSPH = PYB11readonly(static=True)
    m2_CRKSPH = PYB11readonly(static=True)
    m3_CRKSPH = PYB11readonly(static=True)
    m4_CRKSPH = PYB11readonly(static=True)
    gradM0_CRKSPH = PYB11readonly(static=True)
    gradM1_CRKSPH = PYB11readonly(static=True)
    gradM2_CRKSPH = PYB11readonly(static=True)
    gradM3_CRKSPH = PYB11readonly(static=True)
    gradM4_CRKSPH = PYB11readonly(static=True)
    A0_CRKSPH = PYB11readonly(static=True)
    A_CRKSPH = PYB11readonly(static=True)
    B_CRKSPH = PYB11readonly(static=True)
    C_CRKSPH = PYB11readonly(static=True)
    gradA0_CRKSPH = PYB11readonly(static=True)
    gradA_CRKSPH = PYB11readonly(static=True)
    gradB_CRKSPH = PYB11readonly(static=True)
    gradC_CRKSPH = PYB11readonly(static=True)
    surfacePoint = PYB11readonly(static=True)
    voidPoint = PYB11readonly(static=True)
    etaVoidPoints = PYB11readonly(static=True)
    M_SPHCorrection = PYB11readonly(static=True)
    volume = PYB11readonly(static=True)
    linearMomentum = PYB11readonly(static=True)
    totalEnergy = PYB11readonly(static=True)
    mesh = PYB11readonly(static=True)
    hourglassMask = PYB11readonly(static=True)
    faceVelocity = PYB11readonly(static=True)
    faceForce = PYB11readonly(static=True)
    faceMass = PYB11readonly(static=True)
    polyvols = PYB11readonly(static=True)
    massDensityGradient = PYB11readonly(static=True)
    ArtificialViscousClMultiplier = PYB11readonly(static=True)
    ArtificialViscousCqMultiplier = PYB11readonly(static=True)
