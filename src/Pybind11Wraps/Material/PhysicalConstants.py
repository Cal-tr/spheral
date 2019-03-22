#-------------------------------------------------------------------------------
# PhysicalConstants
#-------------------------------------------------------------------------------
from PYB11Generator import *

class PhysicalConstants:
    """
Choose the physical units for a given Spheral run.

This is done by constructing with the user choice for unit length, mass, and time in
SI units (m, kg, sec). All other constants are then derived from those choices.
"""

    #...........................................................................
    # Constructor
    def pyinit(self,
               unitLm = "const double",
               unitMkg = "const double",
               unitTsec = "const double"):
        "Construct based on a unit length, unit mass, and unit time in SI units"
        return

    #...........................................................................
    # Properties
    unitLengthMeters = PYB11property("double", "unitLengthMeters", doc="unit of length in SI")
    unitMassKg       = PYB11property("double", "unitMassKg", doc="unit of length in SI")
    unitTimeSec      = PYB11property("double", "unitTimeSec", doc="unit of length in SI")

    protonMass              = PYB11property("double", "protonMass", doc="proton mass")
    electronMass            = PYB11property("double", "electronMass", doc="electron mass")
    electronCharge          = PYB11property("double", "electronCharge", doc="electron charge")
    G                       = PYB11property("double", "G", doc="gravitational constant")
    c                       = PYB11property("double", "c", doc="speed of light")
    kB                      = PYB11property("double", "kB", doc="Boltzmann constant")
    Navogadro               = PYB11property("double", "Navogadro", doc="Avogadro's constant")
    molarGasConstant        = PYB11property("double", "molarGasConstant",
                                            doc="R: the molar gas constant")
    kelvinsToEnergyPerMole  = PYB11property("double", "kelvinsToEnergyPerMole",
                                            doc="Conversion factor from Kelvins to energy")
    unitMassDensity         = PYB11property("double", "unitMassDensity",
                                            doc="What the unit mass density in these units corresponds to in SI")
    stefanBoltzmannConstant = PYB11property("double", "stefanBoltzmannConstant",
                                            doc="sigma: the Steffan-Boltzmann constant")
    blackBodyConstant = PYB11property("double", "blackBodyConstant",
                                      doc="a: the black body constant")
