#-------------------------------------------------------------------------------
# PolytropicEquationOfState
#-------------------------------------------------------------------------------
from PYB11Generator import *
from EOSAbstractMethods import *

@PYB11template("Dimension")
class PolytropicEquationOfState:

    typedefs = """
    typedef typename %(Dimension)s::Scalar Scalar;
    typedef Field<%(Dimension)s, Scalar> ScalarField;
"""

    #...........................................................................
    # Constructors
    def pyinit(self,
               K = "const double",
               index = "const double",
               mu = "const double",
               constants = "const PhysicalConstants&",
               minimumPressure = ("const double", "-std::numeric_limits<double>::max()"),
               maximumPressure = ("const double",  "std::numeric_limits<double>::max()"),
               minPressureType = ("const MaterialPressureMinType", "MaterialPressureMinType::PressureFloor")):
        "Provides the polytropic EOS: P = K rho^gamma; gamma = (index + 1)/index"

    #...........................................................................
    # Methods
    @PYB11const
    def pressure(self,
                 massDensity = "const Scalar",
                 specificThermalEnergy = "const Scalar"):
        return "Scalar"

    @PYB11const
    def temperature(self,
                    massDensity = "const Scalar",
                    specificThermalEnergy = "const Scalar"):
        return "Scalar"

    @PYB11const
    def specificThermalEnergy(self,
                              massDensity = "const Scalar",
                              temperature = "const Scalar"):
        return "Scalar"

    @PYB11const
    def specificHeat(self,
                     massDensity = "const Scalar",
                     temperature = "const Scalar"):
        return "Scalar"

    @PYB11const
    def soundSpeed(self,
                   massDensity = "const Scalar",
                   specificThermalEnergy = "const Scalar"):
        return "Scalar"

    # @PYB11const
    # @PYB11cppname("gamma")
    # def gamma(self,
    #            massDensity = "const Scalar",
    #            specificThermalEnergy = "const Scalar"):
    #     return "Scalar"

    @PYB11const
    def bulkModulus(self,
                    massDensity = "const Scalar",
                    specificThermalEnergy = "const Scalar"):
        return "Scalar"

    @PYB11const
    def entropy(self,
                massDensity = "const Scalar",
                specificThermalEnergy = "const Scalar"):
        return "Scalar"

    #...........................................................................
    # Properties
    polytropicConstant = PYB11property("double", "polytropicConstant", doc="K: the polytropic constant in front")
    polytropicIndex = PYB11property("double", "polytropicIndex", doc="polytropic index: gamma = (index + 1)/index")
    gamma = PYB11property("double", "gamma", doc="gamma: ratio of specific heats")
    molecularWeight = PYB11property("double", "molecularWeight", doc="mean molecular weight")
    externalPressure = PYB11property("double", "externalPressure", "setExternalPressure", doc="Any external pressure (subtracted from the pressure calculation")
    
#-------------------------------------------------------------------------------
# Add the virtual interface
#-------------------------------------------------------------------------------
PYB11inject(EOSAbstractMethods, PolytropicEquationOfState)
