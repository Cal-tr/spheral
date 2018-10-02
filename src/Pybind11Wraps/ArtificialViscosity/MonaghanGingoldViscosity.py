#-------------------------------------------------------------------------------
# MonaghanGingoldViscosity
#-------------------------------------------------------------------------------
from PYB11Generator import *
from ArtificialViscosity import *
from ArtificialViscosityAbstractMethods import *

@PYB11template("Dimension")
class MonaghanGingoldViscosity(ArtificialViscosity):

    typedefs = """
    typedef %(Dimension)s DIM;
    typedef typename DIM::Scalar Scalar;
    typedef typename DIM::Vector Vector;
    typedef typename DIM::Tensor Tensor;
    typedef typename DIM::SymTensor SymTensor;
    typedef typename DIM::ThirdRankTensor ThirdRankTensor;
"""

    #...........................................................................
    # Constructors
    def pyinit(self,
               Clinear = "const Scalar",
               Cquadratic = "const Scalar",
               linearInExpansion = ("bool", "false"),
               quadraticInExpansion = ("bool", "false")):
        "MonaghanGingoldViscosity constructor"


    #...........................................................................
    # Methods
    @PYB11virtual
    @PYB11const
    def label(self):
        return "std::string"

    #...........................................................................
    # Properties
    linearInExpansion = PYB11property("bool", "linearInExpansion", "linearInExpansion", 
                                      doc="Toggle if the linearviscosity is active for expansion flows")
    quadraticInExpansion = PYB11property("bool", "quadraticInExpansion", "quadraticInExpansion", 
                                         doc="Toggle if the quadratic viscosity is active for expansion flows")
    
#-------------------------------------------------------------------------------
# Inject abstract interface
#-------------------------------------------------------------------------------
PYB11inject(ArtificialViscosityAbstractMethods, MonaghanGingoldViscosity, virtual=True, pure_virtual=False)
