#-------------------------------------------------------------------------------
# ConnectivityMap
#-------------------------------------------------------------------------------
from PYB11Generator import *

@PYB11template("Dimension")
class ConnectivityMap:

    typedefs = """
    typedef typename %(Dimension)s::Scalar Scalar;
    typedef typename %(Dimension)s::Vector Vector;
    typedef typename %(Dimension)s::Tensor Tensor;
    typedef typename %(Dimension)s::SymTensor SymTensor;
    typedef NodeList<%(Dimension)s> NodeListType;
"""

    #...........................................................................
    # Constructors
    def pyinit(self):
        "Default constructor"

    #...........................................................................
    # Methods
    def patchConnectivity(self,
                          flags = "const FieldList<%(Dimension)s, int>&",
                          old2new = "const FieldList<%(Dimension)s, int>&"):
        "Patch the connectivity information"
        return "void"

    @PYB11returnpolicy("reference_internal")
    @PYB11const
    def connectivityForNode(self,
                            nodeList = "const NodeListType*",
                            nodeID = "const int"):
        "Get the set of neighbors for the given (internal!) node in the given NodeList."
        return "const std::vector<std::vector<int>>&"

    @PYB11returnpolicy("reference_internal")
    @PYB11const
    @PYB11pycppname("connectivityForNode")
    def connectivityForNode1(self,
                             nodeListID = "const int",
                             nodeID = "const int"):
        "Get the set of neighbors for the given (internal!) node in the given NodeList."
        return "const std::vector<std::vector<int>>&"

    @PYB11const
    def connectivityIntersectionForNodes(self,
                                         nodeListi = "const int",
                                         i = "const int",
                                         nodeListj = "const int",
                                         j = "const int"):
        "Compute the common neighbors for a pair of nodes."
        return "std::vector< std::vector<int> >"

    @PYB11const
    def connectivityUnionForNodes(self,
                                  nodeListi = "const int",
                                  i = "const int",
                                  nodeListj ="const int",
                                  j = "const int"):
        "Compute the union of neighbors for a pair of nodes."
        return "std::vector< std::vector<int> >"

    @PYB11const
    def numNeighborsForNode(self,
                            nodeListPtr = "const NodeListType*",
                            nodeID = "const int"):
        "Compute the number of neighbors for the given node."
        return "int"

    @PYB11const
    @PYB11pycppname("numNeighborsForNode")
    def numNeighborsForNode1(self,
                             nodeListID = "const int",
                             nodeID = "const int"):
        "Compute the number of neighbors for the given node."
        return "int"

    @PYB11const
    def globalConnectivity(self,
                           boundaries = "std::vector<Boundary<%(Dimension)s>*>&"):
        "Return the connectivity in terms of global node IDs."
        return "std::map<int, std::vector<int> >"

    @PYB11const
    def calculatePairInteraction(self,
                                 nodeListi = "const int",
                                 i = "const int", 
                                 nodeListj = "const int",
                                 j = "const int",
                                 firstGhostNodej ="const int"):
        "Function to determine if given node information (i and j), if the pair should already have been calculated by iterating over each others neighbors."
        return "bool"

    @PYB11const
    def numNodes(self, nodeList="const int"):
        "Return the number of nodes we should walk for the given NodeList."
        return "int"

    @PYB11const
    def ithNode(self,
                nodeList = "const int",
                index = "const int"):
        "The ith node (ordered) in the given NodeList."
        return "int"

    @PYB11returnpolicy("reference_internal")
    @PYB11const
    def nodeList(self, index="const int"):
        "Get the ith NodeList or FluidNodeList."
        return "const NodeListType&"

    @PYB11const
    def nodeListIndex(self,  nodeList="const NodeListType*"):
        "Return which NodeList index in order the given one would be in our connectivity."
        return "unsigned"

    @PYB11const
    def valid(self):
        "Check that the internal data structure is valid."
        return "bool"

    #...........................................................................
    # Properties
    buildGhostConnectivity = PYB11property("bool", "buildGhostConnectivity", doc="Are we building connectivity for ghost nodes?")
    nodeLists = PYB11property("const std::vector<NodeListType*>", "nodeLists", doc="The set of NodeLists we have connectivity for")
