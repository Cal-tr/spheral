#-------------------------------------------------------------------------------
# Class and support classess/functions to allow most of the nodes to be stored
# and later added to the simulation based on some user defined activation function
# The method expand 
#-------------------------------------------------------------------------------
from Spheral import *
import math
import mpi

#
class Ellipse2d:
    def __init__(self,
                 axisA = Vector2d(0.0), # ellipse axis a
                 axisB = Vector2d(0.0), # ellipse axis b (not needed for 2d)
                 center=Vector2d(0.0),  # center
                 abc = Vector2d(0.0)):  # lengths of semi-axis a and b

        assert abc[0]>0.0
        assert abc[1]>0.0  

        self.abc = abc 

        
        ux = axisA.unitVector()
        self.Rot = Tensor2d(ux[0],ux[1],
                         -ux[1],ux[0])
        self.center = self.Rot.dot(center)
        
    def isInside(self,pos):
        POS = self.Rot.dot(pos)-self.center 
        InOut = (POS[0]/self.abc[0])**2 + (POS[1]/self.abc[1])**2
        if InOut <= 1.0:
            return True
        else:
            return False
            
    def isOutside(self,pos):
        POS = self.Rot.dot(pos)-self.center
        InOut = (POS[0]/self.abc[0])**2 + (POS[1]/self.abc[1])**2
        if InOut <= 1.0: 
            return False
        else:
            return True
            
class Ellipse3d:
    def __init__(self,
                 axisA = Vector3d(1.0,0.0,0.0), # ellipse axis a 
                 axisB = Vector3d(0.0,1.0,0.0), # ellipse axis b
                 center = Vector3d(0.0),        # center
                 abc = Vector3d(0.0)):          # lengths of semi-axis a,b,c

        assert abc[0]>0.0
        assert abc[1]>0.0  
        assert abc[2]>0.0

        self.abc = abc
        
        ux = axisA.unitVector()
        uy = ux.cross(axisB).unitVector()
        uz = ux.cross(uy).unitVector()
            
        self.Rot = Tensor3d(ux[0],ux[1],ux[2],
                          uy[0],uy[1],uy[2],
                          uz[0],uz[1],uz[2])
        self.center = self.Rot.dot(center)
            
        
    def isInside(self,pos):
        POS = self.Rot.dot(pos)-self.center 
        InOut = (POS[0]/self.abc[0])**2 + (POS[1]/self.abc[1])**2 + (POS[2]/self.abc[2])**2
        if InOut <= 1.0: 
            return True
        else:
            return False
            
    def isOutside(self,pos):
        POS = self.Rot.dot(pos)-self.center 
        InOut = (POS[0]/self.abc[0])**2 + (POS[1]/self.abc[1])**2 + (POS[2]/self.abc[2])**2
        if InOut <= 1.0: 
            return False
        else:
            return True
       
class ConstantBoundaryWrapper3d(ConstantBoundary3d):
    def __init__(self, db, nodes, constNodes, plane):
        ConstantBoundary3d.__init__(self,db, nodes, constNodes, plane)
        self.nodeListName = nodes.name
    def label(self):
        return "ExpandingDomain_"+self.nodeListName+"_ConstBC"
    def notifyBeforeRedistribution(self):
        return
    def notifyAfterRedistribution(self):
        return
        
class ConstantBoundaryWrapper2d(ConstantBoundary2d):
    def __init__(self, db, nodes, constNodes, plane):
        ConstantBoundary2d.__init__(self, db, nodes, constNodes, plane)
        self.nodeListName = nodes.name
    def label(self):
        return "ExpandingDomain_"+self.nodeListName+"_ConstBC"
    def notifyBeforeRedistribution(self):
        return
    def notifyAfterRedistribution(self):
        return

class defaultExpansionTriggerWrapper:
    def __init__(self,db):
        self.nodeLists = db.nodeLists()
        self.vel = db.globalVelocity
        if db.numSolidNodes>0: 
            self.rho = db.solidMassDensity
            self.eps = db.solidSpecificThermalEnergy
        else:
            self.rho = db.solidMassDensity
            self.eps = db.fluidSpecificThermalEnergy

    def defaultExpansionTrigger(self,nodeListi,i):
        csi = self.nodeLists[i].eos.soundSpeed(self.rho(nodeListi,i),self.eps(nodeListi,i))
        machi = self.vel(nodeListi,i).magnitude()/csi
        result = False
        if machi > 0.1:
            result = True
        return result

def alwayFalseFunc(posi):
    return False

class ConstantRho:
    def __init__(self, rho0):
        assert type(rho0) is float
        self.rho0 = rho0
        return
    def rho(self, r):
        return self.rho0
    def __call__(self, r):
        return self.rho(r)

##################################################################################################
#
#
#
#
##################################################################################################
class ExpandingDomain:
        def __init__(self,
                 integrator=None,           # spheral controller obj
                 constBCPlane=None,         # plane for internal constant boundary
                 redistributer=None,        # we'll need to shuffle things around a bit
                 rho_func = None,           # function for quiessant density
                 removeViolaters=True,      # remove internal nodes that move past the expansion front
                 initialRadius=None,        # radius or vector of major axis lengths
                 initialCenter=None,        # center of sphere/ellipse 
                 ellipseAxisA=None,         # fixed axis
                 ellipseAxisB=None,         # any non collinear axis 
                 bcThickness=None,
                 testZoneThickness=None,    # width of interior zone expansion var is sampled over
                 stepSize=None,
                 expansionTriggerFunc=None,              # func taking in nodelist index and node index returning true/false
                 permanentConstNodeFunc=alwayFalseFunc,  # func taking in position to specify final const nodes
                 additionalFields=None,     # allows user to add additional fields from other physics models 
                 auxillaryFunctions=[],    # to get constantAcceleration to play nice
                 ):

            # we're going to want to know when
            # the last time the expansion was triggered
            self.restart = RestartableObject(self)
            self.last_time = 0.0
            self.last_dt = 0.0
            self.last_cycle = 0
            self.nDeleted = 0
            
            # bool control switches
            self.expansionCompleted=False
            self.removeViolaters = removeViolaters

            # other things we'll need
            self.redistributeTimer = SpheralTimer("Time for redistributing nodes.")
            self.integrator=integrator                      
            self.activeDatabase = integrator.dataBase
            self.maxKernelExtent = self.activeDatabase.maxKernelExtent
            self.dim = self.activeDatabase.nDim
            self.plane = constBCPlane
            self.auxillaryFunctions=auxillaryFunctions
            self.tempBCstorage=[]
            self.constBC=0.0
            self.stepSize = stepSize
            self.testZoneThickness = testZoneThickness
            self.bcThickness = bcThickness
            self.permanentConstNodeFunc=permanentConstNodeFunc
            self.rho_func = rho_func

            # some dimensional aliases we'll need
            if self.dim ==3:
                Ellipse = Ellipse3d
                Plane = Plane3d
                self.IntField = IntField3d
                self.ConstantBoundary = ConstantBoundaryWrapper3d
                self.Vector = Vector3d
                self.State = State3d
                self.StateDerivatives = StateDerivatives3d
                if redistributer:
                    self.redistributer = redistributer
                elif mpi.procs>1:
                    self.redistributer = PeanoHilbertOrderRedistributeNodes3d(self.activeDatabase.maxKernelExtent,workBalance=False)

            elif self.dim ==2:
                Ellipse = Ellipse2d
                Plane = Plane2d
                self.IntField = IntField2d
                self.ConstantBoundary = ConstantBoundaryWrapper2d
                self.Vector = Vector2d
                self.State = State2d
                self.StateDerivatives = StateDerivatives2d
                if redistributer:
                    self.redistributer = redistributer
                elif mpi.procs>1:
                    self.redistributer = PeanoHilbertOrderRedistributeNodes2d(self.activeDatabase.maxKernelExtent,workBalance=False)
            
            assert type(removeViolaters) is bool
            assert type(initialRadius) in [float,int,self.Vector]
            assert type(initialCenter) is self.Vector or initialCenter is None
            assert type(ellipseAxisA) is self.Vector or ellipseAxisA is None
            assert type(ellipseAxisB) is self.Vector or ellipseAxisB is None
            assert (ellipseAxisA is None and ellipseAxisB is None) or (ellipseAxisA is not None and ellipseAxisB is not None)
            assert initialRadius is not None
            
            # set up fields we're tracking
            #----------------------------------------------------------------
            theState = self.State(self.activeDatabase,integrator.physicsPackages())
            theDeriv = self.StateDerivatives(self.activeDatabase,integrator.physicsPackages())

            allFields = list(theState.allIntFields())
            allFields.extend(list(theState.allScalarFields()))
            allFields.extend(list(theState.allVectorFields()))
            allFields.extend(list(theState.allSymTensorFields()))
            allFields.extend(list(theState.allTensorFields()))

            allFields.extend(list(theDeriv.allIntFields()))
            allFields.extend(list(theDeriv.allScalarFields()))
            allFields.extend(list(theDeriv.allVectorFields()))
            allFields.extend(list(theDeriv.allSymTensorFields()))
            allFields.extend(list(theDeriv.allTensorFields()))

            # same important fields we know the state/derivs
            # don't track and we need to check for.
            for package in integrator.physicsPackages():
                if ('Damage' in package.label()):
                    if hasattr(package, 'flaws'):
                        allFields.append(package.flaws)
                if ('ConstantAcceleration' in package.label()):
                    if hasattr(package, 'flags'):
                        allFields.append(package.flags)
            
            # user defined additional fields
            if additionalFields:
                for candidateField in additionalFields:
                        allFields.append(candidateField)
            
            # take only unique entries
            allFields = list(set(allFields))

            self.activeFields=[]
            self.inactiveFields=[]
            activeNodeLists = self.activeDatabase.nodeLists()
            for i in range(len(activeNodeLists)):
                nodeListi = activeNodeLists[i]
                self.activeFields.append([])

                for j in range(len(allFields)):
                    fieldj = allFields[j]
                    nodeListj = fieldj.nodeList()
                    if nodeListi == nodeListj:
                        if fieldj.name == 'position':
                            self.activeFields[i].insert(0,fieldj)
                        else:
                            self.activeFields[i].append(fieldj)

            # how are we doin this expansion?
            #------------------------------------------------------------------
            if expansionTriggerFunc is None:
                expTrigClass = defaultExpansionTriggerWrapper(self.activeDatabase)
                self.expansionTrigger = expTrigClass.defaultExpansionTrigger
            else:
                self.expansionTrigger = expansionTriggerFunc
                
            # active region / const node region / sampling region
            #------------------------------------------------------------------
            if type(initialRadius) is float or type(initialRadius) is int:
                initialRadius == self.Vector.one*initialRadius
             
            if ellipseAxisA is None:
                ellipseAxisA = self.Vector(1.0,0.0)
                ellipseAxisB = self.Vector(0.0,1.0)

            if initialCenter is None: 
                initialCenter = self.Vector.zero

            if testZoneThickness > min(initialRadius):
                print('Warning: testZoneThickness > initialRadius')
                print('    increasing the initialRadius by the testZone Thickness')
                initialRadius += self.Vector.one*testZoneThickness  
            self.activeZoneEllipse = Ellipse(axisA=ellipseAxisA,
                                             axisB=ellipseAxisB,
                                             center=initialCenter,
                                             abc=initialRadius+self.Vector.one*bcThickness)
            self.constBCEllipse = Ellipse(axisA=ellipseAxisA,
                                          axisB=ellipseAxisB,
                                          center=initialCenter,
                                          abc=initialRadius)
            self.testZoneEllipse = Ellipse(axisA=ellipseAxisA,
                                           axisB=ellipseAxisB,
                                           center=initialCenter,
                                           abc=initialRadius-self.Vector.one*testZoneThickness)

            # initialize a bunch of things then deinitialize to play nice with
            # the controller
            #------------------------------------------------------------------
            self.initializeActiveRegionMask()
            self.initializeDerivatives(0,0.0,0.0)
            self.initializeInactiveNodeLists()
            self.redistribute()
            self.setConstantBoundary(0,0.0,0.0)
            self.deinitialize()


        def expand(self, cycle, t, dt):
        #---------------------------------------------------------------
        # method used as a periodic work function to make it all happen
        #---------------------------------------------------------------

            if self.expansionRequired()==1 and not self.expansionCompleted:
                NstoreLocal = 0
                for i in range(len(self.inactiveFields)):
                    NstoreLocal += len(self.inactiveFields[i][0])
                Nstore = mpi.allreduce(NstoreLocal,mpi.SUM)

                if Nstore == 0:
                    self.expansionCompleted = True
                
                #update the ellipses 
                self.activeZoneEllipse.abc+=self.Vector.one*self.stepSize
                self.constBCEllipse.abc+=self.Vector.one*self.stepSize
                self.testZoneEllipse.abc+=self.Vector.one*self.stepSize

                self.readBoundaries()
                self.convertConstantNodes()
                self.clearBoundaries()

                self.transferNodes(cycle,t,dt) 

                self.redistribute()
                self.resetBoundaries(cycle,t,dt)
                self.applyAuxillaryFunctions() 
                
                self.last_time = t
                self.last_dt = dt
                self.last_cycle = cycle

        def expansionRequired(self): 
        #---------------------------------------------------------------
        # stowe essential information for nodes outside the user defined
        # initial local ellipse
        #--------------------------------------------------------------- 
            result = 0
            pos = self.activeDatabase.globalPosition
            nodeLists = self.activeDatabase.nodeLists()
            for nodeListi in range(self.activeDatabase.numNodeLists):
                for i in range(nodeLists[nodeListi].numInternalNodes):
                    posi = pos(nodeListi,i)
                    if self.testZoneEllipse.isOutside(posi) and self.expansionTrigger(nodeListi,i):
                        result = 1

            result = mpi.allreduce(result,mpi.MAX)
            return result

        def initializeInactiveNodeLists(self):
        #---------------------------------------------------------------
        # stowe essential information for nodes outside the user defined
        # initial local ellipse
        #---------------------------------------------------------------
            activeNodeLists = self.activeDatabase.nodeLists()

            for i in range(self.activeDatabase.numNodeLists):
                
                posi = self.activeFields[i][0]
                numFieldsi = len(self.activeFields[i])
                activeNodeLists[i].numGhostNodes=0
                numInternalNodesi = activeNodeLists[i].numInternalNodes 
                self.inactiveFields.append([])
                
                # make the transfer  
                indices_transfer = []
                for j in range(numFieldsi):
                    self.inactiveFields[i].append([])

                    for k in range(numInternalNodesi):  
                        if self.activeZoneEllipse.isOutside(posi[k]):
                            dataType = type(self.activeFields[i][j][k])
                            self.inactiveFields[i][j].append(dataType(self.activeFields[i][j][k]))
                            if j == 0:
                                indices_transfer.append(k)

                activeNodeLists[i].deleteNodes(vector_of_int(indices_transfer)) 
                

        def transferNodes(self,cycle,time,dt):
        #---------------------------------------------------------------
        # takes nodes from the inactive list and adds them to the active
        # list.
        #---------------------------------------------------------------
            activeNodeLists = self.activeDatabase.nodeLists()

            for i in range(self.activeDatabase.numNodeLists):
                numInactiveNodesi = len(self.inactiveFields[i][0])
                numFieldsi = len(self.activeFields[i])

                indices_transfer = [k for k in range(numInactiveNodesi) if self.activeZoneEllipse.isInside(self.inactiveFields[i][0][k])]
                N_transfer = len(indices_transfer)
                activeNodeLists[i].numInternalNodes += N_transfer 
                
                for j in range(numFieldsi):
                    counter=0
                    for index in sorted(indices_transfer, reverse=True): 
                        index_active = activeNodeLists[i].numInternalNodes-N_transfer+counter
                        dataType = type(self.inactiveFields[i][j][index])
                        self.activeFields[i][j][index_active] = dataType(self.inactiveFields[i][j][index])
                        del self.inactiveFields[i][j][index]
                        counter+=1
                        
                    
        def readBoundaries(self):
        #---------------------------------------------------------------
        # we need to read in all the BCs and store their refs temporarily in
        # a list so we can add them back in at the end of the expansion
        #---------------------------------------------------------------
            self.tempBCstorage = []
            for p in self.integrator.physicsPackages():
                self.tempBCstorage.append(p.boundaryConditions()) 
                del self.tempBCstorage[-1][:self.activeDatabase.numNodeLists] 
                
        def redistribute(self):
        #---------------------------------------------------------------
        # redistribution of nodes across processors
        # this was poached from Spheral_Controller.py
        #---------------------------------------------------------------  
            if mpi.procs>1:
                while gc.collect():
                    pass
                self.redistributeTimer.start()
                self.redistributer.redistributeNodes(self.activeDatabase, self.integrator.uniqueBoundaryConditions())
                self.redistributeTimer.stop()
                self.redistributeTimer.printStatus()

        def clearBoundaries(self):
        #---------------------------------------------------------------
        # clears out the bcs setting things up for expansion
        #---------------------------------------------------------------
            packages = self.integrator.physicsPackages()
            for p in packages:
                p.clearBoundaries()
            del self.constBC 
        
        def removeDynamicBoundaryViolaters(self):
        #---------------------------------------------------------------
        # this is an optional method that allows nodes violating the 
        # dynamic const bc to be culled (less relevant with field based
        # movement)
        #---------------------------------------------------------------
            if self.removeViolaters:
                activeNodeLists = self.activeDatabase.nodeLists()
                for j in range(self.activeDatabase.numNodeLists):
                    pos = activeNodeLists[j].positions()
                    indices = [i for i in range(activeNodeLists[j].numInternalNodes) if (self.constBCEllipse.isOutside(pos[i]) or self.permanentConstNodeFunc(pos[i]))]
                    activeNodeLists[j].deleteNodes(vector_of_int(indices)) 
                    self.nDeleted+=len(indices) 
                    print('--------------------------------')
                    print('Total Nodes Deleted: ',self.nDeleted)
                    print('-------------------------------')

        def setConstantBoundary(self,cycle,t,dt):
        #---------------------------------------------------------------
        # creates the internal constant boundary and inserts
        # into physic packages' bc list
        #---------------------------------------------------------------
            self.constBC=[]
            activeNodeLists = self.activeDatabase.nodeLists()
            for j in range(self.activeDatabase.numNodeLists):
                pos = activeNodeLists[j].positions()
                constNodes = [i for i in range(activeNodeLists[j].numInternalNodes) if (self.constBCEllipse.isOutside(pos[i])  or self.permanentConstNodeFunc(pos[i])) and self.activeRegionMask[j][i]==0 ]
                self.constBC.append(self.ConstantBoundary(self.activeDatabase, activeNodeLists[j], vector_of_int(constNodes), self.plane))
            
            packages = self.integrator.physicsPackages()
            for p in packages:
                for bc in self.constBC:
                    p.prependBoundary(bc) 
                
        def resetBoundaries(self,cycle,t,dt):
        #---------------------------------------------------------------
        # sets all the BCs back up including wrapping the constBC gen 
        #---------------------------------------------------------------

            self.setConstantBoundary(cycle,t,dt)
            
            packages = self.integrator.physicsPackages()
            for p in packages:
                for bc in self.tempBCstorage[0]:
                    p.appendBoundary(bc) 
                del self.tempBCstorage[0]

        def deinitialize(self):
        #---------------------------------------------------------------
        # set things back to the way they were 
        #---------------------------------------------------------------
            packages = self.integrator.physicsPackages()
            derivs = eval("StateDerivatives%sd(self.activeDatabase, packages)" % (self.dim)) # redefine in constructor
            derivs.Zero()
        
            for nodeList in self.activeDatabase.nodeLists():
                nodeList.numGhostNodes = 0 

        def initializeDerivatives(self,cycle,t,dt):
        #---------------------------------------------------------------
        # have to set the state of the newly added nodes for the
        # constant bcs buffered values. This is mostly from the
        # controller
        #---------------------------------------------------------------
            
            # make sure we're fresh
            self.deinitialize()

            packages = self.integrator.physicsPackages()
            uniquebcs = self.integrator.uniqueBoundaryConditions()
            state = self.State(self.activeDatabase, packages)
            derivs = self.StateDerivatives(self.activeDatabase, packages)

            # initialize our bcs
            for bc in uniquebcs:
                bc.initializeProblemStartup(False)
    
            # Create ghost nodes for the physics packages to initialize with.
            self.activeDatabase.reinitializeNeighbors()
            self.integrator.setGhostNodes()
            self.activeDatabase.updateConnectivityMap(False)

            
            #self.activeDatabase.reinitializeNeighbors()
            #self.integrator.setGhostNodes()
            #self.activeDatabase.updateConnectivityMap(False)
            self.integrator.applyGhostBoundaries(state, derivs)
            #for bc in uniquebcs:
            #    bc.initializeProblemStartup(False)

            # get our derivatives
            self.integrator.preStepInitialize(state, derivs) 
            dt = self.integrator.selectDt(self.integrator.dtMin, self.integrator.dtMax, state, derivs)
            self.integrator.initializeDerivatives(t, dt, state, derivs)
            self.integrator.evaluateDerivatives(t, dt, self.activeDatabase, state, derivs)
            self.integrator.finalizeDerivatives(t, dt, self.activeDatabase, state, derivs)
            
            # another crack at initialize in case it relies on storing t-1 derivatives
            self.integrator.initializeDerivatives(t, dt, state, derivs) 


        def initializeActiveRegionMask(self):
        #---------------------------------------------------------------
        # create a fields to track if nodes were ever fully active
        # this prevents them from being put back into the const-bc
        #---------------------------------------------------------------
            self.activeRegionMask=[]
            activeNodeLists = self.activeDatabase.nodeLists()
            for i in range(len(activeNodeLists)):
                nodes = activeNodeLists[i]
                self.activeRegionMask.append(self.IntField("expanding boundary active region mask" + nodes.name, nodes, 0))
            self.setActiveRegionMask()


        def setActiveRegionMask(self):
        #---------------------------------------------------------------
        # if were not in the const bc and not inactive set the mask to 1
        #---------------------------------------------------------------
            activeNodeLists = self.activeDatabase.nodeLists()
            for i in range(len(activeNodeLists)):
                nodes = activeNodeLists[i]
                pos   = nodes.positions()
                for j in range(nodes.numInternalNodes):
                    if self.activeZoneEllipse.isInside(pos[j]):
                        self.activeRegionMask[i][j] = 1

        def applyAuxillaryFunctions(self):  
        #---------------------------------------------------------------
        # this is a bandaid to for classes which have data which 
        # must be changed when introducing new nodes. Originally
        # for constant acceleration's flag list.
        #---------------------------------------------------------------

            for f in self.auxillaryFunctions:
                f()
                
        def convertConstantNodes(self):
        #---------------------------------------------------------------
        # takes the const bc nodes and makes into internal nodes.
        #---------------------------------------------------------------
            activeNodeLists = self.activeDatabase.nodeLists()
            
            for i in range(self.activeDatabase.numNodeLists):
                numFieldsi = len(self.activeFields[i])

                # set the ghost node of the const bcs up
                activeNodeLists[i].numGhostNodes=0
                self.constBC[i].setGhostNodes(activeNodeLists[i])
                for j in range(numFieldsi):
                    self.constBC[i].applyGhostBoundary(self.activeFields[i][j])
                
                # isRZ is depericate would need different 
                # implementation if RZ compatibility is desired
                #mass = activeNodeLists[i].mass()
                #pos  = activeNodeLists[i].positions()
                #if self.activeDatabase.isRZ:
                #    for k in self.constBC[i].nodeIndices:
                #        mass[k] *= 2.0*math.pi*pos[k].y

                #self.setConstantNodeDensity()
                

                # alloc room for new internal nodes
                activeNodeLists[i].numInternalNodes = activeNodeLists[i].numNodes 
                Ng = activeNodeLists[i].numGhostNodes

                # code ghost data to new internal slots
                fromIDs = []
                toIDs = []
                fromIDs.extend(range(activeNodeLists[i].firstGhostNode,activeNodeLists[i].numNodes,1))
                toIDs.extend(range(activeNodeLists[i].firstGhostNode-Ng,activeNodeLists[i].numNodes-Ng,1))
                for j in range(numFieldsi):
                    self.activeFields[i][j].copyElements(vector_of_int(fromIDs),vector_of_int(toIDs))

                # clear out the const bcs
                activeNodeLists[i].numGhostNodes = 0 
            
            return
            
        #---------------------------------------------------------------------
        def label(self):
            return "ExpandingDomain"
        #---------------------------------------------------------------------
        def dumpState(self, file, path):
            file.writeObject(self.activeZoneEllipse.abc, path + "/activeZone_abc")
            file.writeObject(self.constBCEllipse.abc, path + "/constBC_abc")
            file.writeObject(self.testZoneEllipse.abc, path + "/testZone_abc")

            file.writeObject(self.last_time, path + "/last_time")
            file.writeObject(self.last_dt, path + "/last_dt")
            file.writeObject(self.last_cycle, path + "/last_cycle")
            file.writeObject(self.nDeleted, path + "/nDeleted")
            return
        #---------------------------------------------------------------------
        def restoreState(self, file, path):
            self.activeZoneEllipse.abc = file.readObject(path+"/activeZone_abc")
            self.constBCEllipse.abc = file.readObject(path+"/constBC_abc")
            self.testZoneEllipse.abc = file.readObject(path+"/testZone_abc")

            self.last_time = file.readObject(path + "/last_time")
            self.last_dt = file.readObject(path + "/last_dt")
            self.last_cycle = file.readObject(path + "/last_cycle")
            self.nDeleted = file.readObject(path + "/nDeleted")

            # give a pass at the auxilliary functions since the
            # number of nodes will have likely changed from the
            # initial conditions of the script on restart
            for f in self.auxillaryFunctions:
                f()
            
            #stored values will be consistent w/ time zero and 
            #need to be clipped to the current inactive region
            for i in range(self.activeDatabase.numNodeLists):
                numInactiveNodesi = len(self.inactiveFields[i][0])
                numFieldsi = len(self.activeFields[i])
                indices_del = [k for k in range(numInactiveNodesi) if self.activeZoneEllipse.isInside(self.inactiveFields[i][0][k])]
                indices_del = sorted(indices_del, reverse=True)
                for j in range(numFieldsi):
                    for index in indices_del:
                        del self.inactiveFields[i][j][index]
                
            return
