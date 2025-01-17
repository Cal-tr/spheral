#include "GeomVector.hh"
#include "Utilities/SpheralFunctions.hh"

namespace Spheral {

//------------------------------------------------------------------------------
// Is the given point above, below, or colinear with the facet?
// Returns 1, -1, 0 respectively.
//------------------------------------------------------------------------------
inline
int
GeomFacet2d::
compare(const GeomFacet2d::Vector& point,
        const double tol) const {
  const double test = mNormal.dot(point - (*mVerticesPtr)[mPoints[0]]);
  return (fuzzyEqual(test, 0.0, tol) ?  0 :
          test > 0.0                 ?  1 :
                                       -1);
}

//------------------------------------------------------------------------------
// Access the points and normal.
//------------------------------------------------------------------------------
inline
const GeomFacet2d::Vector&
GeomFacet2d::
point1() const {
  REQUIRE(mVerticesPtr != 0 and
          mPoints[0] < mVerticesPtr->size());
  return (*mVerticesPtr)[mPoints[0]];
}

inline
const GeomFacet2d::Vector&
GeomFacet2d::
point2() const {
  REQUIRE(mVerticesPtr != 0 and
          mPoints[1] < mVerticesPtr->size());
  return (*mVerticesPtr)[mPoints[1]];
}

inline
unsigned
GeomFacet2d::
ipoint1() const {
  return mPoints[0];
}

inline
unsigned
GeomFacet2d::
ipoint2() const {
  return mPoints[1];
}

inline
const std::vector<unsigned>&
GeomFacet2d::
ipoints() const {
  return mPoints;
}

inline
const GeomFacet2d::Vector&
GeomFacet2d::
normal() const {
  return mNormal;
}

//------------------------------------------------------------------------------
// Compute the position.
//------------------------------------------------------------------------------
inline
GeomFacet2d::Vector
GeomFacet2d::
position() const {
  return 0.5*(point1() + point2());
}

//------------------------------------------------------------------------------
// Compute the area (line length in this case).
//------------------------------------------------------------------------------
inline
double
GeomFacet2d::
area() const {
  return (point2() - point1()).magnitude();
}

//------------------------------------------------------------------------------
// ==
//------------------------------------------------------------------------------
inline
bool
GeomFacet2d::
operator==(const GeomFacet2d& rhs) const {
  return (*mVerticesPtr == *(rhs.mVerticesPtr) and
          mPoints[0] == rhs.mPoints[0] and
          mPoints[1] == rhs.mPoints[1]);
}

//------------------------------------------------------------------------------
// !=
//------------------------------------------------------------------------------
inline
bool
GeomFacet2d::
operator!=(const GeomFacet2d& rhs) const {
  return not (*this == rhs);
}

//------------------------------------------------------------------------------
// Output (ostream) operator.
//------------------------------------------------------------------------------
inline
std::ostream&
operator<<(std::ostream& os, const GeomFacet2d& facet) {
  os << "GeomFacet2d( ivertices : " << facet.ipoint1() << " " << facet.ipoint2() << "\n"
     << "              vertices : " << facet.point1() << " " << facet.point2() << "\n"
     << "                normal : " << facet.normal() 
     << "\n)";
  return os;
}

}
