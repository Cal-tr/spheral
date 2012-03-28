#----------------------------------Spheral++----------------------------------#-
#- newtonRaphson -- implements a simple version of the Newton-Raphson root
#- finding algorithm, combined with a bisection back-up.
#-
#- Based on rtsafe from Numerical recipes.
#- This is a direct port of our C++ version (src/src/Utilities/newtonRaphson.hh).
#-
#- Created by JMO, Thu Sep 23 22:57:33 PDT 2004
#-----------------------------------------------------------------------------#-
from SpheralTestUtilities import fuzzyEqual

def newtonRaphson(functor,
                  x1,
                  x2,
                  xaccuracy = 1.0e-15,
                  yaccuracy = 1.0e-15,
                  maxIterations = 100):

    #- Initialize values for the function and it's derivative.
    xminValues = functor(x1);
    xmaxValues = functor(x2);

    #- Make sure the root is bracketed by the input range.
    if xminValues[0] * xmaxValues[0] > 0.0:
        raise "newtonRaphson ERROR: root must be bracketed by input range"

    #- Is the root already at the min or max range?
    if fuzzyEqual(xminValues[0], 0.0, yaccuracy):
        return x1
    if fuzzyEqual(xmaxValues[0], 0.0, yaccuracy):
        return x2

    #- Initialize the searching parameters.
    xl = 0.0
    xh = 0.0
    if xminValues[0] < 0.0:
        xl = x1
        xh = x2
    else:
        assert xminValues[0] > 0.0 and xmaxValues[0] < 0.0
        xl = x2;
        xh = x1;
    rootSafe = 0.5*(x1 + x2)
    dxold = abs(x2 - x1)
    dx = dxold
    fdf = functor(rootSafe)
    f = fdf[0]
    df = fdf[1]

    #- Iterate until we either converge or achieve the desired accuracy.
    iter = 0
    while iter < maxIterations:
        iter += 1

        #- Bisect if Newton out of range or not decreasing fast enough.
        if (((rootSafe - xh)*df - f)*((rootSafe - xl)*df - f) > 0.0 or
            abs(2.0*f) > abs(dxold*df)):
            dxold = dx
            dx = 0.5*(xh - xl)
            rootSafe = xl + dx
            if (fuzzyEqual(xl, rootSafe, xaccuracy)):
                return rootSafe

        else:
            #- Take a Newton-Raphson step.
            assert not fuzzyEqual(df, 0.0)
            dxold = dx
            dx = f/df
            tmp = rootSafe
            rootSafe -= dx
            if fuzzyEqual(tmp, rootSafe, xaccuracy):
                return rootSafe

        if abs(dx) <= xaccuracy:
            return rootSafe

        fdf = functor(rootSafe)
        f = fdf[0]
        df = fdf[1]
        if f < 0.0:
            xl = rootSafe
        else:
            xh = rootSafe

    raise "newtonRaphson ERROR: did not converge!"
