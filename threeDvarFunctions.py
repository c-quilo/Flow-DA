# Definition of gradient of J
def gradJ(v):
    import numpy as np
    Vv = np.dot(V, v)
    Jmis = np.subtract(Vv, d)
    g1 = Jmis.copy()
    VT = np.transpose(V)
    g2 = np.dot(VT, g1)
    gg2 = np.multiply(invR, g2)
    ggJ = v + gg2
    return ggJ

def PrepareForDA(ug, V):
    import numpy as np

    ##ug.GetFieldNames()
    ##uvwVec = ug.GetScalarField('Tracer')
    uvwVec = ug
    n = len(uvwVec)
    #m = trnc

    xB = uvwVec.copy()
    x0 = np.ones(n)

    Vin = np.linalg.pinv(V)
    v0 = np.dot(Vin, x0)
    HxB = np.copy(xB)
    d = np.subtract(y, HxB)

    return v0, n, xB, d, uvwVec

# Compute the minimum of cost function J

def ThreeD_VAR(V, v0, n, xB, d):
    import numpy as np

    res = minimize(J, v0, method='L-BFGS-B', jac=gradJ, options={'disp': False})

    vDA = np.array([])
    vDA = res.x
    deltaXDA = np.dot(V, vDA)
    xDA = xB + deltaXDA

    return xDA

# Definition of cost function J
def J(v):
    import numpy as np
    vT = np.transpose(v)
    vTv = np.dot(vT, v)
    Vv = np.dot(V, v)
    Jmis = np.subtract(Vv, d)
    JmisT = np.transpose(Jmis)
    RJmis = JmisT.copy()
    J1 = invR * np.dot(Jmis, RJmis)
    Jv = (vTv + J1) / 2
    return Jv
