/*
   DVector        primRhs;     ///< rhs vector for computing the primal vector
   UpdateVector   primVec;     ///< primal vector
   Vector        dualRhs;     ///< rhs vector for computing the dual vector
   UpdateVector   dualVec;     ///< dual vector
   UpdateVector   addVec;      ///< storage for thePvec = &addVec

   Vector        theURbound;  ///< Upper Row    Feasibility bound
   Vector        theLRbound;  ///< Lower Row    Feasibility bound
   Vector        theUCbound;  ///< Upper Column Feasibility bound
   Vector        theLCbound;  ///< Lower Column Feasibility bound


Column Enter
------------
theFrhs              = &primRhs    = -A_N x_N
theFvec              = &primVec    = B^-1 theFrhs
theCoPrhs            = &dualRhs    = 
theCoPvec = theRPvec = &dualVec    = B^-1 theCoPrhs
thePvec   = theCPvec = &addVec     = dualVec * A
theUbound            = &theUCbound = SPxLP::upper() // Upper Column Feasibility bound
theLbound            = &theLCbound = SPxLP::lower() // Lower Column Feasibility bound
theCoUbound          = &theURbound = -lhs()         // Upper Row    Feasibility bound
theCoLbound          = &theLRbound = -rhs()         // Lower Row    Feasibility bound
theUBbound                         = -lhs()/upper()/infinity
theLBbound                         = -rhs()/lower()/-infinity


Column Leave
------------
theFrhs              = &primRhs   = -A_N x_N
theFvec              = &primVec   
theCoPrhs            = &dualRhs
theCoPvec = theRPvec = &dualVec 
thePvec   = theCPvec = &addVec     = dualVec * A
theUbound            = &theUCbound = +/-infinity/-maxobj // Upper Column Feasibility bound
theLbound            = &theLCbound = +/-infinity/-maxobj // Lower Column Feasibility bound
theCoUbound          = &theURbound = +/-infinity/0       // Upper Row    Feasibility bound
theCoLbound          = &theLRbound = +/-infinity/0       // Lower Row    Feasibility bound
theUBbound                         = 0/infinity/-lhs()/upper()
theLBbound                         = 0/-infinity/-rhs()/lower()

Row Enter
---------
theFrhs              = &dualRhs
theFvec              = &dualVec
theCoPrhs            = &primRhs
theCoPvec = theCPvec = &primVec
thePvec   = theRPvec = &addVec     = primVec * A
theUbound            = &theURbound = +/-infinity/0    // Upper Row    Feasibility bound 
theLbound            = &theLRbound = +/-infinity/0    // Lower Row    Feasibility bound
theCoUbound          = &theUCbound = +/-infinity/0    // Upper Column Feasibility bound
theCoLbound          = &theLCbound = +/-infinity/0    // Lower Column Feasibility bound
theUBbound                         = -lhs()/upper()/infinity
theLBbound                         = -rhs()/lower()/-infinity

Row Leave
---------
theFrhs              = &dualRhs    = maxObj()
theFvec              = &dualVec
theCoPrhs            = &primRhs
theCoPvec = theCPvec = &primVec
thePvec   = theRPvec = &addVec     = primVec * A
theUbound            = &theURbound = rhs()          // Upper Row    Feasibility bound 
theLbound            = &theLRbound = lhs()          // Lower Row    Feasibility bound
theCoUbound          = &theUCbound = SPxLP::upper() // Upper Column Feasibility bound
theCoLbound          = &theLCbound = SPxLP::lower() // Lower Column Feasibility bound
theUBbound                         = 0/infinity/-lhs()/upper()
theLBbound                         = 0/-infinity/-rhs()/lower()


// Column    Representation   Row
colset()    =  thevectors   = rowset()
rowset()    =  thecovectors = colset()
&primRhs    =  theFrhs      = &dualRhs
&primVec    =  theFvec      = &dualVec
&dualRhs    =  theCoPrhs    = &primRhs
&dualVec    =  theCoPvec    = &primVec
&addVec     =  thePvec      = &addVec
theCoPvec   =  theRPvec     = thePvec
thePVec     =  theCPvec     = theCoPvec
&theUCbound =  theUbound    = &theURbound
&theLCbound =  theLbound    = &theLRbound
&theURbound =  theCoUbound  = &theUCbound
&theLRbound =  theCoLbound  = &theLCbound
               theUBbound
               theLBbound


theLBbound  <  theFvec   = B^-1 theFrhs   < theUBbound
theLbound   <  thePvec   = A * theCoPvec  < theUbound
theCoLbound <  theCoPvec = B^-1 theCoPrhs < theCoUbound


$x_B = A_B^{-1} b - A_B^{-1} A_N x_N$


    In columnwise case, |theFvec| = $x_B = A_B^{-1} (- A_N x_N)$, where $x_N$
    are either upper or lower bounds for the nonbasic variables (depending on
    the variables |Status|). If these values remain unchanged throughout the
    simplex algorithm, they may be taken directly from LP. However, in the
    entering type algorith they are changed and, hence, retreived from the
    column or row upper or lower bound vectors.
 
    In rowwise case, |theFvec| = $\pi^T_B = (c^T - 0^T A_N) A_B^{-1}$. However,
    this applies only to leaving type algorithm, where no bounds on dual
    variables are altered. In entering type algorithm they are changed and,
    hence, retreived from the column or row upper or lower bound vectors.



theFvec = $x_B     = A_B^{-1} ( - A_N x_N$ ) // Column
        = $\pi^T_B = (c^T - 0^T A_N) A_B^{-1}$ // Row


theUCbound = upper
theLCbound = lower

// Column           Row
- lhs = theURbound = rhs
- rhs = theLRbound = lhs





*/














