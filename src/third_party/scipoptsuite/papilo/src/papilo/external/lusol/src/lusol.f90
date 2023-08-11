!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! File:       lusol.f90
!
! Contains    lu1fac   lu1fad   lu1gau   lu1mar   lu1mRP   lu1mCP   lu1mSP
! Contains    lu1pen   lu1mxc   lu1mxr   lu1or1   lu1or2   lu1or3   lu1or4
! Contains    lu1pq1   lu1pq2   lu1pq3   lu1rec   lu1slk
!             lu1ful   lu1DPP   lu1DCP
! Contains    Hbuild   Hchange  Hdelete  Hdown    Hinsert  Hup
! Contains    lu6sol   lu6L     lu6Lt    lu6U     Lu6Ut    lu6LD    lu6chk
! Contains    lu7add   lu7cyc   lu7elm   lu7for   lu7rnk   lu7zap
! Contains    lu8rpc
!
! Contains    jdamax
!
! This file is an f90 version of most parts of the f77 sparse LU package LUSOL
! (the parts needed by MINOS, SQOPT and SNOPT).  The parts included are
!
!    lusol1.f    Factor a given matrix A from scratch (lu1fac).
!    lusol2.f    Heap-management routines for lu1fac.
!    lusol6a.f   Solve with the current LU factors.
!    lusol7a.f   Utilities for all update routines.
!    lusol8a.f   Replace a column (Bartels-Golub update).
!
! 10 Jan 2010: First f90 version.
! 12 Dec 2011: Had to change ip, iq to p, q to avoid clash with ip, rp.
! 17 Dec 2011: BLAS idamax replaced by private jdamax (taken from sn17util.f90).
!              Note: jdamax( lencol, a(k,k), 1 ) has to become
!                    jdamax( lencol,a(k:m,k),1 )
! 03 Feb 2012: It's ok to have a(k,k) above, but a(k:m) is more illuminating.
! 03 Feb 2012: Bug fixed in lu1DPP and lu1DCP (translation of call daxpy).
! 09 Mar 2013: Begin project for improving efficiency of TRP.
!              Mostly this needs a new version of lu1mxr.
!              Ding Ma and Michael Saunders, Stanford University.
! 03 Apr 2013: New lu1mxr finds max element Amaxr(i) in each modified row i
!              much more efficiently.  Three new local arrays needed:
!              markc(n) and markr(m) in lu1fad, and cols(n) in lu1mxr.
!              This is easy in f90.
! 28 Sep 2015: lu1fad: Change 2 * lenD to 3 * lenD for safety.
! 13 Nov 2015: lu6chk: Remove resetting of Utol1 for TRP
!              to prevent slacks replacing slacks when DUmax is big.
! 12 Dec 2015: lu1slk called before lu1fad to set nslack.
!              lu1fad grabs slacks first during Utri.
! 13 Dec 2015: lu1mxc now handles empty columns correctly.
! 20 Dec 2015: lu1rec returns ilast as output parameter.
! 21 Dec 2015: lu1DCP exits if aijmax <= small.
! 20 Jan 2016: sn28lusol.f90 updated to match sn27lu.f of 21 Dec 2015.
! 25 Jan 2016: Module snConstants replaced by local zero, one, i1.
! 27 Jan 2016: i2 is another local constant.
!              NOTE: Local function jdamax is like a BLAS routine,
!              but no BLAS routines are used in this f90 version of LUSOL.
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

module lusol
  use  lusol_precision,        only : ip, rp

  implicit none
  private
  public    :: lu1fac, lu6sol, lu8rpc
  private   :: jdamax
  intrinsic :: abs, int, max, min, real

  integer(ip),  parameter :: i1   = 1,   i2  = 2
  real(rp),     parameter :: zero = 0.0, one = 1.0

contains

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ! History from lusol1.f
  !
  ! 26 Apr 2002: TCP implemented using heap data structure.
  ! 01 May 2002: lu1DCP implemented.
  ! 07 May 2002: lu1mxc must put 0.0 at top of empty columns.
  ! 09 May 2002: lu1mCP implements Markowitz with cols searched
  !              in heap order.
  !              Often faster (searching 20 or 40 cols) but more dense.
  ! 11 Jun 2002: TRP implemented.
  !              lu1mRP implements Markowitz with Threshold Rook Pivoting.
  !              lu1mxc maintains max col elements.  (Previously lu1max.)
  !              lu1mxr maintains max row elements.
  ! 12 Jun 2002: lu1mCP seems too slow on big problems (e.g. memplus).
  !              Disabled it for the moment.  (Use lu1mar + TCP.)
  ! 14 Dec 2002: TSP implemented.
  !              lu1mSP implements Markowitz with
  !              Threshold Symmetric Pivoting.
  ! 07 Mar 2003: character*1, character*2 changed to f90 form.
  !              Comments changed from * in column to ! in column 1.
  !              Comments kept within column 72 to avoid compiler warning.
  ! 19 Dec 2004: Hdelete(...) has new input argument Hlenin.
  ! 21 Dec 2004: Print Ltol and Lmax with e10.2 instead of e10.1.
  ! 26 Mar 2006: lu1fad: Ignore nsing from lu1ful.
  !              lu1DPP: nsing redefined (but not used by lu1fad).
  !              lu1DCP: nsing redefined (but not used by lu1fad).
  ! 13 Dec 2015: lu1mxc bug on empty cols (setting a(lc) = 0.0).
  !              TRO11X3 starts with col 57 containing two nonzeros = 1e-18.
  !              col 58 is already empty, so col 59 (a slack -1.0) got incorrectly
  !              changed to 0.0.  This explains Matlab error on data with empty cols.
  !              lu1mxc fixed.  TRP and TCP ok now on TRO11X3.
  ! 20 Jan 2016: Current version of lusol1.f90.
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1fac( m    , n    , nelem, lena , luparm, parmlu,       &
                     a    , indc , indr , p    , q     ,               &
                     lenc , lenr , locc , locr ,                       &
                     iploc, iqloc, ipinv, iqinv, w     , inform )

    integer(ip),   intent(in)    :: m, n, nelem, lena

    integer(ip),   intent(inout) :: luparm(30)
    integer(ip),   intent(inout) :: indc(lena), indr(lena),            &
                                    p(m)      , q(n)      ,            &
                                    lenc(n)   , lenr(m)   ,            &
                                    iploc(n)  , iqloc(m)  ,            &
                                    ipinv(m)  , iqinv(n)  ,            &
                                    locc(n)   , locr(m)
    real(rp),      intent(inout) :: parmlu(30), a(lena), w(n)

    integer(ip),   intent(out)   :: inform

    !------------------------------------------------------------------
    ! lu1fac computes a factorization A = L*U, where A is a sparse
    ! matrix with m rows and n columns, P*L*P' is lower triangular
    ! and P*U*Q is upper triangular for certain permutations P, Q
    ! (which are returned in the arrays p, q).
    ! Stability is ensured by limiting the size of the elements of L.
    !
    ! The nonzeros of A are input via the parallel arrays a, indc, indr,
    ! which should contain nelem entries of the form    aij,    i,    j
    ! in any order.  There should be no duplicate pairs         i,    j.
    !
    ! ******************************************************************
    ! *        Beware !!!   The row indices i must be in indc,         *
    ! *              and the column indices j must be in indr.         *
    ! *              (Not the other way round!)                        *
    ! ******************************************************************
    !
    ! It does not matter if some of the entries in a(*) are zero.
    ! Entries satisfying  abs( a(i) ) .le. parmlu(3)  are ignored.
    ! Other parameters in luparm and parmlu are described below.
    !
    ! The matrix A may be singular.  On exit, nsing = luparm(11) gives
    ! the number of apparent singularities.  This is the number of
    ! "small" diagonals of the permuted factor U, as judged by
    ! the input tolerances Utol1 = parmlu(4) and  Utol2 = parmlu(5).
    ! The diagonal element diagj associated with column j of A is
    ! "small" if
    !                 abs( diagj ) .le. Utol1
    ! or
    !                 abs( diagj ) .le. Utol2 * max( uj ),
    !
    ! where max( uj ) is the maximum element in the j-th column of U.
    ! The position of such elements is returned in w(*).  In general,
    ! w(j) = + max( uj ),  but if column j is a singularity,
    ! w(j) = - max( uj ).  Thus, w(j) .le. 0 if column j appears to be
    ! dependent on the other columns of A.
    !
    ! NOTE: lu1fac (like certain other sparse LU packages) does not
    ! treat dense columns efficiently.  This means it will be slow
    ! on "arrow matrices" of the form
    !                  A = (x       a)
    !                      (  x     b)
    !                      (    x   c)
    !                      (      x d)
    !                      (x x x x e)
    ! if the numerical values in the dense column allow it to be
    ! chosen LATE in the pivot order.
    !
    ! With TPP (Threshold Partial Pivoting), the dense column is
    ! likely to be chosen late.
    !
    ! With TCP (Threshold Complete Pivoting), if any of a,b,c,d
    ! is significantly larger than other elements of A, it will
    ! be chosen as the first pivot and the dense column will be
    ! eliminated, giving reasonably sparse factors.
    ! However, if element e is so big that TCP chooses it, the factors
    ! will become dense.  (It's hard to win on these examples!)
    !==================================================================
    !
    !
    ! Notes on the array names
    ! ------------------------
    !
    ! During the LU factorization, the sparsity pattern of the matrix
    ! being factored is stored twice: in a column list and a row list.
    !
    ! The column list is ( a, indc, locc, lenc )
    ! where
    !       a(*)    holds the nonzeros,
    !       indc(*) holds the indices for the column list,
    !       locc(j) points to the start of column j in a(*) and indc(*),
    !       lenc(j) is the number of nonzeros in column j.
    !
    ! The row list is    (    indr, locr, lenr )
    ! where
    !       indr(*) holds the indices for the row list,
    !       locr(i) points to the start of row i in indr(*),
    !       lenr(i) is the number of nonzeros in row i.
    !
    !
    ! At all stages of the LU factorization, p contains a complete
    ! row permutation.  At the start of stage k,  p(1), ..., p(k-1)
    ! are the first k-1 rows of the final row permutation P.
    ! The remaining rows are stored in an ordered list
    !                    ( p, iploc, ipinv )
    ! where
    !       iploc(nz) points to the start in p(*) of the set of rows
    !                 that currently contain nz nonzeros,
    !       ipinv(i)  points to the position of row i in p(*).
    !
    ! For example,
    !       iploc(1) = k   (and this is where rows of length 1 begin),
    !       iploc(2) = k+p  if there are p rows of length 1
    !                      (and this is where rows of length 2 begin).
    !
    ! Similarly for q, iqloc, iqinv.
    !==================================================================
    !
    !
    ! 00 Jun 1983  Original version.
    ! 00 Jul 1987  nrank  saved in luparm(16).
    ! 12 Apr 1989  ipinv, iqinv added as workspace.
    ! 26 Apr 1989  maxtie replaced by maxcol in Markowitz search.
    ! 16 Mar 1992  jumin  saved in luparm(19).
    ! 10 Jun 1992  lu1fad has to move empty rows and cols to the bottom
    !              (via lu1pq3) before doing the dense LU.
    ! 12 Jun 1992  Deleted dense LU (lu1ful, lu1vlu).
    ! 25 Oct 1993  keepLU implemented.
    ! 07 Feb 1994  Added new dense LU (lu1ful, lu1den).
    ! 21 Dec 1994  Bugs fixed in lu1fad (nrank) and lu1ful (ipvt).
    ! 08 Aug 1995  Use p instead of w as parameter to lu1or3 (for F90).
    ! 13 Sep 2000  TPP and TCP options implemented.
    ! 17 Oct 2000  Fixed troubles due to A = empty matrix (Todd Munson).
    ! 01 Dec 2000  Save Lmax, Umax, etc. after both lu1fad and lu6chk.
    !              lu1fad sets them when keepLU = false.
    !              lu6chk sets them otherwise, and includes items
    !              from the dense LU.
    ! 11 Mar 2001  lu6chk now looks at diag(U) when keepLU = false.
    ! 26 Apr 2002  New TCP implementation using heap routines to
    !              store largest element in each column.
    !              New workspace arrays Ha, Hj, Hk required.
    !              For compatibility, borrow space from a, indc, indr
    !              rather than adding new input parameters.
    ! 01 May 2002  lu1den changed to lu1DPP (dense partial  pivoting).
    !              lu1DCP implemented       (dense complete pivoting).
    !              Both TPP and TCP now switch to dense mode and end.
    !
    ! 10 Jan 2010: First f90 version.
    !---------------------------------------------------------------------
    !
    !
    !  INPUT PARAMETERS
    !
    !  m      (not altered) is the number of rows in A.
    !  n      (not altered) is the number of columns in A.
    !  nelem  (not altered) is the number of matrix entries given in
    !         the arrays a, indc, indr.
    !  lena   (not altered) is the dimension of  a, indc, indr.
    !         This should be significantly larger than nelem.
    !         Typically one should have
    !            lena > max( 2*nelem, 10*m, 10*n, 10000 )
    !         but some applications may need more.
    !         On machines with virtual memory it is safe to have
    !         lena "far bigger than necessary", since not all of the
    !         arrays will be used.
    !  a      (overwritten) contains entries   Aij  in   a(1:nelem).
    !  indc   (overwritten) contains the indices i in indc(1:nelem).
    !  indr   (overwritten) contains the indices j in indr(1:nelem).
    !
    !  luparm input parameters:                                Typical value
    !
    !  luparm( 1) = nout     File number for printed messages.         6
    !
    !  luparm( 2) = lprint   Print level.                              0
    !                   <  0 suppresses output.
    !                   =  0 gives error messages.
    !                  >= 10 gives statistics about the LU factors.
    !                  >= 50 gives debug output from lu1fac
    !                        (the pivot row and column and the
    !                        no. of rows and columns involved at
    !                        each elimination step).
    !
    !  luparm( 3) = maxcol   lu1fac: maximum number of columns         5
    !                        searched allowed in a Markowitz-type
    !                        search for the next pivot element.
    !                        For some of the factorization, the
    !                        number of rows searched is
    !                        maxrow = maxcol - 1.
    !
    !  luparm( 6) = 0    =>  TPP: Threshold Partial   Pivoting.        0
    !             = 1    =>  TRP: Threshold Rook      Pivoting.
    !             = 2    =>  TCP: Threshold Complete  Pivoting.
    !             = 3    =>  TSP: Threshold Symmetric Pivoting.
    !             = 4    =>  TDP: Threshold Diagonal  Pivoting.
    !                             (TDP not yet implemented).
    !                        TRP and TCP are more expensive than TPP but
    !                        more stable and better at revealing rank.
    !                        Take care with setting parmlu(1), especially
    !                        with TCP.
    !                        NOTE: TSP and TDP are for symmetric matrices
    !                        that are either definite or quasi-definite.
    !                        TSP is effectively TRP for symmetric matrices.
    !                        TDP is effectively TCP for symmetric matrices.
    !
    !  luparm( 8) = keepLU   lu1fac: keepLU = 1 means the numerical    1
    !                        factors will be computed if possible.
    !                        keepLU = 0 means L and U will be discarded
    !                        but other information such as the row and
    !                        column permutations will be returned.
    !                        The latter option requires less storage.
    !
    !  parmlu input parameters:                                Typical value
    !
    !  parmlu( 1) = Ltol1    Max Lij allowed during Factor.
    !                                                  TPP     10.0 or 100.0
    !                                                  TRP      4.0 or  10.0
    !                                                  TCP      5.0 or  10.0
    !                                                  TSP      4.0 or  10.0
    !                        With TRP and TCP (Rook and Complete Pivoting),
    !                        values less than 25.0 may be expensive
    !                        on badly scaled data.  However,
    !                        values less than 10.0 may be needed
    !                        to obtain a reliable rank-revealing
    !                        factorization.
    !  parmlu( 2) = Ltol2    Max Lij allowed during Updates.            10.0
    !                        during updates.
    !  parmlu( 3) = small    Absolute tolerance for       eps**0.8 = 3.0d-13
    !                        treating reals as zero.
    !  parmlu( 4) = Utol1    Absolute tol for flagging    eps**0.67= 3.7d-11
    !                        small diagonals of U.
    !  parmlu( 5) = Utol2    Relative tol for flagging    eps**0.67= 3.7d-11
    !                        small diagonals of U.
    !                        (eps = machine precision)
    !  parmlu( 6) = Uspace   Factor limiting waste space in  U.      3.0
    !                        In lu1fac, the row or column lists
    !                        are compressed if their length
    !                        exceeds Uspace times the length of
    !                        either file after the last compression.
    !  parmlu( 7) = dens1    The density at which the Markowitz      0.3
    !                        pivot strategy should search maxcol
    !                        columns and no rows.
    !                        (Use 0.3 unless you are experimenting
    !                        with the pivot strategy.)
    !  parmlu( 8) = dens2    the density at which the Markowitz      0.5
    !                        strategy should search only 1 column,
    !                        or (if storage is available)
    !                        the density at which all remaining
    !                        rows and columns will be processed
    !                        by a dense LU code.
    !                        For example, if dens2 = 0.1 and lena is
    !                        large enough, a dense LU will be used
    !                        once more than 10 per cent of the
    !                        remaining matrix is nonzero.
    !
    !
    !  OUTPUT PARAMETERS
    !
    !  a, indc, indr     contain the nonzero entries in the LU factors of A.
    !         If keepLU = 1, they are in a form suitable for use
    !         by other parts of the LUSOL package, such as lu6sol.
    !         U is stored by rows at the start of a, indr.
    !         L is stored by cols at the end   of a, indc.
    !         If keepLU = 0, only the diagonals of U are stored, at the
    !         end of a.
    !  p, q   are the row and column permutations defining the
    !         pivot order.  For example, row p(1) and column q(1)
    !         defines the first diagonal of U.
    !  lenc(1:numl0) contains the number of entries in nontrivial
    !         columns of L (in pivot order).
    !  lenr(1:m) contains the number of entries in each row of U
    !         (in original order).
    !  locc(1:n) = 0 (ready for the LU update routines).
    !  locr(1:m) points to the beginning of the rows of U in a, indr.
    !  iploc, iqloc, ipinv, iqinv  are undefined.
    !  w      indicates singularity as described above.
    !  inform = 0 if the LU factors were obtained successfully.
    !         = 1 if U appears to be singular, as judged by lu6chk.
    !         = 3 if some index pair indc(l), indr(l) lies outside
    !             the matrix dimensions 1:m , 1:n.
    !         = 4 if some index pair indc(l), indr(l) duplicates
    !             another such pair.
    !         = 7 if the arrays a, indc, indr were not large enough.
    !             Their length "lena" should be increase to at least
    !             the value "minlen" given in luparm(13).
    !         = 8 if there was some other fatal error.  (Shouldn't happen!)
    !         = 9 if no diagonal pivot could be found with TSP or TDP.
    !             The matrix must not be sufficiently definite
    !             or quasi-definite.
    !         =10 if there was some other fatal error.
    !
    !  luparm output parameters:
    !
    !  luparm(10) = inform   Return code from last call to any LU routine.
    !  luparm(11) = nsing    No. of singularities marked in the
    !                        output array w(*).
    !  luparm(12) = jsing    Column index of last singularity.
    !  luparm(13) = minlen   Minimum recommended value for  lena.
    !  luparm(14) = maxlen   ?
    !  luparm(15) = nupdat   No. of updates performed by the lu8 routines.
    !  luparm(16) = nrank    No. of nonempty rows of U.
    !  luparm(17) = ndens1   No. of columns remaining when the density of
    !                        the matrix being factorized reached dens1.
    !  luparm(18) = ndens2   No. of columns remaining when the density of
    !                        the matrix being factorized reached dens2.
    !  luparm(19) = jumin    The column index associated with DUmin.
    !  luparm(20) = numL0    No. of columns in initial  L.
    !  luparm(21) = lenL0    Size of initial  L  (no. of nonzeros).
    !  luparm(22) = lenU0    Size of initial  U.
    !  luparm(23) = lenL     Size of current  L.
    !  luparm(24) = lenU     Size of current  U.
    !  luparm(25) = lrow     Length of row file.
    !  luparm(26) = ncp      No. of compressions of LU data structures.
    !  luparm(27) = mersum   lu1fac: sum of Markowitz merit counts.
    !  luparm(28) = nUtri    lu1fac: triangular rows in U.
    !  luparm(29) = nLtri    lu1fac: triangular rows in L.
    !  luparm(30) = nslack   lu1fac: no. of unit vectors at start of U. (info only)
    !
    !
    !
    !  parmlu output parameters:
    !
    !  parmlu(10) = Amax     Maximum element in  A.
    !  parmlu(11) = Lmax     Maximum multiplier in current  L.
    !  parmlu(12) = Umax     Maximum element in current  U.
    !  parmlu(13) = DUmax    Maximum diagonal in  U.
    !  parmlu(14) = DUmin    Minimum diagonal in  U.
    !  parmlu(15) = Akmax    Maximum element generated at any stage
    !                        during TCP factorization.
    !  parmlu(16) = growth   TPP: Umax/Amax    TRP, TCP, TSP: Akmax/Amax
    !  parmlu(17) =
    !  parmlu(18) =
    !  parmlu(19) =
    !  parmlu(20) = resid    lu6sol: residual after solve with U or U'.
    !  ...
    !  parmlu(30) =
    !---------------------------------------------------------------------

    character(1)           :: mnkey
    character(2)           :: kPiv(0:3)
    integer(ip)            :: i, idummy, j, jsing, jumin,              &
                              k, l, l2, lena2, lenH, lenL,             &
                              lenLk, lenU, lenUk, lerr,                &
                              ll, llsave, lm, lmaxr, locH,             &
                              lprint, lPiv, lrow, ltopl,               &
                              lu, mersum, minlen, nbump,               &
                              ncp, ndens1, ndens2,                     &
                              nLtri, nmove, nout, nrank,               &
                              nsing, numl0, numnz, nslack, nUtri
    logical                :: keepLU, TCP, TPP, TRP, TSP
    real(rp)               :: Agrwth, Akmax, Amax, avgmer,             &
                              condU, delem, densty, dincr,             &
                              dm, dn, DUmax, DUmin, growth,            &
                              Lmax, Ltol, small, Ugrwth,               &
                              Umax

    ! Grab relevant input parameters.

    nout   = luparm(1)
    lprint = luparm(2)
    lPiv   = luparm(6)
    keepLU = luparm(8) /= 0

    Ltol   = parmlu(1)  ! Limit on size of Lij
    small  = parmlu(3)  ! Drop tolerance

    TPP    = lPiv == 0  ! Threshold Partial   Pivoting (normal).
    TRP    = lPiv == 1  ! Threshold Rook      Pivoting
    TCP    = lPiv == 2  ! Threshold Complete  Pivoting.
    TSP    = lPiv == 3  ! Threshold Symmetric Pivoting.
    kPiv(0)= 'PP'
    kPiv(1)= 'RP'
    kPiv(2)= 'CP'
    kPiv(3)= 'SP'

    ! Initialize output parameters.

    inform = 0
    minlen = nelem + 2*(m + n)
    numl0  = 0
    lenL   = 0
    lenU   = 0
    lrow   = 0
    mersum = 0
    nUtri  = m
    nLtri  = 0
    ndens1 = 0
    ndens2 = 0
    nrank  = 0
    nsing  = 0
    jsing  = 0
    jumin  = 0
    nslack = 0

    Amax   = zero
    Lmax   = zero
    Umax   = zero
    DUmax  = zero
    DUmin  = zero
    Akmax  = zero

    if (m > n) then
        mnkey  = '>'
    else if (m == n) then
        mnkey  = '='
    else
        mnkey  = '<'
    end if

    ! Float version of dimensions.

    dm     = m
    dn     = n
    delem  = nelem

    ! Initialize workspace parameters.

    luparm(26) = 0             ! ncp
    if (lena < minlen) go to 970

    !-------------------------------------------------------------------
    ! Organize the  aij's  in  a, indc, indr.
    ! lu1or1  deletes small entries, tests for illegal  i,j's,
    !         and counts the nonzeros in each row and column.
    ! lu1or2  reorders the elements of  A  by columns.
    ! lu1or3  uses the column list to test for duplicate entries
    !         (same indices  i,j).
    ! lu1or4  constructs a row list from the column list.
    !-------------------------------------------------------------------
    call lu1or1( m   , n    , nelem, lena , small,                     &
                 a   , indc , indr , lenc , lenr,                      &
                 Amax, numnz, lerr , inform )

    if (nout > 0  .and.  lprint >= 10) then
       densty = 100.0_rp * delem / (dm * dn)
       write(nout, 1000) m, mnkey, n, numnz, Amax, densty
    end if
    if (inform /= 0) go to 930

!!! nelem  = numnz     !!! Don't change nelem.
!!! nelem is now numnz below (it might be less than the input value).

    call lu1or2( n, numnz, lena, a, indc, indr, lenc, locc )
    call lu1or3( m, n, lena, indc, lenc, locc, p, lerr, inform )

    if (inform /= 0) go to 940

    call lu1or4( m, n, numnz, lena, indc, indr, lenc, lenr, locc, locr )

    !------------------------------------------------------------------
    ! Set up lists of rows and columns with equal numbers of nonzeros,
    ! using  indc(*)  as workspace.
    ! 12 Dec 2015: Always call lu1slk here now.
    ! This sets nslack and w(j) = 1.0 for slacks, else 0.0.
    !------------------------------------------------------------------
    call lu1pq1( m, n, lenr, p, iploc, ipinv, indc(numnz + 1) )
    call lu1pq1( n, m, lenc, q, iqloc, iqinv, indc(numnz + 1) )
    call lu1slk( m, n, lena, q, iqloc, a, indc, locc, nslack, w )
    luparm(30) = nslack

    !------------------------------------------------------------------
    ! For TCP, allocate Ha, Hj, Hk at the end of a, indc, indr.
    ! Then compute the factorization  A = L*U.
    !------------------------------------------------------------------
    lenH   = 0                ! Keep -Wmaybe-uninitialized happy.
    lena2  = 0                !
    locH   = 0                !
    lmaxr  = 0                !
    if (TPP .or. TSP) then
       lenH   = 1
       lena2  = lena
       locH   = lena
       lmaxr  = 1
    else if (TRP) then
       lenH   = 1             ! Dummy
       lena2  = lena  - m     ! Reduced length of      a
       locH   = lena          ! Dummy
       lmaxr  = lena2 + 1     ! Start of Amaxr      in a
    else if (TCP) then
       lenH   = n             ! Length of heap
       lena2  = lena  - lenH  ! Reduced length of      a, indc, indr
       locH   = lena2 + 1     ! Start of Ha, Hj, Hk in a, indc, indr
       lmaxr  = 1             ! Dummy
    end if

    call lu1fad( m     , n     , numnz , lena2 , luparm, parmlu,       &
                 a     , indc  , indr  , p     , q     ,               &
                 lenc  , lenr  , locc  , locr  ,                       &
                 iploc , iqloc , ipinv , iqinv , w     ,               &
                 lenH  ,a(locH), indc(locH), indr(locH), a(lmaxr),     &
                 inform, lenL  , lenU  , minlen, mersum,               &
                 nUtri , nLtri , ndens1, ndens2, nrank , nslack,       &
                 Lmax  , Umax  , DUmax , DUmin , Akmax )

    luparm(16) = nrank
    luparm(23) = lenL
    if (inform == 7) go to 970
    if (inform == 9) go to 985
    if (inform ==10) go to 981
    if (inform >  0) go to 980

    if ( keepLU ) then
       !---------------------------------------------------------------
       ! The LU factors are at the top of  a, indc, indr,
       ! with the columns of  L  and the rows of  U  in the order
       !
       ! ( free )   ... ( u3 ) ( l3 ) ( u2 ) ( l2 ) ( u1 ) ( l1 ).
       !
       ! Starting with ( l1 ) and ( u1 ), move the rows of  U  to the
       ! left and the columns of  L  to the right, giving
       !
       ! ( u1 ) ( u2 ) ( u3 ) ...   ( free )   ... ( l3 ) ( l2 ) ( l1 ).
       !
       ! Also, set  numl0 = the number of nonempty columns of L.
       !---------------------------------------------------------------
       lu     = 0
       ll     = lena  + 1
       lm     = lena2 + 1
       ltopl  = ll - lenL - lenU
       lrow   = lenU

       do k = 1, nrank
          i       =   p(k)
          lenUk   = - lenr(i)
          lenr(i) =   lenUk
          j       =   q(k)
          lenLk   = - lenc(j) - 1
          if (lenLk > 0) then
             numl0        = numl0 + 1
             iqloc(numl0) = lenLk
          end if

          if (lu + lenUk < ltopl) then
             !=========================================================
             ! There is room to move ( uk ).  Just right-shift ( lk ).
             !=========================================================
             do idummy = 1, lenLk
                ll       = ll - 1
                lm       = lm - 1
                a(ll)    = a(lm)
                indc(ll) = indc(lm)
                indr(ll) = indr(lm)
             end do
          else
             !=========================================================
             ! There is no room for ( uk ) yet.  We have to
             ! right-shift the whole of the remaining LU file.
             ! Note that ( lk ) ends up in the correct place.
             !=========================================================
             llsave = ll - lenLk
             nmove  = lm - ltopl

             do idummy = 1, nmove
                ll       = ll - 1
                lm       = lm - 1
                a(ll)    = a(lm)
                indc(ll) = indc(lm)
                indr(ll) = indr(lm)
             end do

             ltopl  = ll
             ll     = llsave
             lm     = ll
          end if

          !======================================================
          ! Left-shift ( uk ).
          !======================================================
          locr(i) = lu + 1
          l2      = lm - 1
          lm      = lm - lenUk

          do l = lm, l2
             lu       = lu + 1
             a(lu)    = a(l)
             indr(lu) = indr(l)
          end do
       end do

       !---------------------------------------------------------------
       ! Save the lengths of the nonempty columns of  L,
       ! and initialize  locc(j)  for the LU update routines.
       !---------------------------------------------------------------
       lenc(1:numl0) = iqloc(1:numl0)
       locc(1:n)     = 0

       !---------------------------------------------------------------
       ! Test for singularity.
       ! lu6chk  sets  nsing, jsing, jumin, Lmax, Umax, DUmax, DUmin
       ! (including entries from the dense LU).
       ! input      i1 = 1 means we're calling lu6chk from LUSOL.
       ! output inform = 1 if there are singularities (nsing > 0).
       ! 12 Dec 2015: nslack is now an input.
       !---------------------------------------------------------------
       call lu6chk( i1, m, n, nslack, w, lena, luparm, parmlu,         &
                    a, indc, indr, p, q,                               &
                    lenc, lenr, locc, locr, inform )
       nsing  = luparm(11)
       jsing  = luparm(12)
       jumin  = luparm(19)
       Lmax   = parmlu(11)
       Umax   = parmlu(12)
       DUmax  = parmlu(13)
       DUmin  = parmlu(14)

    else
       !---------------------------------------------------------------
       ! keepLU = 0.  L and U were not kept, just the diagonals of U.
       ! lu1fac will probably be called again soon with keepLU = .true.
       ! 11 Mar 2001: lu6chk revised.  We can call it with keepLU = 0,
       !              but we want to keep Lmax, Umax from lu1fad.
       ! 05 May 2002: Allow for TCP with new lu1DCP.  Diag(U) starts
       !              below lena2, not lena.  Need lena2 in next line.
       ! 12 Dec 2015: nslack is now an input.
       !---------------------------------------------------------------
       call lu6chk( i1, m, n, nslack, w, lena2, luparm, parmlu,        &
                    a, indc, indr, p, q,                               &
                    lenc, lenr, locc, locr, inform )
       nsing  = luparm(11)
       jsing  = luparm(12)
       jumin  = luparm(19)
       DUmax  = parmlu(13)
       DUmin  = parmlu(14)
    end if

    go to 990

    !------------
    ! Error exits.
    !------------
930 inform = 3
    if (lprint >= 0) write(nout, 1300) lerr, indc(lerr), indr(lerr)
    go to 990

940 inform = 4
    if (lprint >= 0) write(nout, 1400) lerr, indc(lerr), indr(lerr)
    go to 990

970 inform = 7
    if (lprint >= 0) write(nout, 1700) lena, minlen
    go to 990

980 inform = 8
    if (lprint >= 0) write(nout, 1800)
    go to 990

981 inform = 10
    go to 990

985 inform = 9
    if (lprint >= 0) write(nout, 1900)

    ! Store output parameters.

990 luparm(10) = inform
    luparm(11) = nsing
    luparm(12) = jsing
    luparm(13) = minlen
    luparm(15) = 0
    luparm(16) = nrank
    luparm(17) = ndens1
    luparm(18) = ndens2
    luparm(19) = jumin
    luparm(20) = numl0
    luparm(21) = lenL
    luparm(22) = lenU
    luparm(23) = lenL
    luparm(24) = lenU
    luparm(25) = lrow
    luparm(27) = mersum
    luparm(28) = nUtri
    luparm(29) = nLtri

    parmlu(10) = Amax
    parmlu(11) = Lmax
    parmlu(12) = Umax
    parmlu(13) = DUmax
    parmlu(14) = DUmin
    parmlu(15) = Akmax

    Agrwth = Akmax  / (Amax + 1.0e-20_rp)
    Ugrwth = Umax   / (Amax + 1.0e-20_rp)
    if ( TPP ) then
        growth = Ugrwth
    else ! TRP or TCP or TSP
        growth = Agrwth
    end if
    parmlu(16) = growth

    !------------------------------------------------------------------
    ! Print statistics for the LU factors.
    !------------------------------------------------------------------
    ncp    = luparm(26)
    condU  = DUmax / max( DUmin, 1.0e-20_rp )
    dincr  = lenL + lenU - nelem
    dincr  = dincr * 100.0_rp / max( delem, one )
    avgmer = mersum
    avgmer = avgmer / dm
    nbump  = m - nUtri - nLtri

    if (nout > 0  .and.  lprint >= 10) then
       if ( TPP ) then
          write(nout, 1100) avgmer, lenL, lenL+lenU, ncp, dincr,       &
                            nUtri, lenU, Ltol, Umax, Ugrwth,           &
                            nLtri, ndens1, Lmax

       else
          write(nout, 1120) kPiv(lPiv), avgmer,                        &
                            lenL, lenL+lenU, ncp, dincr,               &
                            nUtri, lenU, Ltol, Umax, Ugrwth,           &
                            nLtri, ndens1, Lmax, Akmax, Agrwth
       end if

       write(nout, 1200) nbump, ndens2, DUmax, DUmin, condU
    end if

    return

1000 format(' m', i12, ' ', a, 'n', i12, '  Elems', i9,                &
            '  Amax', es10.1, '  Density', f7.2)
1100 format(' Merit', f8.1, '  lenL', i9, '  L+U', i11,                &
            '  Cmpressns', i5, '  Incres', f8.2                        &
      /     ' Utri', i9, '  lenU', i9, '  Ltol', es10.2,               &
            '  Umax', es10.1, '  Ugrwth', es8.1                        &
      /     ' Ltri', i9, '  dense1', i7, '  Lmax', es10.2)
1120 format(' Mer', a2, f8.1, '  lenL', i9, '  L+U', i11,              &
            '  Cmpressns', i5, '  Incres', f8.2                        &
      /     ' Utri', i9, '  lenU', i9, '  Ltol', es10.2,               &
            '  Umax', es10.1, '  Ugrwth', es8.1                        &
      /     ' Ltri', i9, '  dense1', i7, '  Lmax', es10.2,             &
            '  Akmax', es9.1, '  Agrwth', es8.1)
1200 format(' bump', i9, '  dense2', i7, '  DUmax', es9.1,             &
            '  DUmin', es9.1, '  condU', es9.1)
1300 format(/ ' lu1fac  error...  entry  a(', i8, ')  has an illegal', &
              ' row or column index'                                   &
            //' indc, indr =', 2i8)
1400 format(/ ' lu1fac  error...  entry  a(', i8, ')  has the same',   &
              ' indices as an earlier entry'                           &
            //' indc, indr =', 2i8)
1700 format(/ ' lu1fac  error...  insufficient storage'                &
            //' Increase  lena  from', i10, '  to at least', i10)
1800 format(/ ' lu1fac  error...  fatal bug',                          &
              '   (sorry --- this should never happen)')
1900 format(/ ' lu1fac  error...  TSP used but',                       &
              ' diagonal pivot could not be found')

  end subroutine lu1fac

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1fad( m     , n     , nelem , lena  , luparm, parmlu,   &
                     a     , indc  , indr  , p     , q     ,           &
                     lenc  , lenr  , locc  , locr  ,                   &
                     iploc , iqloc , ipinv , iqinv , w     ,           &
                     lenH  , Ha    , Hj    , Hk    , Amaxr ,           &
                     inform, lenL  , lenU  , minlen, mersum,           &
                     nUtri , nLtri , ndens1, ndens2, nrank , nslack,   &
                     Lmax  , Umax  , DUmax , DUmin , Akmax )

    integer(ip),   intent(in)    :: m, n, nelem, lena, lenH, nslack
    integer(ip),   intent(inout) :: luparm(30)
    real(rp),      intent(inout) :: parmlu(30), a(lena), Amaxr(m),     &
                                    w(n), Ha(lenH)
    integer(ip),   intent(inout) :: indc(lena), indr(lena),            &
                                    p(m)    , q(n)    ,                &
                                    lenc(n)   , lenr(m)   ,            &
                                    locc(n) , locr(m) ,                &
                                    iploc(n)  , iqloc(m)  ,            &
                                    ipinv(m), iqinv(n),                &
                                    Hj(lenH)  , Hk(lenH)
    integer(ip),   intent(out)   :: inform, lenL  , lenU  ,            &
                                    minlen, mersum,     &
                                    nUtri , nLtri , ndens1, ndens2, nrank
    real(rp),      intent(out)   :: Lmax, Umax, DUmax, DUmin, Akmax

    !------------------------------------------------------------------
    ! lu1fad  is a driver for the numerical phase of lu1fac.
    ! At each stage it computes a column of  L  and a row of  U,
    ! using a Markowitz criterion to select the pivot element,
    ! subject to a stability criterion that bounds the elements of  L.
    !
    ! 00 Jan 1986  Version documented in LUSOL paper:
    !              Gill, Murray, Saunders and Wright (1987),
    !              Maintaining LU factors of a general sparse matrix,
    !              Linear algebra and its applications 88/89, 239-270.
    !
    ! 02 Feb 1989  Following Suhl and Aittoniemi (1987), the largest
    !              element in each column is now kept at the start of
    !              the column, i.e. in position locc(j) of a and indc.
    !              This should speed up the Markowitz searches.
    !              To save time on highly triangular matrices, we wait
    !              until there are no further columns of length 1
    !              before setting and maintaining that property.
    !
    ! 12 Apr 1989  ipinv and iqinv added (inverses of p and q)
    !              to save searching p and q for rows and columns
    !              altered in each elimination step.  (Used in lu1pq2)
    !
    ! 19 Apr 1989  Code segmented to reduce its size.
    !              lu1gau does most of the Gaussian elimination work.
    !              lu1mar does just the Markowitz search.
    !              lu1mxc moves biggest elements to top of columns.
    !              lu1pen deals with pending fill-in in the row list.
    !              lu1pq2 updates the row and column permutations.
    !
    ! 26 Apr 1989  maxtie replaced by maxcol, maxrow in the Markowitz
    !              search.  maxcol, maxrow change as density increases.
    !
    ! 25 Oct 1993  keepLU implemented.
    !
    ! 07 Feb 1994  Exit main loop early to finish off with a dense LU.
    !              densLU tells lu1fad whether to do it.
    ! 21 Dec 1994  Bug fixed.  nrank was wrong after the call to lu1ful.
    ! 12 Nov 1999  A parallel version of dcopy gave trouble in lu1ful
    !              during left-shift of dense matrix D within a(*).
    !              Fixed this unexpected problem here in lu1fad
    !              by making sure the first and second D don't overlap.
    !
    ! 13 Sep 2000  TCP (Threshold Complete Pivoting) implemented.
    !              lu2max added
    !              (finds aijmax from biggest elems in each col).
    !              Utri, Ltri and Spars1 phases apply.
    !              No switch to Dense CP yet.  (Only TPP switches.)
    ! 14 Sep 2000  imax needed to remember row containing aijmax.
    ! 22 Sep 2000  For simplicity, lu1mxc always fixes
    !              all modified cols.
    !              (TPP spars2 used to fix just the first maxcol cols.)
    ! 08 Nov 2000: Speed up search for aijmax.
    !              Don't need to search all columns if the elimination
    !              didn't alter the col containing the current aijmax.
    ! 21 Nov 2000: lu1slk implemented for Utri phase with TCP
    !              to guard against deceptive triangular matrices.
    !              (Utri used to have aijtol >= 0.9999 to include
    !              slacks, but this allows other 1s to be accepted.)
    !              Utri now accepts slacks, but applies normal aijtol
    !              test to other pivots.
    ! 28 Nov 2000: TCP with empty cols must call lu1mxc and lu2max
    !              with ( lq1, n, ... ), not just ( 1, n, ... ).
    ! 23 Mar 2001: lu1fad bug with TCP.
    !              A col of length 1 might not be accepted as a pivot.
    !              Later it appears in a pivot row and temporarily
    !              has length 0 (when pivot row is removed
    !              but before the column is filled in).  If it is the
    !              last column in storage, the preceding col also thinks
    !              it is "last".  Trouble arises when the preceding col
    !              needs fill-in -- it overlaps the real "last" column.
    !              (Very rarely, same trouble might have happened if
    !              the drop tolerance caused columns to have length 0.)
    !
    !              Introduced ilast to record the last row in row file,
    !                         jlast to record the last col in col file.
    !              lu1rec returns ilast = indr(lrow + 1)
    !                          or jlast = indc(lcol + 1).
    !        ***   (Should be an output parameter, but didn't want to
    !              alter lu1rec's parameter list.)
    !              lu1rec also treats empty rows or cols safely.
    !              (Doesn't eliminate them!)
    !        ***   20 Dec 2015: Made ilast an output as it should be.
    !
    ! 26 Apr 2002: Heap routines added for TCP.
    !              lu2max no longer needed.
    !              imax, jmax used only for printing.
    ! 01 May 2002: lu1DCP implemented (dense complete pivoting).
    !              Both TPP and TCP now switch to dense LU
    !              when density exceeds dens2.
    ! 06 May 2002: In dense mode, store diag(U) in natural order.
    ! 09 May 2002: lu1mCP implemented (Markowitz TCP via heap).
    ! 11 Jun 2002: lu1mRP implemented (Markowitz TRP).
    ! 28 Jun 2002: Fixed call to lu1mxr.
    ! 14 Dec 2002: lu1mSP implemented (Markowitz TSP).
    ! 15 Dec 2002: Both TPP and TSP can grab cols of length 1
    !              during Utri.
    ! 19 Dec 2004: Hdelete(...) has new input argument Hlenin.
    ! 26 Mar 2006: lu1fad returns nrank  = min( mrank, nrank )
    !              and ignores nsing from lu1ful
    !
    ! 10 Jan 2010: First f90 version.
    ! 03 Apr 2013: lu1mxr recoded to improve efficiency of TRP.
    ! 12 Dec 2015: nslack is now an input.
    ! 20 Dec 2015: lu1rec returns ilast as output parameter.
    !------------------------------------------------------------------

    logical                :: Utri, Ltri, spars1, spars2, dense,       &
                              densLU, keepLU, TCP, TPP, TRP, TSP
    real(rp)               :: abest, aijmax, aijtol, amax, &
                              dens1, dens2, diag,          &
                              Lij, Ltol, small, Uspace

    !------------------------------------------------------------------
    ! Local variables
    !---------------
    !
    ! lcol   is the length of the column file.  It points to the last
    !        nonzero in the column list.
    ! lrow   is the analogous quantity for the row file.
    ! lfile  is the file length (lcol or lrow) after the most recent
    !        compression of the column list or row list.
    ! nrowd  and  ncold  are the number of rows and columns in the
    !        matrix defined by the pivot column and row.  They are the
    !        dimensions of the submatrix D being altered at this stage.
    ! melim  and  nelim  are the number of rows and columns in the
    !        same matrix D, excluding the pivot column and row.
    ! mleft  and  nleft  are the number of rows and columns
    !        still left to be factored.
    ! nzchng is the increase in nonzeros in the matrix that remains
    !        to be factored after the current elimination
    !        (usually negative).
    ! nzleft is the number of nonzeros still left to be factored.
    ! nspare is the space we leave at the end of the last row or
    !        column whenever a row or column is being moved to the end
    !        of its file.  nspare = 1 or 2 might help reduce the
    !        number of file compressions when storage is tight.
    !
    ! The row and column ordering permutes A into the form
    !
    !                        ------------------------
    !                         \                     |
    !                          \         U1         |
    !                           \                   |
    !                            --------------------
    !                            |\
    !                            | \
    !                            |  \
    !            P A Q   =       |   \
    !                            |    \
    !                            |     --------------
    !                            |     |            |
    !                            |     |            |
    !                            | L1  |     A2     |
    !                            |     |            |
    !                            |     |            |
    !                            --------------------
    !
    ! where the block A2 is factored as  A2 = L2 U2.
    ! The phases of the factorization are as follows.
    !
    ! Utri   is true when U1 is being determined.
    !        Any column of length 1 is accepted immediately (if TPP).
    !
    ! Ltri   is true when L1 is being determined.
    !        lu1mar exits as soon as an acceptable pivot is found
    !        in a row of length 1.
    !
    ! spars1 is true while the density of the (modified) A2 is less
    !        than the parameter dens1 = parmlu(7) = 0.3 say.
    !        lu1mar searches maxcol columns and maxrow rows,
    !        where  maxcol = luparm(3),  maxrow = maxcol - 1.
    !        lu1mxc is used to keep the biggest element at the top
    !        of all remaining columns.
    !
    ! spars2 is true while the density of the modified A2 is less
    !        than the parameter dens2 = parmlu(8) = 0.6 say.
    !        lu1mar searches maxcol columns and no rows.
    !        lu1mxc could fix up only the first maxcol cols (with TPP).
    !        22 Sep 2000: For simplicity, lu1mxc fixes all modified cols.
    !
    ! dense  is true once the density of A2 reaches dens2.
    !        lu1mar searches only 1 column (the shortest).
    !        lu1mxc could fix up only the first column (with TPP).
    !        22 Sep 2000: For simplicity, lu1mxc fixes all modified cols.
    !------------------------------------------------------------------

    integer(ip)       :: Hlen, Hlenin, hops, h,                &
                         i, ibest, ilast, imax,                &
                         j, jbest, jlast, jmax, lPiv,          &
                         k, kbest, kk, kslack,                 &
                         l, last, lc, lc1, lcol,               &
                         lD, ldiagU, lenD, leni, lenj,         &
                         lfile, lfirst, lfree, limit,          &
                         ll, ll1, lpivc, lpivc1, lpivc2,       &
                         lpivr, lpivr1, lpivr2, lprint,        &
                         lq, lq1, lq2, lr, lr1,                &
                         lrow, ls, lsave, lu, lu1,             &
                         mark, maxcol, maxmn, maxrow, mbest,   &
                         melim, minfre, minmn, mleft,          &
                         mrank, ncold, nelim, nfill,           &
                         nfree, nleft, nout, nrowd, nrowu,     &
                         nsing, nspare, nzchng, nzleft
    integer(ip)       :: markc(n), markr(m)
    real(rp)          :: v

    nout   = luparm(1)
    lprint = luparm(2)
    maxcol = luparm(3)
    lPiv   = luparm(6)
    keepLU = luparm(8) /= 0

    TPP    = lPiv == 0  ! Threshold Partial   Pivoting (normal).
    TRP    = lPiv == 1  ! Threshold Rook      Pivoting
    TCP    = lPiv == 2  ! Threshold Complete  Pivoting.
    TSP    = lPiv == 3  ! Threshold Symmetric Pivoting.

    densLU = .false.
    maxrow = maxcol - 1
    ilast  = m                 ! Assume row m is last in the row file.
    jlast  = n                 ! Assume col n is last in the col file.
    lfile  = nelem
    lrow   = nelem
    lcol   = nelem
    minmn  = min( m, n )
    maxmn  = max( m, n )
    nzleft = nelem
    nspare = 1
    ldiagU = 0                 ! Keep -Wmaybe-uninitialized happy.

    if ( keepLU ) then
       lu1    = lena   + 1
    else ! Store only the diagonals of U in the top of memory.
       ldiagU = lena   - n
       lu1    = ldiagU + 1
    end if

    Ltol   = parmlu(1)
    small  = parmlu(3)
    Uspace = parmlu(6)
    dens1  = parmlu(7)
    dens2  = parmlu(8)
    Utri   = .true.
    Ltri   = .false.
    spars1 = .false.
    spars2 = .false.
    dense  = .false.
    kslack = 0        ! 12 Dec 2015: Count slacks accepted during Utri.

    ! Check parameters.

    Ltol   = max( Ltol, 1.0001_rp )
    dens1  = min( dens1, dens2 )

    ! Initialize output parameters.
    ! lenL, lenU, minlen, mersum, nUtri, nLtri, ndens1, ndens2, nrank,
    ! nslack, are already initialized by lu1fac.

    lenL   = 0
    lenU   = 0
    minlen = 0
    mersum = 0
    nLtri  = 0
    nUtri  = 0
    ndens1 = 0
    ndens2 = 0
    nrank  = 0

    Lmax   = zero
    Umax   = zero
    DUmax  = zero
    DUmin  = 1.0e+20_rp
    if (nelem == 0) Dumin = zero
    Akmax  = zero
    hops   = 0

    ! More initialization.

    if (TPP .or. TSP) then ! Don't worry yet about lu1mxc.
       aijmax = zero
       aijtol = zero
       Hlen   = 1

    else ! TRP or TCP
       ! Move biggest element to top of each column.
       ! Set w(*) to mark slack columns (unit vectors).
       ! 12 Dec 2015: lu1fac (lu1slk) sets w(*) before lu1fad.
       ! 13 Dec 2015: lu1mxc fixed (empty cols caused trouble).

       call lu1mxc( i1, n, q, a, indc, lenc, locc )
     ! call lu1slk( m, n, lena, q, iqloc, a, locc, w )
    end if

    if (TRP) then ! Find biggest element in each row.
       mark = 0
       call lu1mxr( mark, i1, m, m, n, lena, inform,       &
                    a, indc, lenc, locc, indr, lenr, locr, &
                    p, markc, markr, Amaxr )
       if (inform > 0) go to 981
    end if

    if (TCP) then ! Set Ha(1:Hlen) = biggest element in each column,
                  ! Hj(1:Hlen) = corresponding column indices.
                  ! 17 Dec 2015: Allow for empty columns.
       Hlen  = 0
       do kk = 1, n
          Hlen     = Hlen + 1
          j        = q(kk)
          if (lenc(j) > 0) then
             lc   = locc(j)
             amax = abs( a(lc) )
          else
             amax = zero
          end if
          Ha(Hlen) = amax
          Hj(Hlen) = j
          Hk(j)    = Hlen
       end do

       ! Build the heap, creating new Ha, Hj and setting Hk(1:Hlen).

       call Hbuild( Ha, Hj, Hk, Hlen, Hlen, hops )
    end if

    !------------------------------------------------------------------
    ! Start of main loop.
    !------------------------------------------------------------------
    mleft  = m + 1
    nleft  = n + 1

    do 800 nrowu = 1, minmn

       ! mktime = (nrowu / ntime) + 4
       ! eltime = (nrowu / ntime) + 9
       mleft  = mleft - 1
       nleft  = nleft - 1

       ! Bail out if there are no nonzero rows left.

       if (iploc(1) > m) go to 900

       ! For TCP, the largest Aij is at the top of the heap.

       if ( TCP ) then
          aijmax = Ha(1)      ! Marvelously easy !
          Akmax  = max( Akmax, aijmax )
          aijtol = aijmax / Ltol
       end if

       !===============================================================
       ! Find a suitable pivot element.
       !===============================================================
       if ( Utri ) then
          !------------------------------------------------------------
          ! So far all columns have had length 1.
          ! We are still looking for the (backward) triangular part of A
          ! that forms the first rows and columns of U.
          ! 12 Dec 2015: Use nslack and kslack to choose slacks first.
          !------------------------------------------------------------
          lq1    = iqloc(1)
          lq2    = n
          if (m > 1) lq2 = iqloc(2) - 1

          if (kslack < nslack) then
             do lq = lq1, lq2
                j  = q(lq)
                if (w(j) > zero) then ! Accept a slack
                   kslack = kslack + 1
                   jbest  = j
                   lc     = locc(jbest)
                   ibest  = indc(lc)
                   abest  = a(lc)
                   mbest  = 0
                   go to 300
                end if
             end do

             ! DEBUG ERROR
             ! write(*,*) 'slack not found'
             ! write(*,*) 'kslack, nslack =', kslack, nslack
             ! stop

          else if (kslack == nslack) then  ! Maybe print msg
             if (lprint >= 50) then
                write(nout,*) 'Slacks ended.  nslack =', nslack
             end if
             kslack = nslack + 1          ! So print happens once
          end if

          ! All slacks will be grabbed before we get here.

          if (lq1 <= lq2) then  ! There are more cols of length 1.
             if (TPP .or. TSP) then
                jbest  = q(lq1)   ! Grab the first one.

             else ! TRP or TCP    ! Scan all columns of length 1.
                jbest  = 0

                do lq = lq1, lq2
                   j      = q(lq)
                   ! 12 Dec 2015: Slacks grabbed earlier.
                   ! if (w(j) > zero) then ! Accept a slack
                   !   jbest  = j
                   !   go to 250
                   ! end if

                   lc     = locc(j)
                   amax   = abs( a(lc) )
                   if (TRP) then
                      i      = indc(lc)
                      aijtol = Amaxr(i) / Ltol
                   end if

                   if (amax >= aijtol) then
                      jbest  = j
                      go to 250
                   end if
                end do
             end if

250          if (jbest > 0) then
                lc     = locc(jbest)
                ibest  = indc(lc)
                mbest  = 0
                go to 300
             end if
          end if

          ! This is the end of the U triangle.
          ! We will not return to this part of the code.
          ! TPP and TSP call lu1mxc for the first time
          ! (to move biggest element to top of each column).

          if (lprint >= 50) then
             write(nout, 1100) 'Utri ended.  spars1 = true'
          end if
          Utri   = .false.
          Ltri   = .true.
          spars1 = .true.
          nUtri  =  nrowu - 1
          if (TPP .or. TSP) then
             call lu1mxc( lq1, n, q, a, indc, lenc, locc )
          end if
       end if

       if ( spars1 ) then
          !------------------------------------------------------------
          ! Perform a Markowitz search.
          ! Search cols of length 1, then rows of length 1,
          ! then   cols of length 2, then rows of length 2, etc.
          !------------------------------------------------------------
          ! if (TPP) then ! 12 Jun 2002: Next line disables lu1mCP below
          if (TPP .or. TCP) then
             call lu1mar( m    , n     , lena  , maxmn,          &
                          TCP  , aijtol, Ltol  , maxcol, maxrow, &
                          ibest, jbest , mbest ,                 &
                          a    , indc  , indr  , p     , q,      &
                          lenc , lenr  , locc  , locr  ,         &
                          iploc, iqloc )

          else if (TRP) then
             call lu1mRP( m    , n     , lena  , maxmn,     &
                          Ltol , maxcol, maxrow,            &
                          ibest, jbest , mbest ,            &
                          a    , indc  , indr  , p    , q,  &
                          lenc , lenr  , locc  , locr ,     &
                          iploc, iqloc , Amaxr )

             ! else if (TCP) then ! Disabled by test above
             ! call lu1mCP( m    , n     , lena  , aijtol, &
             !              ibest, jbest , mbest ,         &
             !              a    , indc  , indr  ,         &
             !              lenc , lenr  , locc  ,         &
             !              Hlen , Ha    , Hj    )

          else if (TSP) then
             call lu1mSP( m    , n     , lena  , maxmn, &
                          Ltol , maxcol, &
                          ibest, jbest , mbest , &
                          a    , indc  , q    , locc , iqloc )
             if (ibest == 0) go to 990
          end if

          if ( Ltri ) then

             ! So far all rows have had length 1.
             ! We are still looking for the (forward) triangle of A
             ! that forms the first rows and columns of L.

             if (mbest > 0) then
                Ltri   = .false.
                nLtri  =  nrowu - 1 - nUtri
                if (lprint >= 50) then
                   write(nout, 1100) 'Ltri ended.'
                end if
             end if

          else    ! See if what's left is as dense as dens1.

             if (nzleft  >=  (dens1 * mleft) * nleft) then
                spars1 = .false.
                spars2 = .true.
                ndens1 =  nleft
                maxrow =  0
                if (lprint >= 50) then
                   write(nout, 1100) 'spars1 ended.  spars2 = true'
                end if
             end if
          end if

       else if ( spars2 .or. dense ) then
          !------------------------------------------------------------
          ! Perform a restricted Markowitz search,
          ! looking at only the first maxcol columns.  (maxrow = 0.)
          !------------------------------------------------------------
        ! if (TPP) then ! 12 Jun 2002: Next line disables lu1mCP below
          if (TPP .or. TCP) then
             call lu1mar( m    , n     , lena  , maxmn,          &
                          TCP  , aijtol, Ltol  , maxcol, maxrow, &
                          ibest, jbest , mbest ,                 &
                          a    , indc  , indr  , p     , q,      &
                          lenc , lenr  , locc  , locr  ,         &
                          iploc, iqloc )

          else if (TRP) then
             call lu1mRP( m    , n     , lena  , maxmn,     &
                          Ltol , maxcol, maxrow,            &
                          ibest, jbest , mbest ,            &
                          a    , indc  , indr  , p    , q,  &
                          lenc , lenr  , locc  , locr ,     &
                          iploc, iqloc , Amaxr )

             ! else if (TCP) then ! Disabled by test above
             ! call lu1mCP( m    , n     , lena  , aijtol, &
             !              ibest, jbest , mbest ,         &
             !              a    , indc  , indr  ,         &
             !              lenc , lenr  , locc  ,         &
             !              Hlen , Ha    , Hj    )

          else if (TSP) then
             call lu1mSP( m    , n     , lena  , maxmn, &
                          Ltol , maxcol,                &
                          ibest, jbest , mbest ,        &
                          a    , indc  , q    , locc , iqloc )
             if (ibest == 0) go to 985
          end if

          ! See if what's left is as dense as dens2.

          if ( spars2 ) then
             if (nzleft  >=  (dens2 * mleft) * nleft) then
                spars2 = .false.
                dense  = .true.
                ndens2 =  nleft
                maxcol =  1
                if (lprint >= 50) then
                   write(nout, 1100) 'spars2 ended.  dense = true'
                end if
             end if
          end if
       end if

       !---------------------------------------------------------------
       ! See if we can finish quickly.
       !---------------------------------------------------------------
       if ( dense  ) then
          lenD   = mleft * nleft
          nfree  = lu1 - 1

          ! 28 Sep 2015: Change 2 to 3 for safety.
          if (nfree >= 3 * lenD) then

             ! There is room to treat the remaining matrix as
             ! a dense matrix D.
             ! We may have to compress the column file first.
             ! 12 Nov 1999: D used to be put at the
             !              beginning of free storage (lD = lcol + 1).
             !              Now put it at the end     (lD = lu1 - lenD)
             !              so the left-shift in lu1ful will not
             !              involve overlapping storage
             !              (fatal with parallel dcopy).

             densLU = .true.
             ndens2 = nleft
             lD     = lu1 - lenD
             if (lcol >= lD) then
                call lu1rec( n, .true., luparm, lcol, jlast, &
                             lena, a, indc, lenc, locc )
                lfile  = lcol
             end if

             go to 900
          end if
       end if

       !===============================================================
       ! The best aij has been found.
       ! The pivot row ibest and the pivot column jbest
       ! define a dense matrix D of size nrowd x ncold.
       !===============================================================
300    ncold  = lenr(ibest)
       nrowd  = lenc(jbest)
       melim  = nrowd  - 1
       nelim  = ncold  - 1
       mersum = mersum + mbest
       lenL   = lenL   + melim
       lenU   = lenU   + ncold
       if (lprint >= 50) then
          if (nrowu == 1) then
             write(nout, 1100) 'lu1fad debug:'
          end if
          if ( TPP .or. TRP .or. TSP ) then
             write(nout, 1200) nrowu, ibest, jbest, nrowd, ncold
          else ! TCP
             jmax   = Hj(1)
             imax   = indc(locc(jmax))
             write(nout, 1200) nrowu, ibest, jbest, nrowd, ncold, &
                               imax , jmax , aijmax
          end if
       end if

       !===============================================================
       ! Allocate storage for the next column of  L  and next row of  U.
       ! Initially the top of a, indc, indr are used as follows:
       !
       !            ncold       melim       ncold        melim
       !
       ! a      |...........|...........|ujbest..ujn|li1......lim|
       !
       ! indc   |...........|  lenr(i)  |  lenc(j)  |  markl(i)  |
       !
       ! indr   |...........| iqloc(i)  |  jfill(j) |  ifill(i)  |
       !
       !       ^           ^             ^           ^            ^
       !       lfree   lsave             lu1         ll1          oldlu1
       !
       ! Later the correct indices are inserted:
       !
       ! indc   |           |           |           |i1........im|
       !
       ! indr   |           |           |jbest....jn|ibest..ibest|
       !
       !===============================================================
       if ( keepLU ) then
          ! relax
       else
          ! Always point to the top spot.
          ! Only the current column of L and row of U will
          ! take up space, overwriting the previous ones.
          lu1    = ldiagU + 1
       end if
       ll1    = lu1   - melim
       lu1    = ll1   - ncold
       lsave  = lu1   - nrowd
       lfree  = lsave - ncold

       ! Make sure the column file has room.
       ! Also force a compression if its length exceeds a certain limit.

       limit  = int(Uspace*real(lfile))  +  m  +  n  +  1000
       minfre = ncold  + melim
       nfree  = lfree  - lcol
       if (nfree < minfre  .or.  lcol > limit) then
          call lu1rec( n, .true., luparm, lcol, jlast, &
                       lena, a, indc, lenc, locc )
          lfile  = lcol
          nfree  = lfree - lcol
          if (nfree < minfre) go to 970
       end if

       ! Make sure the row file has room.

       minfre = melim + ncold
       nfree  = lfree - lrow
       if (nfree < minfre  .or.  lrow > limit) then
          call lu1rec( m, .false., luparm, lrow, ilast, &
                       lena, a, indr, lenr, locr )
          lfile  = lrow
          nfree  = lfree - lrow
          if (nfree < minfre) go to 970
       end if

       !===============================================================
       ! Move the pivot element to the front of its row
       ! and to the top of its column.
       !===============================================================
       lpivr  = locr(ibest)
       lpivr1 = lpivr + 1
       lpivr2 = lpivr + nelim

       do l = lpivr, lpivr2
          if (indr(l) == jbest) exit
       end do

       indr(l)     = indr(lpivr)
       indr(lpivr) = jbest

       lpivc  = locc(jbest)
       lpivc1 = lpivc + 1
       lpivc2 = lpivc + melim

       do l = lpivc, lpivc2
          if (indc(l) == ibest) exit
       end do

       indc(l)     = indc(lpivc)
       indc(lpivc) = ibest
       abest       = a(l)
       a(l)        = a(lpivc)
       a(lpivc)    = abest

       if ( keepLU ) then
          ! relax
       else
          ! Store just the diagonal of U, in natural order.
          !!!   a(ldiagU + nrowu) = abest ! This was in pivot order.
          a(ldiagU + jbest) = abest
       end if

       !==============================================================
       ! Delete pivot col from heap.
       ! Hk tells us where it is in the heap.
       !==============================================================
       if ( TCP ) then
          kbest  = Hk(jbest)
          Hlenin = Hlen
          call Hdelete( Ha, Hj, Hk, Hlenin, Hlen, n, kbest, h )
          hops   = hops + h
       end if

       !===============================================================
       ! Delete the pivot row from the column file
       ! and store it as the next row of  U.
       ! Set indr(lu) = 0    to initialize jfill ptrs on columns of D,
       !     indc(lu) = lenj to save the original column lengths.
       !===============================================================
       a(lu1)    = abest
       indr(lu1) = jbest
       indc(lu1) = nrowd
       lu        = lu1

       diag      = abs( abest )
       Umax      = max(  Umax, diag )
       DUmax     = max( DUmax, diag )
       DUmin     = min( DUmin, diag )

       do lr = lpivr1, lpivr2
          lu      = lu + 1
          j       = indr(lr)
          lenj    = lenc(j)
          lenc(j) = lenj - 1
          lc1     = locc(j)
          last    = lc1 + lenc(j)

          do l = lc1, last
             if (indc(l) == ibest) exit
          end do

          a(lu)      = a(l)
          indr(lu)   = 0
          indc(lu)   = lenj
          Umax       = max( Umax, abs( a(lu) ) )
          a(l)       = a(last)
          indc(l)    = indc(last)
          indc(last) = 0       ! Free entry
          if (j == jlast) lcol = lcol - 1
       end do

       !===============================================================
       ! Delete the pivot column from the row file
       ! and store the nonzeros of the next column of  L.
       ! Set  indc(ll) = 0      to initialize markl(*) markers,
       ! indr(ll) = 0           to initialize ifill(*) row fill-in cntrs,
       ! indc(ls) = leni        to save the original row lengths,
       ! indr(ls) = iqloc(i)    to save parts of  iqloc(*),
       ! iqloc(i) = lsave - ls  to point to the nonzeros of  L
       !          = -1, -2, -3, ... in mark(*).
       !===============================================================
       indc(lsave) = ncold
       if (melim == 0) go to 700

       ll     = ll1 - 1
       ls     = lsave
       abest  = one / abest

       do lc = lpivc1, lpivc2
          ll       = ll + 1
          ls       = ls + 1
          i        = indc(lc)
          leni     = lenr(i)
          lenr(i)  = leni - 1
          lr1      = locr(i)
          last     = lr1 + lenr(i)

          do l = lr1, last
             if (indr(l) == jbest) exit
          end do

          indr(l)    = indr(last)
          indr(last) = 0       ! Free entry
          if (i == ilast) lrow = lrow - 1

          a(ll)      = - a(lc) * abest
          Lij        = abs( a(ll) )
          Lmax       = max( Lmax, Lij )
          !!!!! DEBUG
          ! if (Lij > Ltol) then
          ! write( *  ,*) ' Big Lij!!!', nrowu
          ! write(nout,*) ' Big Lij!!!', nrowu
          ! end if

          indc(ll)   = 0
          indr(ll)   = 0
          indc(ls)   = leni
          indr(ls)   = iqloc(i)
          iqloc(i)   = lsave - ls
       end do

       !===============================================================
       ! Do the Gaussian elimination.
       ! This involves adding a multiple of the pivot column
       ! to all other columns in the pivot row.
       !
       ! Sometimes more than one call to lu1gau is needed to allow
       ! compression of the column file.
       ! lfirst  says which column the elimination should start with.
       ! minfre  is a bound on the storage needed for any one column.
       ! lu      points to off-diagonals of u.
       ! nfill   keeps track of pending fill-in in the row file.
       !===============================================================
       if (nelim == 0) go to 700
       lfirst = lpivr1
       minfre = mleft + nspare
       lu     = 1
       nfill  = 0

400    call lu1gau( m     , melim , ncold , nspare, small ,         &
                    lpivc1, lpivc2, lfirst, lpivr2, lfree , minfre, &
                    ilast , jlast , lrow  , lcol  , lu    , nfill , &
                    a     , indc  , indr  ,                         &
                    lenc  , lenr  , locc  , locr  ,                 &
                    iqloc , a(ll1), indc(ll1),                      &
                    a(lu1), indr(ll1), indr(lu1) )

       if (lfirst > 0) then

          ! The elimination was interrupted.
          ! Compress the column file and try again.
          ! lfirst, lu and nfill have appropriate new values.

          call lu1rec( n, .true., luparm, lcol, jlast, &
                       lena, a, indc, lenc, locc )
          lfile  = lcol
          lpivc  = locc(jbest)
          lpivc1 = lpivc + 1
          lpivc2 = lpivc + melim
          nfree  = lfree - lcol
          if (nfree < minfre) go to 970
          go to 400
       end if

       !===============================================================
       ! The column file has been fully updated.
       ! Deal with any pending fill-in in the row file.
       !===============================================================
       if (nfill > 0) then

          ! Compress the row file if necessary.
          ! lu1gau has set nfill to be the number of pending fill-ins
          ! plus the current length of any rows that need to be moved.

          minfre = nfill
          nfree  = lfree - lrow
          if (nfree < minfre) then
             call lu1rec( m, .false., luparm, lrow, ilast, &
                          lena, a, indr, lenr, locr )
             lfile  = lrow
             lpivr  = locr(ibest)
             lpivr1 = lpivr + 1
             lpivr2 = lpivr + nelim
             nfree  = lfree - lrow
             if (nfree < minfre) go to 970
          end if

          ! Move rows that have pending fill-in to end of the row file.
          ! Then insert the fill-in.

          call lu1pen( m     , melim , ncold , nspare, ilast, &
                       lpivc1, lpivc2, lpivr1, lpivr2, lrow , &
                       lenc  , lenr  , locc  , locr  ,        &
                       indc  , indr  , indr(ll1), indr(lu1) )
       end if

       !===============================================================
       ! Restore the saved values of iqloc.
       ! Insert the correct indices for the col of L and the row of U.
       !===============================================================
700    lenr(ibest) = 0
       lenc(jbest) = 0

       ll          = ll1 - 1
       ls          = lsave

       do lc  = lpivc1, lpivc2
          ll       = ll + 1
          ls       = ls + 1
          i        = indc(lc)
          iqloc(i) = indr(ls)
          indc(ll) = i
          indr(ll) = ibest
       end do

       lu          = lu1 - 1

       do lr  = lpivr, lpivr2
          lu       = lu + 1
          indr(lu) = indr(lr)
       end do

       !===============================================================
       ! Free the space occupied by the pivot row
       ! and update the column permutation.
       ! Then free the space occupied by the pivot column
       ! and update the row permutation.
       !
       ! nzchng is found in both calls to lu1pq2, but we use it only
       ! after the second.
       !===============================================================
       call lu1pq2( ncold, nzchng, &
                    indr(lpivr), indc( lu1 ), lenc, iqloc, q, iqinv )

       call lu1pq2( nrowd, nzchng, &
                    indc(lpivc), indc(lsave), lenr, iploc, p, ipinv )

       nzleft = nzleft + nzchng

       !===============================================================
       ! lu1mxr resets Amaxr(i) in each modified row i.
       ! lu1mxc moves the largest aij to the top of each modified col j.
       ! 28 Jun 2002: Note that cols of L have an implicit diag of 1.0,
       !              so lu1mxr is called with ll1, not ll1+1, whereas
       !              lu1mxc is called with             lu1+1.
       !===============================================================
       if (Utri .and. TPP) then
          ! Relax -- we're not keeping big elements at the top yet.

       else
          if (TRP  .and.  melim > 0) then
             ! Beware: The parts of p that we need are in indc(ll1:ll)
             ! 28 Sep 2015: inform is now an output.

             mark = mark + 1
             call lu1mxr( mark, ll1, ll, m, n, lena, inform,     &
                          a, indc, lenc, locc, indr, lenr, locr, &
                          indc, markc, markr, Amaxr )
                        ! ^^^^  Here are the p(k1:k2) needed by lu1mxr.
             if (inform > 0) go to 981
          end if

          if (nelim > 0) then
             call lu1mxc( lu1+1, lu, indr, a, indc, lenc, locc )

             if (TCP) then ! Update modified columns in heap
                ! 20 Dec 2015: Allow for empty columns.
                do kk = lu1+1, lu
                   j    = indr(kk)
                   k    = Hk(j)
                   if (lenc(j) > 0) then
                      v = abs( a(locc(j)) ) ! Biggest aij in column j
                   else
                      v = zero
                   end if
                   call Hchange( Ha, Hj, Hk, Hlen, n, k, v, j, h )
                   hops = hops + h
                end do
             end if
          end if
       end if

       !===============================================================
       ! Negate lengths of pivot row and column so they will be
       ! eliminated during compressions.
       !===============================================================
       lenr(ibest) = - ncold
       lenc(jbest) = - nrowd

       ! Test for fatal bug: row or column lists overwriting L and U.

       if (lrow > lsave) go to 980
       if (lcol > lsave) go to 980

       ! Reset the file lengths if pivot row or col was at the end.

       if (ibest == ilast) then
          lrow = locr(ibest)
       end if

       if (jbest == jlast) then
          lcol = locc(jbest)
       end if
800 end do

    !------------------------------------------------------------------
    ! End of main loop.
    !------------------------------------------------------------------

    !------------------------------------------------------------------
    ! Normal exit.
    ! Move empty rows and cols to the end of p, q.
    ! Then finish with a dense LU if necessary.
    !------------------------------------------------------------------
900 inform = 0
    call lu1pq3( m, lenr, p, ipinv, mrank )
    call lu1pq3( n, lenc, q, iqinv, nrank )
    nrank  = min( mrank, nrank )

    if ( densLU ) then
       call lu1ful( m     , n    , lena , lenD , lu1 , TPP, &
                    mleft , nleft, nrank, nrowu,            &
                    lenL  , lenU , nsing,                   &
                    keepLU, small,                          &
                    a     , a(lD), indc , indr , p   , q,   &
                    lenc  , lenr , locc , ipinv, locr )
       !***     21 Dec 1994: Bug in next line.
       !***     nrank  = nrank - nsing.  Changed to next line:
       !***     nrank  = minmn - nsing

       !***     26 Mar 2006: Previous line caused bug with m<n and nsing>0.
       ! Don't mess with nrank any more.  Let end of lu1fac handle it.
    end if

    minlen = lenL  +  lenU  +  2*(m + n)
    go to 990

    ! Not enough space free after a compress.
    ! Set  minlen  to an estimate of the necessary value of  lena.

970 inform = 7
    minlen = lena  +  lfile  +  2*(m + n)
    go to 990

    ! Fatal error.  This will never happen!
    ! (Famous last words.)

980 inform = 8
    go to 990

    ! Fatal error in lu1mxr.  This will never happen!

981 inform = 10
    go to 990

    ! Fatal error with TSP.  Diagonal pivot not found.

985 inform = 9

    ! Exit.

990 return

1100 format(/ 1x, a)
1200 format(' nrowu', i7,     '   i,jbest', 2i7, '   nrowd,ncold', 2i6, &
            '   i,jmax', 2i7, '   aijmax', es10.2)

  end subroutine lu1fad

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1gau( m     , melim , ncold , nspare, small ,         &
                     lpivc1, lpivc2, lfirst, lpivr2, lfree , minfre, &
                     ilast , jlast , lrow  , lcol  , lu    , nfill , &
                     a     , indc  , indr  ,                         &
                     lenc  , lenr  , locc  , locr  ,                 &
                     mark  , al    , markl ,                         &
                     au    , ifill , jfill )

    integer(ip),   intent(in)    :: m, melim, ncold, nspare,          &
                                    lpivc1, lpivc2, lpivr2, lfree, minfre
    integer(ip),   intent(in)    :: locr(*), mark(*)
    real(rp),      intent(in)    :: small
    real(rp),      intent(in)    :: al(melim), au(ncold)

    integer(ip),   intent(inout) :: ilast, jlast, lfirst, lrow, lcol, lu, nfill
    real(rp),      intent(inout) :: a(*)
    integer(ip),   intent(inout) :: locc(*), indc(*), indr(*), lenc(*), lenr(*), &
                                    markl(melim), ifill(melim), jfill(ncold)

    !------------------------------------------------------------------
    ! lu1gau does most of the work for each step of
    ! Gaussian elimination.
    ! A multiple of the pivot column is added to each other column j
    ! in the pivot row.  The column list is fully updated.
    ! The row list is updated if there is room, but some fill-ins may
    ! remain, as indicated by ifill and jfill.
    !
    ! Input:
    ! ilast    is the row    at the end of the row    list.
    ! jlast    is the column at the end of the column list.
    ! lfirst   is the first column to be processed.
    ! lu + 1   is the corresponding element of U in au(*).
    ! nfill    keeps track of pending fill-in.
    ! a(*)     contains the nonzeros for each column j.
    ! indc(*)  contains the row indices for each column j.
    ! al(*)    contains the new column of L.  A multiple of it is
    !          used to modify each column.
    ! mark(*)  has been set to -1, -2, -3, ... in the rows
    !          corresponding to nonzero 1, 2, 3, ... of the col of L.
    ! au(*)    contains the new row of U.  Each nonzero gives the
    !          required multiple of the column of L.
    !
    ! Workspace:
    ! markl(*) marks the nonzeros of L actually used.
    !          (A different mark, namely j, is used for each column.)
    !
    ! Output:
    ! ilast     New last row    in the row    list.
    ! jlast     New last column in the column list.
    ! lfirst    = 0 if all columns were completed,
    !           > 0 otherwise.
    ! lu        returns the position of the last nonzero of U
    !           actually used, in case we come back in again.
    ! nfill     keeps track of the total extra space needed in the
    !           row file.
    ! ifill(ll) counts pending fill-in for rows involved in the new
    !           column of L.
    ! jfill(lu) marks the first pending fill-in stored in columns
    !           involved in the new row of U.
    !
    ! 16 Apr 1989: First version of lu1gau.
    ! 23 Apr 1989: lfirst, lu, nfill are now input and output
    !              to allow re-entry if elimination is interrupted.
    ! 23 Mar 2001: Introduced ilast, jlast.
    ! 27 Mar 2001: Allow fill-in "in situ" if there is already room
    !              up to but NOT INCLUDING the end of the
    !              row or column file.
    !              Seems safe way to avoid overwriting empty rows/cols
    !              at the end.  (May not be needed though, now that we
    !              have ilast and jlast.)
    !
    ! 10 Jan 2010: First f90 version.
    ! 28 Feb 2010: Declare intent and local variables.
    !------------------------------------------------------------------

    logical            :: atend
    integer(ip)        :: i, j, k, l, l1, l2, last, lc, lc1, lc2, &
                          leni, lenj, ll, lr, lr1, lrep,          &
                          ndone, ndrop, nfree
    real(rp)           :: aij, uj


    do 600 lr = lfirst, lpivr2
       j      = indr(lr)
       lenj   = lenc(j)
       nfree  = lfree - lcol
       if (nfree < minfre) go to 900

       !---------------------------------------------------------------
       ! Inner loop to modify existing nonzeros in column  j.
       ! The "do l = lc1, lc2" loop performs most of the arithmetic
       ! involved in the whole LU factorization.
       ! ndone  counts how many multipliers were used.
       ! ndrop  counts how many modified nonzeros are negligibly small.
       !---------------------------------------------------------------
       lu     = lu + 1
       uj     = au(lu)
       lc1    = locc(j)
       lc2    = lc1 + lenj - 1
       atend  = j == jlast
       ndone  = 0
       if (lenj == 0) go to 500

       ndrop  = 0

       do l = lc1, lc2
          i        =   indc(l)
          ll       = - mark(i)
          if (ll > 0) then
             ndone     = ndone + 1
             markl(ll) = j
             a(l)      = a(l)  +  al(ll) * uj
             if (abs( a(l) ) <= small) then
                ndrop  = ndrop + 1
             end if
          end if
       end do

       !---------------------------------------------------------------
       ! Remove any negligible modified nonzeros from both
       ! the column file and the row file.
       !---------------------------------------------------------------
       if (ndrop == 0) go to 500
       k      = lc1

       do l = lc1, lc2
          i        = indc(l)
          if (abs( a(l) ) > small) then
             a(k)     = a(l)
             indc(k)  = i
             k        = k + 1
             cycle
          end if

          ! Delete the nonzero from the row file.

          lenj     = lenj    - 1
          lenr(i)  = lenr(i) - 1
          lr1      = locr(i)
          last     = lr1 + lenr(i)

          do lrep = lr1, last
             if (indr(lrep) == j) exit
          end do

          indr(lrep) = indr(last)
          indr(last) = 0
          if (i == ilast) lrow = lrow - 1
       end do

       ! Free the deleted elements from the column file.

       do l = k, lc2
          indc(l) = 0
       end do
       if (atend) lcol = k - 1

       !---------------------------------------------------------------
       ! Deal with the fill-in in column j.
       !---------------------------------------------------------------
500    if (ndone == melim) go to 590

       ! See if column j already has room for the fill-in.

       if (atend) go to 540
       last   = lc1  + lenj - 1
       l1     = last + 1
       l2     = last + (melim - ndone)
       ! 27 Mar 2001: Be sure it's not at or past end of the col file.
       if (l2 >= lcol) go to 520

       do l = l1, l2
          if (indc(l) /= 0) go to 520
       end do
       go to 540

       ! We must move column j to the end of the column file.
       ! First, leave some spare room at the end of the
       ! current last column.
       ! 14 Jul 2015: (William Gandler) Fix deceptive loop
       !              do l = lcol + 1, lcol + nspare
       !                 lcol    = l

520    l1      = lcol + 1
       l2      = lcol + nspare
       do l = l1, l2
       !  lcol    = l
          indc(l) = 0     ! Spare space is free.
       end do
       lcol    = l2

       atend   = .true.
       jlast   = j
       l1      = lc1
       lc1     = lcol + 1
       locc(j) = lc1

       do l = l1, last
          lcol       = lcol + 1
          a(lcol)    = a(l)
          indc(lcol) = indc(l)
          indc(l)    = 0      ! Free space.
       end do

       !---------------------------------------------------------------
       ! Inner loop for the fill-in in column j.
       ! This is usually not very expensive.
       !---------------------------------------------------------------
540    last   = lc1 + lenj - 1
       ll     = 0

       do lc = lpivc1, lpivc2
          ll         = ll + 1
          if (markl(ll) ==  j  ) cycle
          aij        = al(ll)*uj
          if (abs(aij) <= small) cycle
          lenj       = lenj + 1
          last       = last + 1
          a(last)    = aij
          i          = indc(lc)
          indc(last) = i
          leni       = lenr(i)

          ! Add 1 fill-in to row i if there is already room.
          ! 27 Mar 2001: Be sure it's not at or past the end
          ! of the row file.

          l      = locr(i) + leni
          if (l < lrow  .and.  indr(l) <= 0) then
             indr(l) = j
             lenr(i) = leni + 1
          else

             ! Row i does not have room for the fill-in.
             ! Increment ifill(ll) to count how often this has
             ! happened to row i.  Also, add m to the row index
             ! indc(last) in column j to mark it as a fill-in that is
             ! still pending.

             ! If this is the first pending fill-in for row i,
             ! nfill includes the current length of row i
             ! (since the whole row has to be moved later).

             ! If this is the first pending fill-in for column j,
             ! jfill(lu) records the current length of column j
             ! (to shorten the search for pending fill-ins later).

             if (ifill(ll) == 0) nfill     = nfill + leni + nspare
             if (jfill(lu) == 0) jfill(lu) = lenj
             nfill      = nfill     + 1
             ifill(ll)  = ifill(ll) + 1
             indc(last) = m + i
          end if
       end do

       if ( atend ) lcol = last

       ! End loop for column  j.  Store its final length.

590    lenc(j) = lenj
600 end do

    ! Successful completion.

    lfirst = 0
    return

    ! Interruption.  We have to come back in after the
    ! column file is compressed.  Give lfirst a new value.
    ! lu and nfill will retain their current values.

900 lfirst = lr
    return

  end subroutine lu1gau

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1mar( m    , n     , lena  , maxmn,          &
                     TCP  , aijtol, Ltol  , maxcol, maxrow, &
                     ibest, jbest , mbest ,                 &
                     a    , indc  , indr  , p     , q    ,  &
                     lenc , lenr  , locc  , locr  , iploc, iqloc )

    logical,       intent(in)    :: TCP
    integer(ip),   intent(in)    :: m, n, lena, maxmn, maxcol, maxrow
    real(rp),      intent(in)    :: aijtol, Ltol, a(lena)
    integer(ip),   intent(in)    :: indc(lena), indr(lena), p(m)    , q(n)    , &
                                    lenc(n)   , lenr(m)   , iploc(n), iqloc(m), &
                                    locc(n)   , locr(m)
    integer(ip),   intent(out)   :: ibest, jbest, mbest

    !------------------------------------------------------------------
    ! lu1mar  uses a Markowitz criterion to select a pivot element
    ! for the next stage of a sparse LU factorization,
    ! subject to a Threshold Partial Pivoting stability criterion (TPP)
    ! that bounds the elements of L.
    !
    ! 00 Jan 1986: Version documented in LUSOL paper:
    !              Gill, Murray, Saunders and Wright (1987),
    !              "Maintaining LU factors of a general sparse matrix",
    !              Linear algebra and its applications 88/89, 239-270.
    !
    ! 02 Feb 1989: Following Suhl and Aittoniemi (1987), the largest
    !              element in each column is now kept at the start of
    !              the column, i.e. in position locc(j) of a and indc.
    !              This should speed up the Markowitz searches.
    !
    ! 26 Apr 1989: Both columns and rows searched during spars1 phase.
    !              Only columns searched during spars2 phase.
    !              maxtie replaced by maxcol and maxrow.
    ! 05 Nov 1993: Initializing  "mbest = m * n"  wasn't big enough when
    !              m = 10, n = 3, and last column had 7 nonzeros.
    ! 09 Feb 1994: Realised that "mbest = maxmn * maxmn" might overflow.
    !              Changed to    "mbest = maxmn * 1000".
    ! 27 Apr 2000: On large example from Todd Munson,
    !              that allowed  "if (mbest .le. nz1**2) go to 900"
    !              to exit before any pivot had been found.
    !              Introduced kbest = mbest / nz1.
    !              Most pivots can be rejected with no integer(ip) multiply.
    !              True merit is evaluated only if it's as good as the
    !              best so far (or better).  There should be no danger
    !              of integer(ip) overflow unless A is incredibly
    !              large and dense.
    !
    ! 10 Sep 2000  TCP, aijtol added for Threshold Complete Pivoting.
    !
    ! 10 Jan 2010: First f90 version.
    ! 12 Dec 2011: Declare intent.
    !------------------------------------------------------------------

    integer(ip)            :: i, j, kbest, lc, lc1, lc2, len1,          &
                              lp, lp1, lp2, lq, lq1, lq2, lr, lr1, lr2, &
                              merit, ncol, nrow, nz, nz1
    real(rp)               :: abest, aij, amax, cmax, lbest
    real(rp),    parameter :: gamma  = 2.0

    ! gamma  is "gamma" in the tie-breaking rule TB4 in the LUSOL paper.

    !------------------------------------------------------------------
    ! Search cols of length nz = 1, then rows of length nz = 1,
    ! then   cols of length nz = 2, then rows of length nz = 2, etc.
    !------------------------------------------------------------------
    abest  = zero
    lbest  = zero
    ibest  = 0
    kbest  = maxmn + 1
    mbest  = -1
    ncol   = 0
    nrow   = 0
    nz1    = 0

NZS:do nz = 1, maxmn
       ! nz1    = nz - 1
       ! if (mbest .le. nz1**2) go to 900
       if (kbest <= nz1) exit NZS
       if (ibest >  0  ) then
          if (ncol >= maxcol) go to 200
       end if
       if (nz    >  m  ) go to 200

       !---------------------------------------------------------------
       ! Search the set of columns of length  nz.
       !---------------------------------------------------------------
       lq1    = iqloc(nz)
       lq2    = n
       if (nz < m) lq2 = iqloc(nz + 1) - 1

Cols:  do lq = lq1, lq2
          ncol   = ncol + 1
          j      = q(lq)
          lc1    = locc(j)
          lc2    = lc1 + nz1
          amax   = abs( a(lc1) )

          ! Test all aijs in this column.
          ! amax is the largest element (the first in the column).
          ! cmax is the largest multiplier if aij becomes pivot.

          if ( TCP ) then
             if (amax < aijtol) cycle Cols ! Nothing in whole column
          end if

Colj:     do lc = lc1, lc2
             i      = indc(lc)
             len1   = lenr(i) - 1
             ! merit  = nz1 * len1
           ! if (merit > mbest) cycle
             if (len1  > kbest) cycle Colj

             ! aij  has a promising merit.
             ! Apply the stability test.
             ! We require  aij  to be sufficiently large compared to
             ! all other nonzeros in column  j.  This is equivalent
             ! to requiring cmax to be bounded by Ltol.

             if (lc == lc1) then

                ! This is the maximum element, amax.
                ! Find the biggest element in the rest of the column
                ! and hence get cmax.  We know cmax .le. 1, but
                ! we still want it exactly in order to break ties.
                ! 27 Apr 2002: Settle for cmax = 1.

                aij    = amax
                cmax   = one

                ! cmax   = zero
                ! do 140 l = lc1 + 1, lc2
                ! cmax  = max( cmax, abs( a(l) ) )
                ! 140            continue
                ! cmax   = cmax / amax
             else

                ! aij is not the biggest element, so cmax >= 1.
                ! Bail out if cmax will be too big.

                aij    = abs( a(lc) )
                if ( TCP ) then ! Absolute test for Complete Pivoting
                   if (aij      < aijtol) cycle Colj
                else !!! TPP
                   if (aij*Ltol < amax  ) cycle Colj
                end if
                cmax   = amax / aij
             end if

             ! aij  is big enough.  Its maximum multiplier is cmax.

             merit  = nz1 * len1
             if (merit == mbest) then

                ! Break ties.
                ! (Initializing mbest < 0 prevents getting here if
                ! nothing has been found yet.)
                ! In this version we minimize cmax
                ! but if it is already small we maximize the pivot.

                if (lbest <= gamma  .and.  cmax <= gamma) then
                   if (abest >= aij ) cycle Colj
                else
                   if (lbest <= cmax) cycle Colj
                end if
             end if

             ! aij  is the best pivot so far.

             ibest  = i
             jbest  = j
             kbest  = len1
             mbest  = merit
             abest  = aij
             lbest  = cmax
             if (nz == 1) exit NZS
          end do Colj

          ! Finished with that column.

          if (ibest > 0) then
             if (ncol >= maxcol) exit Cols
          end if
       end do Cols

       !---------------------------------------------------------------
       ! Search the set of rows of length  nz.
       !---------------------------------------------------------------
! 200  if (mbest .le. nz*nz1) go to 900
200    if (kbest <= nz    ) exit NZS
       if (ibest > 0) then
          if (nrow >= maxrow) go to 290
       end if
       if (nz > n) go to 290

       lp1    = iploc(nz)
       lp2    = m
       if (nz < n) lp2 = iploc(nz + 1) - 1

Rows:  do lp = lp1, lp2
          nrow   = nrow + 1
          i      = p(lp)
          lr1    = locr(i)
          lr2    = lr1 + nz1

Rowi:     do lr = lr1, lr2
             j      = indr(lr)
             len1   = lenc(j) - 1
             ! merit  = nz1 * len1
           ! if (merit > mbest) cycle
             if (len1  > kbest) cycle Rowi

             ! aij  has a promising merit.
             ! Find where  aij  is in column  j.

             lc1    = locc(j)
             lc2    = lc1 + len1
             amax   = abs( a(lc1) )
             do lc = lc1, lc2
                if (indc(lc) == i) exit
             end do

             ! Apply the same stability test as above.

             aij    = abs( a(lc) )
             if ( TCP ) then   !!! Absolute test for Complete Pivoting
                if (aij < aijtol) cycle Rowi
             end if

             if (lc == lc1) then

                ! This is the maximum element, amax.
                ! Find the biggest element in the rest of the column
                ! and hence get cmax.  We know cmax .le. 1, but
                ! we still want it exactly in order to break ties.
                ! 27 Apr 2002: Settle for cmax = 1.

                cmax   = one

                ! cmax   = zero
                !     do 240 l = lc1 + 1, lc2
                !        cmax  = max( cmax, abs( a(l) ) )
                ! 240 continue
                ! cmax   = cmax / amax
             else

                ! aij is not the biggest element, so cmax .ge. 1.
                ! Bail out if cmax will be too big.

                if ( TCP ) then
                   ! relax
                else
                   if (aij*Ltol < amax) cycle Rowi
                end if
                cmax   = amax / aij
             end if

             ! aij  is big enough.  Its maximum multiplier is cmax.

             merit  = nz1 * len1
             if (merit == mbest) then

                ! Break ties as before.
                ! (Initializing mbest < 0 prevents getting here if
                ! nothing has been found yet.)

                if (lbest <= gamma  .and.  cmax <= gamma) then
                   if (abest >= aij ) cycle Rowi
                else
                   if (lbest <= cmax) cycle Rowi
                end if
             end if

             ! aij  is the best pivot so far.

             ibest  = i
             jbest  = j
             kbest  = len1
             mbest  = merit
             abest  = aij
             lbest  = cmax
             if (nz == 1) exit NZS
          end do Rowi

          ! Finished with that row.

          if (ibest > 0) then
             if (nrow >= maxrow) exit Rows
          end if
       end do Rows

       ! See if it's time to quit.

290    if (ibest > 0) then
          if (nrow >= maxrow  .and.  ncol >= maxcol) exit NZS
       end if

       ! Press on with next nz.

       nz1    = nz
       if (ibest > 0) kbest = mbest / nz1
    end do NZS

  end subroutine lu1mar

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1mRP( m    , n     , lena  , maxmn,     &
                     Ltol , maxcol, maxrow,            &
                     ibest, jbest , mbest ,            &
                     a    , indc  , indr  , p    ,  q, &
                     lenc , lenr  , locc  , locr ,     &
                     iploc, iqloc , Amaxr )

    integer(ip),   intent(in)    :: m, n, lena, maxmn, maxcol, maxrow
    real(rp),      intent(in)    :: Ltol
    integer(ip),   intent(in)    :: indc(lena), indr(lena), p(m)    , q(n)    , &
                                    lenc(n)   , lenr(m)   , iploc(n), iqloc(m), &
                                    locc(n)   , locr(m)
    real(rp),      intent(in)    :: a(lena)   , Amaxr(m)
    integer(ip),   intent(out)   :: ibest, jbest, mbest

    !------------------------------------------------------------------
    ! lu1mRP  uses a Markowitz criterion to select a pivot element
    ! for the next stage of a sparse LU factorization,
    ! subject to a Threshold Rook Pivoting stability criterion (TRP)
    ! that bounds the elements of L and U.
    !
    ! 11 Jun 2002: First version of lu1mRP derived from lu1mar.
    ! 11 Jun 2002: Current version of lu1mRP.
    !
    ! 10 Jan 2010: First f90 version.
    ! 12 Dec 2011: Declare intent.
    !------------------------------------------------------------------

    integer(ip)            :: i, j, kbest, lc, lc1, lc2, len1,          &
                              lp, lp1, lp2, lq, lq1, lq2, lr, lr1, lr2, &
                              merit, ncol, nrow, nz, nz1
    real(rp)               :: abest, aij, amax, atoli, atolj


    !------------------------------------------------------------------
    ! Search cols of length nz = 1, then rows of length nz = 1,
    ! then   cols of length nz = 2, then rows of length nz = 2, etc.
    !------------------------------------------------------------------
    abest  = zero
    ibest  = 0
    kbest  = maxmn + 1
    mbest  = -1
    ncol   = 0
    nrow   = 0
    nz1    = 0

NZS:do nz = 1, maxmn
       ! nz1    = nz - 1
     ! if (mbest .le. nz1**2) go to 900
       if (kbest <= nz1) exit NZS
       if (ibest >  0  ) then
          if (ncol >= maxcol) go to 200
       end if
       if (nz    >  m  ) go to 200

       !---------------------------------------------------------------
       ! Search the set of columns of length  nz.
       !---------------------------------------------------------------
       lq1    = iqloc(nz)
       lq2    = n
       if (nz < m) lq2 = iqloc(nz + 1) - 1

Cols:  do lq = lq1, lq2
          ncol   = ncol + 1
          j      = q(lq)
          lc1    = locc(j)
          lc2    = lc1 + nz1
          amax   = abs( a(lc1) )
          atolj  = amax / Ltol    ! Min size of pivots in col j

          ! Test all aijs in this column.

Colj:     do lc = lc1, lc2
             i      = indc(lc)
             len1   = lenr(i) - 1
           ! merit  = nz1 * len1
           ! if (merit > mbest) cycle Colj
             if (len1  > kbest) cycle Colj

             ! aij  has a promising merit.
             ! Apply the Threshold Rook Pivoting stability test.
             ! First we require aij to be sufficiently large
             ! compared to other nonzeros in column j.
             ! Then  we require aij to be sufficiently large
             ! compared to other nonzeros in row    i.

             aij    = abs( a(lc) )
             if (aij      < atolj   ) cycle Colj
             if (aij*Ltol < Amaxr(i)) cycle Colj

             ! aij  is big enough.

             merit  = nz1 * len1
             if (merit == mbest) then

                ! Break ties.
                ! (Initializing mbest < 0 prevents getting here if
                ! nothing has been found yet.)

                if (abest >= aij) cycle Colj
             end if

             ! aij  is the best pivot so far.

             ibest  = i
             jbest  = j
             kbest  = len1
             mbest  = merit
             abest  = aij
             if (nz == 1) exit NZS
          end do Colj

          ! Finished with that column.

          if (ibest > 0) then
             if (ncol >= maxcol) exit Cols
          end if
       end do Cols

       !---------------------------------------------------------------
       ! Search the set of rows of length  nz.
       !---------------------------------------------------------------
! 200  if (mbest .le. nz*nz1) go to 900
200    if (kbest <= nz    ) exit NZS
       if (ibest > 0) then
          if (nrow >= maxrow) go to 290
       end if
       if (nz > n) go to 290

       lp1    = iploc(nz)
       lp2    = m
       if (nz < n) lp2 = iploc(nz + 1) - 1

Rows:  do lp = lp1, lp2
          nrow   = nrow + 1
          i      = p(lp)
          lr1    = locr(i)
          lr2    = lr1 + nz1
          atoli  = Amaxr(i) / Ltol   ! Min size of pivots in row i

Rowi:     do lr = lr1, lr2
             j      = indr(lr)
             len1   = lenc(j) - 1
           ! merit  = nz1 * len1
           ! if (merit > mbest) cycle
             if (len1  > kbest) cycle Rowi

             ! aij  has a promising merit.
             ! Find where  aij  is in column j.

             lc1    = locc(j)
             lc2    = lc1 + len1
             amax   = abs( a(lc1) )
             do lc = lc1, lc2
                if (indc(lc) == i) exit
             end do

             ! Apply the Threshold Rook Pivoting stability test.
             ! First we require aij to be sufficiently large
             ! compared to other nonzeros in row    i.
             ! Then  we require aij to be sufficiently large
             ! compared to other nonzeros in column j.

             aij    = abs( a(lc) )
             if (aij      < atoli) cycle Rowi
             if (aij*Ltol < amax ) cycle Rowi

             ! aij  is big enough.

             merit  = nz1 * len1
             if (merit == mbest) then

                ! Break ties as before.
                ! (Initializing mbest < 0 prevents getting here if
                ! nothing has been found yet.)

                if (abest >= aij ) cycle Rowi
             end if

             ! aij  is the best pivot so far.

             ibest  = i
             jbest  = j
             kbest  = len1
             mbest  = merit
             abest  = aij
             if (nz == 1) exit NZS
          end do Rowi ! This was loop 260

          ! Finished with that row.

          if (ibest > 0) then
             if (nrow >= maxrow) exit Rows
          end if
       end do Rows

       ! See if it's time to quit.

290    if (ibest > 0) then
          if (nrow >= maxrow  .and.  ncol >= maxcol) exit NZS
       end if

       ! Press on with next nz.

       nz1    = nz
       if (ibest > 0) kbest  = mbest / nz1
    end do NZS

  end subroutine lu1mRP

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1mCP( m     , n     , lena  , aijtol, &
                     ibest , jbest , mbest ,         &
                     a     , indc  , indr  ,         &
                     lenc  , lenr  , locc  ,         &
                     Hlen  , Ha    , Hj    )

    integer(ip),   intent(in)    :: m, n, lena, Hlen
    integer(ip),   intent(in)    :: indc(lena), indr(lena), &
                                    lenc(n)   , lenr(m)   , locc(n), Hj(Hlen)
    real(rp),      intent(in)    :: aijtol
    real(rp),      intent(in)    :: a(lena)   , Ha(Hlen)
    integer(ip),   intent(out)   :: ibest, jbest, mbest

    !------------------------------------------------------------------
    ! lu1mCP  uses a Markowitz criterion to select a pivot element
    ! for the next stage of a sparse LU factorization,
    ! subject to a Threshold Complete Pivoting stability criterion (TCP)
    ! that bounds the elements of L and U.
    !
    ! 09 May 2002: First version of lu1mCP.
    !              It searches columns only, using the heap that
    !              holds the largest element in each column.
    ! 09 May 2002: Current version of lu1mCP.
    !
    ! 10 Jan 2010: First f90 version.
    ! 12 Dec 2011: Declare intent.
    !------------------------------------------------------------------

    integer(ip)            :: i, j, kheap, lc, lc1, lc2, len1, lenj, &
                              maxcol, merit, ncol, nz1
    real(rp)               :: abest, aij, amax, cmax, lbest
    real(rp),    parameter :: gamma = 2.0

    ! gamma  is "gamma" in the tie-breaking rule TB4 in the LUSOL paper.

    !------------------------------------------------------------------
    ! Search up to maxcol columns stored at the top of the heap.
    ! The very top column helps initialize mbest.
    !------------------------------------------------------------------
    abest  = zero
    lbest  = zero
    ibest  = 0
    jbest  = Hj(1)               ! Column at the top of the heap
    lenj   = lenc(jbest)
    mbest  = lenj * Hlen         ! Bigger than any possible merit
    maxcol = 40                  ! ??? Big question
    ncol   = 0                   ! No. of columns searched

Cols: do kheap = 1, Hlen

       amax   = Ha(kheap)
       if (amax < aijtol) cycle Cols

       ncol   = ncol + 1
       j      = Hj(kheap)
       !---------------------------------------------------------------
       ! This column has at least one entry big enough (the top one).
       ! Search the column for other possibilities.
       !---------------------------------------------------------------
       lenj   = lenc(j)
       nz1    = lenj - 1
       lc1    = locc(j)
       lc2    = lc1 + nz1
    !--amax   = abs( a(lc1) )

       ! Test all aijs in this column.
       ! amax is the largest element (the first in the column).
       ! cmax is the largest multiplier if aij becomes pivot.

Colj:  do lc = lc1, lc2
          i      = indc(lc)
          len1   = lenr(i) - 1
          merit  = nz1 * len1
          if (merit > mbest) cycle Colj

          ! aij  has a promising merit.

          if (lc == lc1) then

             ! This is the maximum element, amax.
             ! Find the biggest element in the rest of the column
             ! and hence get cmax.  We know cmax .le. 1, but
             ! we still want it exactly in order to break ties.
             ! 27 Apr 2002: Settle for cmax = 1.

             aij    = amax
             cmax   = one

             ! cmax   = zero
             !     do 140 l = lc1 + 1, lc2
             !        cmax  = max( cmax, abs( a(l) ) )
             ! 140 continue
             ! cmax   = cmax / amax
          else

             ! aij is not the biggest element, so cmax .ge. 1.
             ! Bail out if cmax will be too big.

             aij    = abs( a(lc) )
             if (aij < aijtol) cycle Colj
             cmax   = amax / aij
          end if

          ! aij  is big enough.  Its maximum multiplier is cmax.

          if (merit == mbest) then

             ! Break ties.
             ! (Initializing mbest "too big" prevents getting here if
             ! nothing has been found yet.)
             ! In this version we minimize cmax
             ! but if it is already small we maximize the pivot.

             if (lbest <= gamma  .and.  cmax <= gamma) then
                if (abest >= aij ) cycle Colj
             else
                if (lbest <= cmax) cycle Colj
             end if
          end if

          ! aij  is the best pivot so far.

          ibest  = i
          jbest  = j
          mbest  = merit
          abest  = aij
          lbest  = cmax
          if (merit == 0) exit Cols ! Col or row of length 1
       end do Colj

       if (ncol >= maxcol) exit Cols
    end do Cols

  end subroutine lu1mCP

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1mSP( m    , n     , lena  , maxmn, &
                     Ltol , maxcol,                &
                     ibest, jbest , mbest ,        &
                     a    , indc  , q    , locc , iqloc )

    integer(ip),   intent(in)    :: m, n, lena, maxmn, maxcol
    real(rp),      intent(in)    :: Ltol, a(lena)
    integer(ip),   intent(in)    :: indc(lena), q(n), iqloc(m), locc(n)

    integer(ip),   intent(out)   :: ibest, jbest, mbest

    !------------------------------------------------------------------
    ! lu1mSP  is intended for symmetric matrices that are either
    ! definite or quasi-definite.
    ! lu1mSP  uses a Markowitz criterion to select a pivot element for
    ! the next stage of a sparse LU factorization of a symmetric matrix,
    ! subject to a Threshold Symmetric Pivoting stability criterion
    ! (TSP) restricted to diagonal elements to preserve symmetry.
    ! This bounds the elements of L and U and should have rank-revealing
    ! properties analogous to Threshold Rook Pivoting for unsymmetric
    ! matrices.
    !
    ! 14 Dec 2002: First version of lu1mSP derived from lu1mRP.
    !              There is no safeguard to ensure that A is symmetric.
    ! 14 Dec 2002: Current version of lu1mSP.
    !
    ! 10 Jan 2010: First f90 version.
    ! 12 Dec 2011: Declare intent.
    !------------------------------------------------------------------

    integer(ip)            :: i, j, kbest, lc, lc1, lc2, &
                              lq, lq1, lq2, merit, ncol, nz, nz1
    real(rp)               :: abest, aij, amax, atolj


    !------------------------------------------------------------------
    ! Search cols of length nz = 1, then cols of length nz = 2, etc.
    !------------------------------------------------------------------
    abest  = zero
    ibest  = 0
    kbest  = maxmn + 1
    mbest  = -1
    ncol   = 0
    nz1    = 0

NZS:do nz = 1, maxmn
     ! nz1    = nz - 1
     ! if (mbest <= nz1**2) exit
       if (kbest <= nz1   ) exit NZS
       if (ibest > 0) then
          if (ncol >= maxcol) go to 200
       end if
       if (nz > m) go to 200

       !---------------------------------------------------------------
       ! Search the set of columns of length  nz.
       !---------------------------------------------------------------
       lq1    = iqloc(nz)
       lq2    = n
       if (nz < m) lq2 = iqloc(nz + 1) - 1

Cols:  do lq = lq1, lq2
          ncol   = ncol + 1
          j      = q(lq)
          lc1    = locc(j)
          lc2    = lc1 + nz1
          amax   = abs( a(lc1) )
          atolj  = amax / Ltol    ! Min size of pivots in col j

          ! Test all aijs in this column.
          ! Ignore everything except the diagonal.

Colj:     do lc = lc1, lc2
             i      = indc(lc)
             if (i /= j) cycle Colj     ! Skip off-diagonals.
             ! merit  = nz1 * nz1
             ! if (merit > mbest) cycle
             if (nz1   > kbest) cycle Colj

             ! aij  has a promising merit.
             ! Apply the Threshold Partial Pivoting stability test
             ! (which is equivalent to Threshold Rook Pivoting for
             ! symmetric matrices).
             ! We require aij to be sufficiently large
             ! compared to other nonzeros in column j.

             aij    = abs( a(lc) )
             if (aij < atolj  ) cycle Colj

             ! aij  is big enough.

             merit  = nz1 * nz1
             if (merit == mbest) then

                ! Break ties.
                ! (Initializing mbest < 0 prevents getting here if
                ! nothing has been found yet.)

                if (abest >= aij) cycle Colj
             end if

             ! aij  is the best pivot so far.

             ibest  = i
             jbest  = j
             kbest  = nz1
             mbest  = merit
             abest  = aij
             if (nz == 1) exit NZS
          end do Colj

          ! Finished with that column.

          if (ibest > 0) then
             if (ncol >= maxcol) exit Cols
          end if
       end do Cols

       ! See if it's time to quit.

200    if (ibest > 0) then
          if (ncol >= maxcol) exit NZS
       end if

       ! Press on with next nz.

       nz1    = nz
       if (ibest > 0) kbest  = mbest / nz1
    end do NZS

  end subroutine lu1mSP

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1pen( m     , melim , ncold , nspare, ilast, &
                     lpivc1, lpivc2, lpivr1, lpivr2, lrow , &
                     lenc  , lenr  , locc  , locr  ,        &
                     indc  , indr  , ifill , jfill )

    integer(ip),   intent(in)    :: m, melim, ncold, nspare, &
                                    lpivc1, lpivc2, lpivr1, lpivr2
    integer(ip),   intent(in)    :: locc(*), ifill(melim), jfill(ncold)
    integer(ip),   intent(inout) :: lrow
    integer(ip),   intent(inout) :: indc(*), indr(*), lenc(*), lenr(*)
    integer(ip),   intent(inout) :: locr(*)
    integer(ip),   intent(out)   :: ilast

    !------------------------------------------------------------------
    ! lu1pen deals with pending fill-in in the row file.
    ! ifill(ll) says if a row involved in the new column of L
    ! has to be updated.  If positive, it is the total
    ! length of the final updated row.
    ! jfill(lu) says if a column involved in the new row of U
    ! contains any pending fill-ins.  If positive, it points
    ! to the first fill-in in the column that has yet to be
    ! added to the row file.
    !
    ! 16 Apr 1989: First version of lu1pen.
    ! 23 Mar 2001: ilast used and updated.
    !
    ! 10 Jan 2010: First f90 version.
    ! 12 Dec 2011: Declare intent.
    ! 14 Jul 2015: (William Gandler) Fix deceptive loop 
    !              do l = lrow + 1, lrow + nspare
    !                 lrow    = l
    !------------------------------------------------------------------

    integer(ip)       :: i, j, l, l1, l2, last, lc, lc1, lc2, ll, lr, lr1, lr2, lu

    ll     = 0

    do lc = lpivc1, lpivc2
       ll = ll + 1
       if (ifill(ll) == 0) cycle

       ! Another row has pending fill.
       ! First, add some spare space at the end
       ! of the current last row.
       ! 14 Jul 2015: (William Gandler) Fix deceptive loop
       !              (same as fix in previous comment)

       l1     = lrow + 1
       l2     = lrow + nspare
       do l = l1, l2
       !  lrow    = l
          indr(l) = 0
       end do
       lrow   = l2

       ! Now move row i to the end of the row file.

       i       = indc(lc)
       ilast   = i
       lr1     = locr(i)
       lr2     = lr1 + lenr(i) - 1
       locr(i) = lrow + 1

       do lr = lr1, lr2
          lrow       = lrow + 1
          indr(lrow) = indr(lr)
          indr(lr)   = 0
       end do

       lrow    = lrow + ifill(ll)
    end do

    ! Scan all columns of  D  and insert the pending fill-in
    ! into the row file.

    lu     = 1

    do lr = lpivr1, lpivr2
       lu     = lu + 1
       if (jfill(lu) == 0) cycle
       j      = indr(lr)
       lc1    = locc(j) + jfill(lu) - 1
       lc2    = locc(j) + lenc(j)   - 1

       do lc = lc1, lc2
          i      = indc(lc) - m
          if (i > 0) then
             indc(lc)   = i
             last       = locr(i) + lenr(i)
             indr(last) = j
             lenr(i)    = lenr(i) + 1
          end if
       end do
    end do

  end subroutine lu1pen

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1mxc( k1, k2, q, a, indc, lenc, locc )

    integer(ip),   intent(in)    :: k1, k2
    integer(ip),   intent(in)    :: q(k2), lenc(*), locc(*)
    integer(ip),   intent(inout) :: indc(*)
    real(rp),      intent(inout) :: a(*)

    !------------------------------------------------------------------
    ! lu1mxc  moves the largest element in each of columns q(k1:k2)
    ! to the top of its column.
    ! If k1 > k2, nothing happens.
    !
    ! 06 May 2002: (and earlier)
    !              All columns k1:k2 must have one or more elements.
    ! 07 May 2002: Allow for empty columns.  The heap routines need to
    !              find 0.0 as the "largest element".
    !
    ! 10 Jan 2010: First f90 version.
    ! 12 Dec 2011: Declare intent.
    ! 13 Dec 2015: BUG!  We can't set a(lc1) = zero for an empty col.
    !              We need to fix the heap routines another way.
    !              Here, fixed the case lenc(j) = 0.
    !------------------------------------------------------------------

    integer(ip)            :: i, j, k, l, lc, lc1, lc2
    real(rp)               :: amax

    do k = k1, k2
       j      = q(k)
       lc1    = locc(j)

       ! The next 10 lines are equivalent to
       ! l      = idamax( lenc(j), a(lc1), 1 )  +  lc1 - 1
       ! >>>>>>>>
       lc2    = lc1 + lenc(j) - 1
       amax   = zero
       l      = lc1

       do lc = lc1, lc2
          if (amax < abs( a(lc) )) then
             amax   =  abs( a(lc) )
             l      =  lc
          end if
       end do
       ! >>>>>>>>

       ! Note that empty columns do nothing (l = lc1).
       if (l > lc1) then
          amax      = a(l)
          a(l)      = a(lc1)
          a(lc1)    = amax
          i         = indc(l)
          indc(l)   = indc(lc1)
          indc(lc1) = i
       end if
    end do

  end subroutine lu1mxc

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1mxr( mark, k1, k2, m, n, lena, inform,      &
                     a, indc, lenc, locc, indr, lenr, locr, &
                     p, markc, markr, Amaxr )

    integer(ip),   intent(in)    :: mark, k1, k2, m, n, lena
    integer(ip),   intent(out)   :: inform
    integer(ip),   intent(in)    :: indc(lena), lenc(n), locc(n),       &
                                    indr(lena), lenr(m), locr(m), p(k2)
    real(rp),      intent(in)    :: a(lena)

    integer(ip),   intent(inout) :: markc(n), markr(m)
    real(rp),      intent(inout) :: Amaxr(m)

    !------------------------------------------------------------------
    ! lu1mxr  finds the largest element in each of rows i = p(k1:k2)
    ! and stores it in each Amaxr(i).
    ! The nonzeros are stored column-wise in (a,indc,lenc,locc)
    ! and their structure is     row-wise in (  indr,lenr,locr).
    !
    ! 11 Jun 2002: First version of lu1mxr.
    !              Allow for empty columns.
    ! 10 Jan 2010: First f90 version.
    ! 12 Dec 2011: Declare intent.
    ! 03 Apr 2013: Recoded to improve efficiency.  Need new arrays
    !              markc(n), markr(m) and local array cols(n).
    !
    !              First call:  mark = 0, k1 = 1, k2 = m.
    !              Initialize all of markc(n), markr(m), Amaxr(m).
    !              Columns are searched only once.
    !              cols(n) is not used.
    !
    !              Later: mark := mark + 1 (greater than for previous call).
    !              Cols involved in rows p(k1:k2) are searched only once.
    !              cols(n) is local storage.
    !              markc(:), markr(:) are marked (= mark) in some places.
    !              For next call with new mark,
    !              all of markc, markr will initially appear unmarked.
    ! 28 Sep 2015: inform is now an output to mean i is invalid.
    !------------------------------------------------------------------

    integer(ip)            :: cols(n)
    integer(ip)            :: i, j, k, lc, lc1, lc2, lr, lr1, lr2, ncol

    inform = 0

    if (mark == 0) then    ! First call: Find Amaxr(1:m) for original A.
       markr(1:m) = 0
       markc(1:n) = 0
       Amaxr(1:m) = zero
       do j = 1, n
          lc1   = locc(j)
          lc2   = lc1 + lenc(j) - 1
          do lc = lc1, lc2
             i  = indc(lc)
             Amaxr(i) = max( Amaxr(i), abs(a(lc)) )
          end do
       end do

    else                    ! Later calls: Find Amaxr(i) for rows i = p(k1:k2).

       ncol = 0
       do k = k1, k2        ! Search rows to find which cols are involved.
          i        = p(k)
          markr(i) = mark   ! Mark this row
          Amaxr(i) = zero
          lr1   = locr(i)
          lr2   = lr1 + lenr(i) - 1
          do lr = lr1, lr2     ! Mark all unmarked cols in this row.
             j  = indr(lr)     ! Build up a list of which ones they are.
             if (markc(j) /= mark) then
                 markc(j)  = mark
                 ncol      = ncol + 1
                 cols(ncol)= j
             end if
          end do
       end do

       do k = 1, ncol       ! Search involved columns.
          j     = cols(k)
          lc1   = locc(j)
          lc2   = lc1 + lenc(j) - 1
          do lc = lc1, lc2
             i  = indc(lc)
             ! 25 Sep 2015: Check for invalid i that would cause a crash.
             ! if (i > m) then
             !    write(*,*) 'lu1mxr fatal error: i =', i
             !    inform = 10
             !    return
             ! end if
             if (markr(i) == mark) then
                 Amaxr(i)  = max( Amaxr(i), abs(a(lc)) )
             end if
          end do
       end do
    end if

  end subroutine lu1mxr

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1or1( m, n, nelem, lena, small,  &
                     a, indc, indr, lenc, lenr, &
                     Amax, numnz, lerr, inform )

    integer(ip),   intent(in)    :: m, n, nelem, lena
    real(rp),      intent(in)    :: small

    real(rp),      intent(inout) :: a(lena)
    integer(ip),   intent(inout) :: indc(lena), indr(lena)

    integer(ip),   intent(out)   :: lerr, inform
    integer(ip),   intent(out)   :: lenc(n), lenr(m)

    !------------------------------------------------------------------
    ! lu1or1  organizes the elements of an  m by n  matrix  A  as
    ! follows.  On entry, the parallel arrays   a, indc, indr,
    ! contain  nelem  entries of the form     aij,    i,    j,
    ! in any order.  nelem  must be positive.
    !
    ! Entries not larger than the input parameter  small  are treated as
    ! zero and removed from   a, indc, indr.  The remaining entries are
    ! defined to be nonzero.  numnz  returns the number of such nonzeros
    ! and  Amax  returns the magnitude of the largest nonzero.
    ! The arrays  lenc, lenr  return the number of nonzeros in each
    ! column and row of  A.
    !
    ! inform = 0  on exit, except  inform = 1  if any of the indices in
    ! indc, indr  imply that the element  aij  lies outside the  m by n
    ! dimensions of  A.
    !
    ! xx Feb 1985: Original version.
    ! 17 Oct 2000: a, indc, indr now have size lena to allow nelem = 0.
    !
    ! 10 Jan 2010: First f90 version.
    ! 12 Dec 2011: Declare intent and local variables.
    !------------------------------------------------------------------

    integer(ip)           :: i, j, l, numnz
    real(rp)              :: Amax


    lenr(1:m) = 0
    lenc(1:n) = 0

    lerr   = 0
    Amax   = zero
    numnz  = nelem

    do l = nelem, 1, -1
       if (abs(a(l)) > small) then
          i      = indc(l)
          j      = indr(l)
          Amax   = max( Amax, abs(a(l)) )
          if (i < 1  .or.  i > m) go to 910
          if (j < 1  .or.  j > n) go to 910
          lenr(i) = lenr(i) + 1
          lenc(j) = lenc(j) + 1
       else

          ! Replace a negligible element by last element.  Since
          ! we are going backwards, we know the last element is ok.

          a(l)    = a(numnz)
          indc(l) = indc(numnz)
          indr(l) = indr(numnz)
          numnz   = numnz - 1
       end if
    end do

    inform = 0
    return

910 lerr   = l
    inform = 1
    return

  end subroutine lu1or1

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1or2( n, numa, lena, a, inum, jnum, lenc, locc )

    integer(ip),   intent(in)    :: n, numa, lena
    integer(ip),   intent(in)    :: lenc(n)

    integer(ip),   intent(inout) :: inum(lena), jnum(lena)
    real(rp),      intent(inout) :: a(lena)

    integer(ip),   intent(out)   :: locc(n)

    !------------------------------------------------------------------
    ! lu1or2  sorts a list of matrix elements  a(i,j)  into column
    ! order, given  numa  entries  a(i,j),  i,  j  in the parallel
    ! arrays  a, inum, jnum  respectively.  The matrix is assumed
    ! to have n columns and an arbitrary number of rows.
    !
    ! On entry, lenc(*) must contain the length of each column.
    !
    ! On exit,  a(*) and inum(*)  are sorted,  jnum(*) = 0,  and
    ! locc(j)  points to the start of column j.
    !
    ! lu1or2  is derived from mc20ad, a routine in the Harwell
    ! Subroutine Library, author J. K. Reid.

    ! xx Feb 1985: Original version.
    ! 17 Oct 2000: a, inum, jnum now have size lena to allow nelem = 0.
    !
    ! 10 Jan 2010: First f90 version.
    ! 12 Dec 2011: Declare intent and local variables.
    !------------------------------------------------------------------

    integer(ip)        :: i, ice, icep, j, ja, jb, jce, jcep, l
    real(rp)           :: ace, acep

    ! Set  loc(j)  to point to the beginning of column  j.

    l = 1
    do j  = 1, n
       locc(j) = l
       l       = l + lenc(j)
    end do

    ! Sort the elements into column order.
    ! The algorithm is an in-place sort and is of order  numa.

    do i = 1, numa
       ! Establish the current entry.
       jce     = jnum(i)
       if (jce == 0) cycle
       ace     = a(i)
       ice     = inum(i)
       jnum(i) = 0

       ! Chain from current entry.

       do j = 1, numa

          ! The current entry is not in the correct position.
          ! Determine where to store it.

          l         = locc(jce)
          locc(jce) = locc(jce) + 1

          ! Save the contents of that location.

          acep = a(l)
          icep = inum(l)
          jcep = jnum(l)

          ! Store current entry.

          a(l)    = ace
          inum(l) = ice
          jnum(l) = 0

          ! If next current entry needs to be processed,
          ! copy it into current entry.

          if (jcep == 0) exit
          ace = acep
          ice = icep
          jce = jcep
       end do
    end do

    ! Reset loc(j) to point to the start of column j.

    ja = 1
    do j  = 1, n
       jb      = locc(j)
       locc(j) = ja
       ja      = jb
    end do

  end subroutine lu1or2

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1or3( m, n, lena, indc, lenc, locc, iw, lerr, inform )

    integer(ip),   intent(in)    :: m, n, lena
    integer(ip),   intent(in)    :: indc(lena), lenc(n), locc(n)

    integer(ip),   intent(out)   :: lerr, inform
    integer(ip),   intent(out)   :: iw(m)

    !------------------------------------------------------------------
    ! lu1or3  looks for duplicate elements in an m by n matrix A
    ! defined by the column list  indc, lenc, locc.
    ! iw  is used as a work vector of length  m.
    !
    ! xx Feb 1985: Original version.
    ! 17 Oct 2000: indc, indr now have size lena to allow nelem = 0.
    !
    ! 10 Jan 2010: First f90 version.
    ! 12 Dec 2011: Declare intent and local variables.
    !------------------------------------------------------------------

    integer(ip)        :: i, j, l, l1, l2

    iw(1:m) = 0
    lerr    = 0

    do j = 1, n
       if (lenc(j) > 0) then
          l1   = locc(j)
          l2   = l1 + lenc(j) - 1

          do l = l1, l2
             i = indc(l)
             if (iw(i) == j) go to 910
             iw(i) = j
          end do
       end if
    end do

    inform = 0
    return

910 lerr   = l
    inform = 1
    return

  end subroutine lu1or3

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1or4( m, n, nelem, lena, indc, indr, lenc, lenr, locc, locr )

    integer(ip),   intent(in)    :: m, n, nelem, lena
    integer(ip),   intent(in)    :: indc(lena), lenc(n), locc(n), lenr(m)
    integer(ip),   intent(out)   :: indr(lena), locr(m)

    !------------------------------------------------------------------
    ! lu1or4     constructs a row list  indr, locr
    ! from a corresponding column list  indc, locc,
    ! given the lengths of both columns and rows in  lenc, lenr.
    !
    ! xx Feb 1985: Original version.
    ! 17 Oct 2000: indc, indr now have size lena to allow nelem = 0.
    !
    ! 10 Jan 2010: First f90 version.
    ! 12 Dec 2011: Declare intent and local variables.
    !------------------------------------------------------------------

    integer(ip)        :: i, j, jdummy, l, l1, l2, lr

    ! Initialize  locr(i)  to point just beyond where the
    ! last component of row  i  will be stored.

    l      = 1
    do i = 1, m
       l       = l + lenr(i)
       locr(i) = l
    end do

    ! By processing the columns backwards and decreasing  locr(i)
    ! each time it is accessed, it will end up pointing to the
    ! beginning of row  i  as required.

    l2     = nelem
    j      = n + 1

    do jdummy = 1, n
       j  = j - 1
       if (lenc(j) > 0) then
          l1 = locc(j)

          do l = l1, l2
             i        = indc(l)
             lr       = locr(i) - 1
             locr(i)  = lr
             indr(lr) = j
          end do

          l2     = l1 - 1
       end if
    end do

  end subroutine lu1or4

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1pq1( m, n, len, iperm, loc, inv, num )

    integer(ip),   intent(in)    :: m, n
    integer(ip),   intent(in)    :: len(m)
    integer(ip),   intent(out)   :: iperm(m), loc(n), inv(m)
    integer(ip),   intent(out)   :: num(n) ! workspace

    !------------------------------------------------------------------
    ! lu1pq1  constructs a permutation  iperm  from the array  len.
    !
    ! On entry:
    ! len(i)  holds the number of nonzeros in the i-th row (say)
    !         of an m by n matrix.
    ! num(*)  can be anything (workspace).
    !
    ! On exit:
    ! iperm   contains a list of row numbers in the order
    !         rows of length 0,  rows of length 1,..., rows of length n.
    ! loc(nz) points to the first row containing  nz  nonzeros,
    !         nz = 1, n.
    ! inv(i)  points to the position of row i within iperm(*).
    !
    ! 10 Jan 2010: First f90 version.
    ! 12 Dec 2011: Declare intent and local variables.
    !------------------------------------------------------------------

    integer(ip)        :: i, l, nz, nzero

    ! Count the number of rows of each length.

    nzero    = 0
    num(1:n) = 0
    loc(1:n) = 0

    do i = 1, m
       nz      = len(i)
       if (nz == 0) then
          nzero   = nzero   + 1
       else
          num(nz) = num(nz) + 1
       end if
    end do

    ! Set starting locations for each length.

    l      = nzero + 1
    do nz  = 1, n
       loc(nz) = l
       l       = l + num(nz)
       num(nz) = 0
    end do

    ! Form the list.

    nzero  = 0
    do i = 1, m
       nz = len(i)
       if (nz == 0) then
          nzero = nzero + 1
          iperm(nzero) = i
       else
          l        = loc(nz) + num(nz)
          iperm(l) = i
          num(nz)  = num(nz) + 1
       end if
    end do

    ! Define the inverse of iperm.

    do l = 1, m
       i      = iperm(l)
       inv(i) = l
    end do

  end subroutine lu1pq1

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1pq2( nzpiv, nzchng, indr, lenold, lennew, iqloc, q, iqinv )

    integer(ip),   intent(in)    :: nzpiv
    integer(ip),   intent(in)    :: lenold(nzpiv), lennew(*)
    integer(ip),   intent(inout) :: indr(nzpiv), iqloc(*), q(*), iqinv(*)
    integer(ip),   intent(out)   :: nzchng

    !===============================================================
    ! lu1pq2 frees the space occupied by the pivot row,
    ! and updates the column permutation q.
    !
    ! Also used to free the pivot column and update the row perm p.
    !
    ! nzpiv   (input)    is the length of the pivot row (or column).
    ! nzchng  (output)   is the net change in total nonzeros.
    !
    ! 14 Apr 1989:  First version.
    !
    ! 10 Jan 2010: First f90 version.
    ! 12 Dec 2011: Declare intent and local variables.
    !===============================================================

    integer(ip) :: j, jnew, l, lnew, lr, next, nz, nznew

    nzchng = 0

    do lr  = 1, nzpiv
       j        = indr(lr)
       indr(lr) = 0
       nz       = lenold(lr)
       nznew    = lennew(j)

       if (nz /= nznew) then
          l        = iqinv(j)
          nzchng   = nzchng + (nznew - nz)

          ! l above is the position of column j in q  (so j = q(l)).

          if (nz < nznew) then   ! Column j has to move toward the end of q.
110          next        = nz + 1
             lnew        = iqloc(next) - 1
             if (lnew /= l) then
                jnew        = q(lnew)
                q(l)        = jnew
                iqinv(jnew) = l
             end if
             l           = lnew
             iqloc(next) = lnew
             nz          = next
             if (nz < nznew) go to 110

          else   ! Column j has to move toward the front of q.
120          lnew        = iqloc(nz)
             if (lnew /= l) then
                jnew        = q(lnew)
                q(l)        = jnew
                iqinv(jnew) = l
             end if
             l           = lnew
             iqloc(nz)   = lnew + 1
             nz          = nz   - 1
             if (nz > nznew) go to 120
          end if

          q(lnew)  = j
          iqinv(j) = lnew
       end if
    end do

  end subroutine lu1pq2

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1pq3( n, len, iperm, iw, nrank )

    integer(ip),   intent(in)    :: n
    integer(ip),   intent(in)    :: len(n)
    integer(ip),   intent(inout) :: iperm(n)
    integer(ip),   intent(out)   :: iw(n)   ! workspace

    !------------------------------------------------------------------
    ! lu1pq3  looks at the permutation  iperm(*)  and moves any entries
    ! to the end whose corresponding length  len(*)  is zero.
    !
    ! 09 Feb 1994: Added work array iw(*) to improve efficiency.
    !
    ! 10 Jan 2010: First f90 version.
    ! 12 Dec 2011: Declare intent and local variables.
    !------------------------------------------------------------------

    integer(ip)        :: i, k, nrank, nzero

    nrank  = 0
    nzero  = 0

    do k = 1, n
       i = iperm(k)

       if (len(i) == 0) then
          nzero     = nzero + 1
          iw(nzero) = i
       else
          nrank        = nrank + 1
          iperm(nrank) = i
       end if
    end do

    do k = 1, nzero
       iperm(nrank + k) = iw(k)
    end do

  end subroutine lu1pq3

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1rec( n, reals, luparm, ltop, ilast, &
                     lena, a, ind, lenc, locc )

    logical,       intent(in)    :: reals
    integer(ip),   intent(in)    :: n, lena
    integer(ip),   intent(inout) :: ltop
    integer(ip),   intent(out)   :: ilast
    integer(ip),   intent(inout) :: luparm(30), ind(lena), lenc(n), locc(n)
    real(rp),      intent(inout) :: a(lena)

    !------------------------------------------------------------------
    ! lu1rec recovers space in the column or row lists.
    ! 00 Jun 1983: Original version of lu1rec followed John Reid's
    !              compression routine in LA05.  It recovered space
    !              in ind(*) and optionally a(*) by eliminating entries
    !              with ind(l) = 0.
    !              The elements of ind(*) could not be negative.
    !              If len(i) was positive, entry i contained
    !              that many elements, starting at  loc(i).
    !              Otherwise, entry i was eliminated.
    !
    ! 23 Mar 2001: Realised we could have len(i) = 0 in rare cases!
    !              (Mostly during TCP when the pivot row contains
    !              a column of length 1 that couldn't be a pivot.)
    !              Revised storage scheme to
    !                 keep        entries with       ind(l) >  0,
    !                 squeeze out entries with -n <= ind(l) <= 0,
    !              and to allow len(i) = 0.
    !              Empty items are moved to the end of the compressed
    !              ind(*) and/or a(*) arrays are given one empty space.
    !              Items with len(i) < 0 are still eliminated.
    !
    ! 27 Mar 2001: Decided to use only ind(l) > 0 and = 0 in lu1fad.
    !              Still have to keep entries with len(i) = 0.
    !
    ! On exit:
    ! ltop         is the length of useful entries in ind(*), a(*).
    ! ind(ltop+1)  is "i=ilast" such that len(i), loc(i) belong to the
    !              last item in ind(*), a(*).
    !
    ! 10 Jan 2010: First f90 version.
    ! 12 Dec 2011: Declare intent and local variables.
    ! 20 Dec 2015: ilast is output instead of ind(ltop+1).
    !------------------------------------------------------------------

    integer(ip)        :: i, k, klast, l, leni, lprint, nempty, nout

    nempty = 0

    do i = 1, n
       leni = lenc(i)
       if (leni > 0) then
          l       = locc(i) + leni - 1
          lenc(i) = ind(l)
          ind(l)  = - (n + i)
       else if (leni == 0) then
          nempty  = nempty + 1
       end if
    end do

    k      = 0
    klast  = 0    ! Previous k
    ilast  = 0    ! Last entry moved.

    do l = 1, ltop
       i = ind(l)
       if (i > 0) then
          k      = k + 1
          ind(k) = i
          if (reals) a(k) = a(l)

       else if (i < -n) then     ! This is the end of entry  i.
          i       = - (i + n)
          ilast   = i
          k       = k + 1

          ind(k)  = lenc(i)
          if (reals) a(k) = a(l)
          locc(i) = klast + 1
          lenc(i) = k     - klast
          klast   = k
       end if
    end do

    ! Move any empty items to the end, adding 1 free entry for each.

    if (nempty > 0) then
       do i = 1, n
          if (lenc(i) == 0) then
             k       = k + 1
             locc(i) = k
             ind(k)  = 0
             ilast   = i
          end if
       end do
    end if

    nout   = luparm(1)
    lprint = luparm(2)
    if (lprint >= 50) write(nout, 1000) ltop, k, reals, nempty
    luparm(26) = luparm(26) + 1  ! ncp

    ! 20 Dec 2015: Return ilast itself instead of ind(ltop + 1).

    ltop        = k
  ! ind(ltop+1) = ilast
    return

1000 format(' lu1rec.  File compressed from', i10, '   to', i10, l3, '  nempty =', i8)

  end subroutine lu1rec

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1slk( m, n, lena, q, iqloc, a, indc, locc, nslack, w )

    integer(ip),   intent(in)    :: m, n, lena
    integer(ip),   intent(in)    :: q(n), iqloc(m), indc(lena), locc(n)
    integer(ip),   intent(out)   :: nslack
    real(rp),      intent(in)    :: a(lena)
    real(rp),      intent(out)   :: w(n)

    !------------------------------------------------------------------
    ! lu1slk  sets w(j) > 0 if column j is a unit vector.
    !
    ! 21 Nov 2000: First version.  lu1fad needs it for TCP.
    !              Note that w(*) is nominally an integer(ip) array,
    !              but the only spare space is the double array w(*).
    !
    ! 10 Jan 2010: First f90 version.
    ! 12 Dec 2011: Declare intent and local variables.
    ! 12 Dec 2015: Always call lu1slk from lu1fac to obtain nslack.
    !              Need indc(*) and markr(*) to count 1 slack per row.
    !------------------------------------------------------------------

    integer(ip)   :: markr(m)
    integer(ip)   :: i, j, lc1, lq, lq1, lq2

    nslack     = 0
    markr(1:m) = 0
    w(1:n)     = zero

    ! Check all columns of length 1.

    lq1    = iqloc(1)
    lq2    = n
    if (m > 1) lq2 = iqloc(2) - 1

    do lq = lq1, lq2
       j      = q(lq)
       lc1    = locc(j)
       if (abs( a(lc1) ) == one) then
          i      = indc(lc1)
          if (markr(i) == 0) then
             nslack   = nslack + 1
             markr(i) = i
             w(j)     = one
          end if
       end if
    end do

  end subroutine lu1slk

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1ful( m     , n    , lena , lenD , lu1 , TPP, &
                     mleft , nleft, nrank, nrowu,            &
                     lenL  , lenU , nsing,                   &
                     keepLU, small,                          &
                     a     , d    , indc , indr , p   , q,   &
                     lenc  , lenr , locc , ipinv, ipvt )

    logical,       intent(in)    :: TPP, keepLU
    integer(ip),   intent(in)    :: m, n, lena, lenD, lu1,   &
                                    mleft, nleft, nrank, nrowu
    integer(ip),   intent(in)    :: locc(n)
    real(rp),      intent(in)    :: small

    integer(ip),   intent(inout) :: lenL, lenU
    integer(ip),   intent(inout) :: indc(lena), indr(lena), p(m), q(n), &
                                    lenc(n)   , lenr(m)
    real(rp),      intent(inout) :: a(lena)

    integer(ip),   intent(out)   :: ipvt(m), ipinv(m)   ! workspace
    integer(ip),   intent(out)   :: nsing  ! not used outside
    real(rp),      intent(out)   :: d(lenD)

    !------------------------------------------------------------------
    ! lu1ful computes a dense (full) LU factorization of the
    ! mleft by nleft matrix that remains to be factored at the
    ! beginning of the nrowu-th pass through the main loop of lu1fad.
    !
    ! 02 May 1989: First version.
    ! 05 Feb 1994: Column interchanges added to lu1DPP.
    ! 08 Feb 1994: ipinv reconstructed, since lu1pq3 may alter p.
    !
    ! 10 Jan 2010: First f90 version.
    ! 12 Dec 2011: Declare intent and local variables.
    !------------------------------------------------------------------

    integer(ip)         :: i, ibest, ipbase, j, jbest, k, l, l1, l2, &
                           la, lc, lc1, lc2, ld, ldbase, ldiagU,     &
                           lkk, lkn, ll, lq, lu, nrowd, ncold
    real(rp)            :: ai, aj

    !------------------------------------------------------------------
    ! If lu1pq3 moved any empty rows, reset ipinv = inverse of p.
    !------------------------------------------------------------------
    if (nrank < m) then
       do l    = 1, m
          i        = p(l)
          ipinv(i) = l
       end do
    end if

    !------------------------------------------------------------------
    ! Copy the remaining matrix into the dense matrix D.
    !------------------------------------------------------------------
    d(1:lenD) = zero

    ipbase = nrowu - 1
    ldbase = 1 - nrowu

    do lq = nrowu, n
       j      = q(lq)
       lc1    = locc(j)
       lc2    = lc1 + lenc(j) - 1

       do lc = lc1, lc2
          i      = indc(lc)
          ld     = ldbase + ipinv(i)
          d(ld)  = a(lc)
       end do

       ldbase = ldbase + mleft
    end do

    !------------------------------------------------------------------
    ! Call our favorite dense LU factorizer.
    !------------------------------------------------------------------
    if ( TPP ) then
       call lu1DPP( d, mleft, mleft, nleft, small, nsing, ipvt, q(nrowu) )
    else
       call lu1DCP( d, mleft, mleft, nleft, small, nsing, ipvt, q(nrowu) )
    end if

    !------------------------------------------------------------------
    ! Move D to the beginning of A,
    ! and pack L and U at the top of a, indc, indr.
    ! In the process, apply the row permutation to p.
    ! lkk points to the diagonal of U.
    !------------------------------------------------------------------
    a(1:lenD) = d(1:lenD)

    ldiagU = lena - n
    lkk    = 1
    lkn    = lenD - mleft + 1
    lu     = lu1

    do k  = 1, min( mleft, nleft )
       l1 = ipbase + k
       l2 = ipbase + ipvt(k)
       if (l1 /= l2) then
          i      = p(l1)
          p(l1)  = p(l2)
          p(l2)  = i
       end if
       ibest  = p(l1)
       jbest  = q(l1)

       if ( keepLU ) then
          !===========================================================
          ! Pack the next column of L.
          !===========================================================
          la     = lkk
          ll     = lu
          nrowd  = 1

          do i  = k + 1, mleft
             la = la + 1
             ai = a(la)
             if (abs( ai ) > small) then
                nrowd    = nrowd + 1
                ll       = ll    - 1
                a(ll)    = ai
                indc(ll) = p( ipbase + i )
                indr(ll) = ibest
             end if
          end do

          !===========================================================
          ! Pack the next row of U.
          ! We go backwards through the row of D
          ! so the diagonal ends up at the front of the row of  U.
          ! Beware -- the diagonal may be zero.
          !===========================================================
          la     = lkn + mleft
          lu     = ll
          ncold  = 0

          do j = nleft, k, -1
             la     = la - mleft
             aj     = a(la)
             if (abs( aj ) > small  .or.  j == k) then
                ncold    = ncold + 1
                lu       = lu    - 1
                a(lu)    = aj
                indr(lu) = q(ipbase + j)
             end if
          end do

          lenr(ibest) = - ncold
          lenc(jbest) = - nrowd
          lenL        =   lenL + nrowd - 1
          lenU        =   lenU + ncold
          lkn         =   lkn  + 1

       else
          !===========================================================
          ! Store just the diagonal of U, in natural order.
          !===========================================================
          a(ldiagU + jbest) = a(lkk)
       end if

       lkk    = lkk  + mleft + 1
    end do

  end subroutine lu1ful

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1DPP( a, lda, m, n, small, nsing, ipvt, q )

    integer(ip),   intent(in)    :: lda, m, n
    real(rp),      intent(in)    :: small

    integer(ip),   intent(inout) :: q(n)
    real(rp),      intent(inout) :: a(lda,n)

    integer(ip),   intent(out)   :: nsing   ! not used outside
    integer(ip),   intent(out)   :: ipvt(m)

    !------------------------------------------------------------------
    ! lu1DPP factors a dense m x n matrix A by Gaussian elimination,
    ! using row interchanges for stability, as in dgefa from LINPACK.
    ! This version also uses column interchanges if all elements in a
    ! pivot column are smaller than (or equal to) "small".  Such columns
    ! are changed to zero and permuted to the right-hand end.
    !
    ! As in LINPACK, ipvt(*) keeps track of pivot rows.
    ! Rows of U are interchanged, but we don't have to physically
    ! permute rows of L.  In contrast, column interchanges are applied
    ! directly to the columns of both L and U, and to the column
    ! permutation vector q(*).
    !
    ! 02 May 1989: First version derived from dgefa
    !              in LINPACK (version dated 08/14/78).
    ! 05 Feb 1994: Generalized to treat rectangular matrices
    !              and use column interchanges when necessary.
    !              ipvt is retained, but column permutations are applied
    !              directly to q(*).
    ! 21 Dec 1994: Bug found via example from Steve Dirkse.
    !              Loop 100 added to set ipvt(*) for singular rows.
    ! 26 Mar 2006: nsing redefined (see below).
    !              Changed to implicit none.
    !
    ! 10 Jan 2010: First f90 version.  Need to do more f90-ing.
    ! 12 Dec 2011: Declare intent and local variables.
    ! 03 Feb 2012: a is intent(inout), not (out).
    !              a(kp1:m,j) = t*a(kp1:m,k) + a(kp1:m,j)  needs the last :m
    !------------------------------------------------------------------
    !
    ! On entry:
    !
    ! a       Array holding the matrix A to be factored.
    ! lda     The leading dimension of the array  a.
    ! m       The number of rows    in  A.
    ! n       The number of columns in  A.
    ! small   A drop tolerance.  Must be zero or positive.
    !
    ! On exit:
    !
    ! a       An upper triangular matrix and the multipliers
    !         which were used to obtain it.
    !         The factorization can be written  A = L*U  where
    ! L       is a product of permutation and unit lower
    !         triangular matrices and  U  is upper triangular.
    ! nsing   Number of singularities detected.
    ! 26 Mar 2006: nsing redefined to be more meaningful.
    !              Users may define rankU = n - nsing and regard
    !              U as upper-trapezoidal, with the first rankU columns
    !              being triangular and the rest trapezoidal.
    !              It would be better to return rankU, but we still
    !              return nsing for compatibility (even though lu1fad
    !              no longer uses it).
    ! ipvt    Records the pivot rows.
    ! q       A vector to which column interchanges are applied.
    !------------------------------------------------------------------

    integer(ip)            :: i, j, k, kp1, l, last, lencol, rankU
    real(rp)               :: t


    rankU  = 0
    k      = 1
    last   = n

    !------------------------------------------------------------------
    ! Start of elimination loop.
    !------------------------------------------------------------------
10  kp1    = k + 1
    lencol = m - k + 1

    ! Find l, the pivot row.

    l       = jdamax( lencol, a(k:m,k), i1 ) + k - 1
    ipvt(k) = l

    if (abs( a(l,k) ) <= small) then
       !==============================================================
       ! Do column interchange, changing old pivot column to zero.
       ! Reduce "last" and try again with same k.
       !==============================================================
       j       = q(last)
       q(last) = q(k)
       q(k)    = j

       do i = 1, k - 1
          t         = a(i,last)
          a(i,last) = a(i,k)
          a(i,k)    = t
       end do

       do i = k, m
          t         = a(i,last)
          a(i,last) = zero
          a(i,k)    = t
       end do

       last     = last - 1
       if (k <= last) go to 10

    else
       rankU  = rankU + 1
       if (k < m) then
          !===========================================================
          ! Do row interchange if necessary.
          !===========================================================
          if (l /= k) then
             t      = a(l,k)
             a(l,k) = a(k,k)
             a(k,k) = t
          end if

          !===========================================================
          ! Compute multipliers.
          ! Do row elimination with column indexing.
          !===========================================================
          t = - one / a(k,k)
          ! call dscal ( m-k, t, a(kp1,k), i1 )
          a(kp1:m,k) = t*a(kp1:m,k)

          do j = kp1, last
             t    = a(l,j)
             if (l /= k) then
                a(l,j) = a(k,j)
                a(k,j) = t
             end if
             ! call daxpy ( m-k, t, a(kp1,k), i1, a(kp1,j), i1 )
             a(kp1:m,j) = t*a(kp1:m,k) + a(kp1:m,j)
          end do

          k = k + 1
          if (k <= last) go to 10
       end if
    end if

    ! Set ipvt(*) for singular rows.

    do k = last + 1, m
       ipvt(k) = k
    end do

    nsing  = n - rankU

  end subroutine lu1DPP

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu1DCP( a, lda, m, n, small, nsing, ipvt, q )

    integer(ip),   intent(in)    :: lda, m, n
    real(rp),      intent(in)    :: small

    integer(ip),   intent(inout) :: q(n)
    real(rp),      intent(inout) :: a(lda,n)

    integer(ip),   intent(out)   :: nsing   ! not used outside
    integer(ip),   intent(out)   :: ipvt(m)

    !------------------------------------------------------------------
    ! lu1DCP factors a dense m x n matrix A by Gaussian elimination,
    ! using Complete Pivoting (row and column interchanges) for
    ! stability.
    ! This version also uses column interchanges if all elements in a
    ! pivot column are smaller than (or equal to) "small".  Such columns
    ! are changed to zero and permuted to the right-hand end.
    !
    ! As in LINPACK's dgefa, ipvt(*) keeps track of pivot rows.
    ! Rows of U are interchanged, but we don't have to physically
    ! permute rows of L.  In contrast, column interchanges are applied
    ! directly to the columns of both L and U, and to the column
    ! permutation vector q(*).
    !
    ! 01 May 2002: First dense Complete Pivoting, derived from lu1DPP.
    ! 07 May 2002: Another break needed at end of first loop.
    ! 26 Mar 2006: Cosmetic mods while looking for "nsing" bug when m<n.
    !              nsing redefined (see below).
    !              Changed to implicit none.
    !
    ! 10 Jan 2010: First f90 version.
    ! 12 Dec 2011: Declare intent and local variables.
    ! 03 Feb 2012: a(kp1:m,j) = t*a(kp1:m,k) + a(kp1:m,j)  needs the last :m
    ! 21 Dec 2015: t = 0 caused divide by zero.
    !              Add test to exit if aijmax <= small.
    !------------------------------------------------------------------
    !
    ! On entry:
    ! a       Array holding the matrix A to be factored.
    ! lda     The leading dimension of the array  a.
    ! m       The number of rows    in  A.
    ! n       The number of columns in  A.
    ! small   A drop tolerance.  Must be zero or positive.
    !
    ! On exit:
    ! a       An upper triangular matrix and the multipliers
    !         that were used to obtain it.
    !         The factorization can be written A = L*U, where
    !         L is a product of permutation and unit lower
    !         triangular matrices and U is upper triangular.
    ! nsing   Number of singularities detected.
    !
    ! 26 Mar 2006: nsing redefined to be more meaningful.
    !              Users may define rankU = n - nsing and regard
    !              U as upper-trapezoidal, with the first rankU columns
    !              being triangular and the rest trapezoidal.
    !              It would be better to return rankU, but we still
    !              return nsing for compatibility (even though lu1fad
    !              no longer uses it).
    ! ipvt    Records the pivot rows.
    ! q       A vector to which column interchanges are applied.
    !------------------------------------------------------------------

    real(rp)               :: aijmax, ajmax, t
    integer(ip)            :: i, imax, j, jlast, jmax, jnew, &
                              k, kp1, l, last, lencol, rankU

    rankU  = 0
    lencol = m + 1
    last   = n

    !-----------------------------------------------------------------
    ! Start of elimination loop.
    !-----------------------------------------------------------------
    do k = 1, n
       kp1    = k + 1
       lencol = lencol - 1

       ! Find the biggest aij in row imax and column jmax.

       aijmax = zero
       imax   = k
       jmax   = k
       jlast  = last

       do j = k, jlast
10        l      = jdamax( lencol, a(k:m,j), i1 ) + k - 1
          ajmax  = abs(a(l,j))

          if (ajmax <= small) then
             !========================================================
             ! Do column interchange, changing old column to zero.
             ! Reduce "last" and try again with same j.
             !========================================================
             jnew    = q(last)
             q(last) = q(j)
             q(j)    = jnew

             do i = 1, k - 1
                t         = a(i,last)
                a(i,last) = a(i,j)
                a(i,j)    = t
             end do

             do i = k, m
                t         = a(i,last)
                a(i,last) = zero
                a(i,j)    = t
             end do

             last   = last - 1
             if (j <= last) go to 10   ! repeat
             go to 200                 ! break
          end if

          ! Check if this column has biggest aij so far.

          if (aijmax < ajmax) then
             aijmax  =   ajmax
             imax    =   l
             jmax    =   j
          end if

          if (j >= last) go to 200   ! break
       end do

200    ipvt(k) = imax

       ! 21 Dec 2015: Exit if aijmax is essentially zero.

       if (aijmax <= small) go to 500
       rankU  = rankU + 1

       if (jmax /= k) then   ! Do column interchange (k and jmax).
          jnew    = q(jmax)
          q(jmax) = q(k)
          q(k)    = jnew

          do i = 1, m
             t         = a(i,jmax)
             a(i,jmax) = a(i,k)
             a(i,k)    = t
          end do
       end if

       if (k < m) then       ! Do row interchange if necessary.
          t         = a(imax,k)
          if (imax /= k) then
             a(imax,k) = a(k,k)
             a(k,k)    = t
          end if

          !===========================================================
          ! Compute multipliers.
          ! Do row elimination with column indexing.
          !===========================================================
          t      = - one / t
          ! call dscal ( m-k, t, a(kp1,k), i1 )
          a(kp1:m,k) = t*a(kp1:m,k)

          do j = kp1, last
             t         = a(imax,j)
             if (imax /= k) then
                a(imax,j) = a(k,j)
                a(k,j)    = t
             end if
             ! call daxpy ( m-k, t, a(kp1,k), i1, a(kp1,j), i1 )
             a(kp1:m,j) = t*a(kp1:m,k) + a(kp1:m,j)
          end do

       else
          go to 500               ! break
       end if

       if (k >= last) go to 500 ! break
    end do

    ! Set ipvt(*) for singular rows.

500 do k = last + 1, m
       ipvt(k) = k
    end do

    nsing  = n - rankU

  end subroutine lu1DCP

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ! File lusol2.f90
  !
  ! Hbuild   Hchange  Hdelete  Hdown    Hinsert  Hup
  !
  ! Heap-management routines for LUSOL's lu1fac.
  ! May be useful for other applications.
  !
  ! 11 Feb 2002: MATLAB version derived from "Algorithms" by R. Sedgewick.
  ! 03 Mar 2002: F77    version derived from MATLAB version.
  ! 07 May 2002: Safeguard input parameters k, N, Nk.
  !              We don't want them to be output!
  ! 19 Dec 2004: Hdelete: Nin is new input parameter for length of Hj, Ha.
  ! 12 Dec 2011: First f90 version.
  ! 19 Dec 2015: Current version of lusol2.f90.
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !
  ! For LUSOL, the heap structure involves three arrays of length N.
  ! N        is the current number of entries in the heap.
  ! Ha(1:N)  contains the values that the heap is partially sorting.
  !          For LUSOL they are double precision values -- the largest
  !          element in each remaining column of the updated matrix.
  !          The biggest entry is in Ha(1), the top of the heap.
  ! Hj(1:N)  contains column numbers j.
  !          Ha(k) is the biggest entry in column j = Hj(k).
  ! Hk(1:N)  contains indices within the heap.  It is the
  !          inverse of Hj(1:N), so  k = Hk(j)  <=>  j = Hj(k).
  !          Column j is entry k in the heap.
  ! hops     is the number of heap operations,
  !          i.e., the number of times an entry is moved
  !          (the number of "hops" up or down the heap).
  ! Together, Hj and Hk let us find values inside the heap
  ! whenever we want to change one of the values in Ha.
  ! For other applications, Ha may need to be some other data type,
  ! like the keys that sort routines operate on.
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine Hbuild( Ha, Hj, Hk, N, Nk, hops )

    integer(ip),   intent(in)    :: N, Nk
    integer(ip),   intent(inout) :: Hj(N), Hk(Nk)
    integer(ip),   intent(out)   :: hops
    real(rp),      intent(inout) :: Ha(N)

    !==================================================================
    ! Hbuild initializes the heap by inserting each element of Ha.
    ! Input:  Ha, Hj.
    ! Output: Ha, Hj, Hk, hops.
    !
    ! 01 May 2002: Use k for new length of heap, not k-1 for old length.
    ! 05 May 2002: Use kk in call to stop loop variable k being altered.
    !              (Actually Hinsert no longer alters that parameter.)
    ! 07 May 2002: ftnchek wants us to protect Nk, Ha(k), Hj(k) too.
    ! 07 May 2002: Current version of Hbuild.
    ! 12 Dec 2011: First f90 version.
    !==================================================================

    integer(ip) :: h, jv, k, kk, Nkk
    real(rp)    :: v

    Nkk  = Nk
    hops = 0
    do k = 1, N
       kk    = k
       v     = Ha(k)
       jv    = Hj(k)
       call Hinsert( Ha, Hj, Hk, kk, Nkk, v, jv, h )
       hops  = hops + h
    end do

  end subroutine Hbuild

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine Hchange( Ha, Hj, Hk, N, Nk, k, v, jv, hops )

    integer(ip),   intent(in)    :: N, Nk, k, jv
    real(rp),      intent(in)    :: v
    integer(ip),   intent(inout) :: Hj(N), Hk(Nk)
    real(rp),      intent(inout) :: Ha(N)
    integer(ip),   intent(out)   :: hops

    !==================================================================
    ! Hchange changes Ha(k) to v in heap of length N.
    !
    ! 01 May 2002: Need Nk for length of Hk.
    ! 07 May 2002: Protect input parameters N, Nk, k.
    ! 07 May 2002: Current version of Hchange.
    ! 12 Dec 2011: First f90 version.
    !==================================================================

    integer(ip) :: kx, Nx, Nkx
    real(rp)    :: v1

    Nx     = N
    Nkx    = Nk
    kx     = k
    v1     = Ha(k)
    Ha(k)  = v
    Hj(k)  = jv
    Hk(jv) = k
    if (v1 < v) then
       call Hup   ( Ha, Hj, Hk, Nx, Nkx, kx, hops )
    else
       call Hdown ( Ha, Hj, Hk, Nx, Nkx, kx, hops )
    end if

  end subroutine Hchange

  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine Hdelete( Ha, Hj, Hk, Nin, N, Nk, k, hops )

    integer(ip),   intent(in)    :: Nin, Nk, k
    integer(ip),   intent(inout) :: N
    integer(ip),   intent(inout) :: Hj(Nin), Hk(Nk)
    real(rp),      intent(inout) :: Ha(Nin)
    integer(ip),   intent(out)   :: hops

    !==================================================================
    ! Hdelete deletes Ha(k) from heap of length N.
    !
    ! 03 Apr 2002: Current version of Hdelete.
    ! 01 May 2002: Need Nk for length of Hk.
    ! 07 May 2002: Protect input parameters N, Nk, k.
    ! 19 Dec 2004: Nin is new input parameter for length of Hj, Ha.
    ! 19 Dec 2004: Current version of Hdelete.
    ! 12 Dec 2011: First f90 version.
    !==================================================================

    integer(ip) :: jv, kx, Nkx, Nx
    real(rp)    :: v

    kx    = k
    Nkx   = Nk
    Nx    = N
    v     = Ha(N)
    jv    = Hj(N)
    N     = N - 1
    hops  = 0
    if (k <= N) then
       call Hchange( Ha, Hj, Hk, Nx, Nkx, kx, v, jv, hops )
    end if

  end subroutine Hdelete

  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine Hdown ( Ha, Hj, Hk, N, Nk, kk, hops )

    integer(ip),   intent(in)    :: N, Nk, kk
    integer(ip),   intent(inout) :: Hj(N), Hk(Nk)
    real(rp),      intent(inout) :: Ha(N)
    integer(ip),   intent(out)   :: hops

    !==================================================================
    ! Hdown  updates heap by moving down tree from node k.
    !
    ! 01 May 2002: Need Nk for length of Hk.
    ! 05 May 2002: Change input parameter k to kk to stop k being output.
    ! 05 May 2002: Current version of Hdown.
    ! 12 Dec 2011: First f90 version.
    !==================================================================

    integer(ip) :: j, jj, jv, k, N2
    real(rp)    :: v

    k     = kk
    hops  = 0
    v     = Ha(k)
    jv    = Hj(k)
    N2    = N/2

    do
       if (k > N2) exit
       hops   = hops + 1
       j      = k+k
       if (j < N) then
          if (Ha(j) < Ha(j+1)) j = j+1
       end if
       if (v >= Ha(j)) exit
       Ha(k)  = Ha(j)
       jj     = Hj(j)
       Hj(k)  = jj
       Hk(jj) =  k
       k      =  j
    end do

    Ha(k)  =  v
    Hj(k)  = jv
    Hk(jv) =  k

  end subroutine Hdown

  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine Hinsert( Ha, Hj, Hk, N, Nk, v, jv, hops )

    integer(ip),   intent(in)    :: Nk, jv
    real(rp),      intent(in)    :: v
    integer(ip),   intent(inout) :: N
    integer(ip),   intent(inout) :: Hj(N), Hk(Nk)
    real(rp),      intent(inout) :: Ha(N)
    integer(ip),   intent(out)   :: hops

    !==================================================================
    ! Hinsert inserts (v,jv) into heap of length N-1
    ! to make heap of length N.
    !
    ! 03 Apr 2002: First version of Hinsert.
    ! 01 May 2002: Require N to be final length, not old length.
    !              Need Nk for length of Hk.
    ! 07 May 2002: Protect input parameters N, Nk.
    ! 07 May 2002: Current version of Hinsert.
    ! 12 Dec 2011: First f90 version.
    !==================================================================

    integer(ip) :: kk, Nkk, Nnew

    Nnew     = N
    Nkk      = Nk
    kk       = Nnew
    Ha(Nnew) =  v
    Hj(Nnew) = jv
    Hk(jv)   = Nnew
    call Hup   ( Ha, Hj, Hk, Nnew, Nkk, kk, hops )

  end subroutine Hinsert

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine Hup   ( Ha, Hj, Hk, N, Nk, kk, hops )

    integer(ip),   intent(in)    :: N, Nk, kk
    integer(ip),   intent(inout) :: Hj(N), Hk(Nk)
    real(rp),      intent(inout) :: Ha(N)
    integer(ip),   intent(out)   :: hops

    !==================================================================
    ! Hup updates heap by moving up tree from node k.
    !
    ! 01 May 2002: Need Nk for length of Hk.
    ! 05 May 2002: Change input parameter k to kk to stop k being output.
    ! 05 May 2002: Current version of Hup.
    ! 13 Dec 2011: First f90 version.
    !==================================================================

    integer(ip) :: j, jv, k, k2
    real(rp)    :: v

    k     = kk
    hops  = 0
    v     = Ha(k)
    jv    = Hj(k)

    do
       if (k <  2) exit
       k2    = k/2
       if (v < Ha(k2)) exit
       hops  = hops + 1
       Ha(k) = Ha(k2)
       j     = Hj(k2)
       Hj(k) =  j
       Hk(j) =  k
       k     = k2
    end do

    Ha(k)  =  v
    Hj(k)  = jv
    Hk(jv) =  k

  end subroutine Hup

  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ! File lusol6a.f90
  !
  ! lu6sol   lu6L     lu6Lt     lu6U     Lu6Ut   lu6LD    lu6chk
  !
  ! 26 Apr 2002: lu6 routines put into a separate file.
  ! 15 Dec 2002: lu6sol modularized via lu6L, lu6Lt, lu6U, lu6Ut.
  !              lu6LD implemented to allow solves with LDL' or L|D|L'.
  ! 23 Apr 2004: lu6chk modified.  TRP can judge singularity better
  !              by comparing all diagonals to DUmax.
  ! 27 Jun 2004: lu6chk.  Allow write only if nout > 0 .
  ! 13 Dec 2011: First f90 version.
  ! 20 Jan 2016: Current version of lusol6a.f90.
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu6sol( mode, m, n, v, w,       &
                     lena, luparm, parmlu,   &
                     a, indc, indr, p, q,    &
                     lenc, lenr, locc, locr, &
                     inform )

    integer(ip),   intent(in)    :: mode, m, n, lena
    integer(ip),   intent(in)    :: indc(lena), indr(lena), p(m), q(n),   &
                                    lenc(n), lenr(m), locc(n), locr(m)
    real(rp),      intent(in)    :: a(lena)

    integer(ip),   intent(inout) :: luparm(30)
    real(rp),      intent(inout) :: parmlu(30), v(m), w(n)

    integer(ip),   intent(out)   :: inform

    !-----------------------------------------------------------------------
    ! lu6sol  uses the factorization  A = L U  as follows:
    !
    ! mode
    !  1    v  solves   L v = v(input).   w  is not touched.
    !  2    v  solves   L'v = v(input).   w  is not touched.
    !  3    w  solves   U w = v.          v  is not altered.
    !  4    v  solves   U'v = w.          w  is destroyed.
    !  5    w  solves   A w = v.          v  is altered as in 1.
    !  6    v  solves   A'v = w.          w  is destroyed.
    !
    ! If mode = 3,4,5,6, v and w must not be the same arrays.
    !
    ! If lu1fac has just been used to factorize a symmetric matrix A
    ! (which must be definite or quasi-definite), the factors A = L U
    ! may be regarded as A = LDL', where D = diag(U).  In such cases,
    !
    ! mode
    !  7    v  solves   A v = L D L'v = v(input).   w  is not touched.
    !  8    v  solves       L |D| L'v = v(input).   w  is not touched.
    !
    ! p(*), q(*)        hold row and column numbers in pivotal order.
    ! lenc(k)           is the length of the k-th column of initial L.
    ! lenr(i)           is the length of the i-th row of U.
    ! locc(*)           is not used.
    ! locr(i)           is the start  of the i-th row of U.
    !
    ! U is assumed to be in upper-trapezoidal form (nrank by n).
    ! The first entry for each row is the diagonal element
    ! (according to the permutations p, q).  It is stored at
    ! location locr(i) in a(*), indr(*).
    !
    ! On exit, inform = 0 except as follows.
    ! If mode = 3,4,5,6 and if U (and hence A) is singular, then
    ! inform = 1 if there is a nonzero residual in solving the system
    ! involving U.  parmlu(20) returns the norm of the residual.
    !
    ! July 1987:   Early version.
    ! 09 May 1988: f77 version.
    ! 27 Apr 2000: Abolished the dreaded "computed go to".
    !              But hard to change other "go to"s to "if then else".
    ! 15 Dec 2002: lu6L, lu6Lt, lu6U, lu6Ut added to modularize lu6sol.
    ! 13 Dec 2011: First f90 version.
    !--------------------------------------------------------------------

    if      (mode == 1) then             ! Solve  L v(new) = v.
       call lu6L  ( inform, m, n, v,    &
                    lena, luparm, parmlu, a, indc, indr, lenc )

    else if (mode == 2) then             ! Solve  L'v(new) = v.
       call lu6Lt ( inform, m, n, v,    &
                    lena, luparm, parmlu, a, indc, indr, lenc )

    else if (mode == 3) then             ! Solve  U w = v.
       call lu6U  ( inform, m, n, v, w, &
                    lena, luparm, parmlu, a, indr, p, q, lenr, locr )

    else if (mode == 4) then             ! Solve  U'v = w.
       call lu6Ut ( inform, m, n, v, w, &
                    lena, luparm, parmlu, a, indr, p, q, lenr, locr )

    else if (mode == 5) then             ! Solve  A w      = v
                                         ! via    L v(new) = v
                                         ! and    U w = v(new).
       call lu6L  ( inform, m, n, v,    &
                    lena, luparm, parmlu, a, indc, indr, lenc )
       call lu6U  ( inform, m, n, v, w, &
                    lena, luparm, parmlu, a, indr, p, q, lenr, locr )

    else if (mode == 6) then             ! Solve  A'v = w
                                         ! via    U'v = w
                                         ! and    L'v(new) = v.
       call lu6Ut ( inform, m, n, v, w, &
                    lena, luparm, parmlu, a, indr, p, q, lenr, locr )
       call lu6Lt ( inform, m, n, v,    &
                    lena, luparm, parmlu, a, indc, indr, lenc )

    else if (mode == 7) then             ! Solve  LDv(bar) = v
                                         ! and    L'v(new) = v(bar).
       call lu6LD ( inform, i1, m, n, v, &
                    lena, luparm, parmlu, a, indc, indr, lenc, locr )
       call lu6Lt ( inform, m, n, v,    &
                    lena, luparm, parmlu, a, indc, indr, lenc )

    else if (mode == 8) then             ! Solve  L|D|v(bar) = v
                                         ! and    L'v(new) = v(bar).
       call lu6LD ( inform, i2, m, n, v, &
                    lena, luparm, parmlu, a, indc, indr, lenc, locr )
       call lu6Lt ( inform, m, n, v,    &
                    lena, luparm, parmlu, a, indc, indr, lenc )
    end if

  end subroutine lu6sol

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu6L  ( inform, m, n, v, &
                     lena, luparm, parmlu, a, indc, indr, lenc )

    integer(ip),   intent(in)    :: m, n, lena
    integer(ip),   intent(in)    :: indc(lena), indr(lena), lenc(n)
    real(rp),      intent(in)    :: a(lena)

    integer(ip),   intent(inout) :: luparm(30)
    real(rp),      intent(inout) :: parmlu(30), v(m)

    integer(ip),   intent(out)   :: inform

    !------------------------------------------------------------------
    ! lu6L   solves   L v = v(input).
    !
    ! 15 Dec 2002: First version derived from lu6sol.
    ! 15 Dec 2002: Current version.
    ! 13 Dec 2011: First f90 version.
    !------------------------------------------------------------------

    integer(ip) :: i, ipiv, j, k, l, l1, ldummy, len, lenL, lenL0, numL, numL0
    real(rp)    :: small, vpiv

    numL0  = luparm(20)
    lenL0  = luparm(21)
    lenL   = luparm(23)
    small  = parmlu(3)
    inform = 0
    l1     = lena + 1

    do k = 1, numL0
       len   = lenc(k)
       l     = l1
       l1    = l1 - len
       ipiv  = indr(l1)
       vpiv  = v(ipiv)

       if (abs(vpiv) > small) then
          !***** This loop could be coded specially.
          do ldummy = 1, len
             l    = l - 1
             j    = indc(l)
             v(j) = v(j) + a(l)*vpiv
          end do
       end if
    end do

    l      = lena - lenL0 + 1
    numL   = lenL - lenL0

    !***** This loop could be coded specially.

    do ldummy = 1, numL
       l      = l - 1
       i      = indr(l)
       if (abs(v(i)) > small) then
          j    = indc(l)
          v(j) = v(j) + a(l)*v(i)
       end if
    end do

    ! Exit.

    luparm(10) = inform

  end subroutine lu6L

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu6Lt ( inform, m, n, v, &
                     lena, luparm, parmlu, a, indc, indr, lenc )

    integer(ip),   intent(in)    :: m, n, lena
    integer(ip),   intent(in)    :: indc(lena), indr(lena), lenc(n)
    real(rp),      intent(in)    :: a(lena)
    integer(ip),   intent(inout) :: luparm(30)
    real(rp),      intent(inout) :: parmlu(30), v(m)
    integer(ip),   intent(out)   :: inform

    !------------------------------------------------------------------
    ! lu6Lt  solves   L'v = v(input).
    !
    ! 15 Dec 2002: First version derived from lu6sol.
    ! 15 Dec 2002: Current version.
    ! 13 Dec 2011: First f90 version.
    !------------------------------------------------------------------

    integer(ip) :: i, ipiv, j, k, l, l1, l2, len, lenL, lenL0, numL0
    real(rp)    :: small, sum


    numL0  = luparm(20)
    lenL0  = luparm(21)
    lenL   = luparm(23)
    small  = parmlu(3)
    inform = 0
    l1     = lena - lenL + 1
    l2     = lena - lenL0

    !***** This loop could be coded specially.
    do l = l1, l2
       j     = indc(l)
       if (abs(v(j)) > small) then
          i     = indr(l)
          v(i)  = v(i) + a(l)*v(j)
       end if
    end do

    do k = numL0, 1, -1
       len   = lenc(k)
       sum   = zero
       l1    = l2 + 1
       l2    = l2 + len

       !***** This loop could be coded specially.
       do l = l1, l2
          j     = indc(l)
          sum   = sum + a(l)*v(j)
       end do

       ipiv    = indr(l1)
       v(ipiv) = v(ipiv) + sum
    end do

    ! Exit.

    luparm(10) = inform

  end subroutine lu6Lt

  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu6U  ( inform, m, n, v, w, &
                     lena, luparm, parmlu, a, indr, p, q, lenr, locr )

    integer(ip),   intent(in)    :: m, n, lena
    integer(ip),   intent(in)    :: indr(lena), p(m), q(n), lenr(m), locr(m)
    real(rp),      intent(in)    :: a(lena)
    real(rp),      intent(in)    :: v(m)

    integer(ip),   intent(inout) :: luparm(30)
    real(rp),      intent(inout) :: parmlu(30)

    integer(ip),   intent(out)   :: inform
    real(rp),      intent(out)   :: w(n)

    !------------------------------------------------------------------
    ! lu6U   solves   U w = v.          v  is not altered.
    !
    ! 15 Dec 2002: First version derived from lu6sol.
    ! 15 Dec 2002: Current version.
    ! 13 Dec 2011: First f90 version.
    !------------------------------------------------------------------

    integer(ip)            :: i, j, k, klast, l, l1, l2, l3, nrank, nrank1
    real(rp)               :: resid, small, t


    nrank  = luparm(16)
    small  = parmlu(3)
    inform = 0
    nrank1 = nrank + 1
    resid  = zero

    ! Find the first nonzero in v(1:nrank), counting backwards.

    do klast = nrank, 1, -1
       i     = p(klast)
       if (abs(v(i)) > small) exit
    end do

    do k = klast + 1, n
       j    = q(k)
       w(j) = zero
    end do

    ! Do the back-substitution, using rows 1:klast of U.

    do k  = klast, 1, -1
       i  = p(k)
       t  = v(i)
       l1 = locr(i)
       l2 = l1 + 1
       l3 = l1 + lenr(i) - 1

       !***** This loop could be coded specially.
       do l = l2, l3
          j = indr(l)
          t = t - a(l)*w(j)
       end do

       j  = q(k)
       if (abs(t) <= small) then
          w(j) = zero
       else
          w(j) = t/a(l1)
       end if
    end do

    ! Compute residual for overdetermined systems.

    do k = nrank1, m
       i     = p(k)
       resid = resid + abs(v(i))
    end do

    ! Exit.

    if (resid > zero) inform = 1
    luparm(10) = inform
    parmlu(20) = resid

  end subroutine lu6U

  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu6Ut ( inform, m, n, v, w, &
                     lena, luparm, parmlu, a, indr, p, q, lenr, locr )

    integer(ip),   intent(in)    :: m, n, lena
    integer(ip),   intent(in)    :: indr(lena), p(m), q(n), lenr(m), locr(m)
    real(rp),      intent(in)    :: a(lena)
    integer(ip),   intent(inout) :: luparm(30)
    real(rp),      intent(inout) :: parmlu(30), w(n)
    integer(ip),   intent(out)   :: inform
    real(rp),      intent(out)   :: v(m)

    !------------------------------------------------------------------
    ! lu6Ut  solves   U'v = w.          w  is destroyed.
    !
    ! 15 Dec 2002: First version derived from lu6sol.
    ! 15 Dec 2002: Current version.
    ! 13 Dec 2011: First f90 version.
    !------------------------------------------------------------------

    integer(ip)            :: i, j, k, l, l1, l2, nrank, nrank1
    real(rp)               :: resid, small, t


    nrank  = luparm(16)
    small  = parmlu(3)
    inform = 0
    nrank1 = nrank + 1
    resid  = zero

    do k = nrank1, m
       i     = p(k)
       v(i)  = zero
    end do

    ! Do the forward-substitution, skipping columns of U(transpose)
    ! when the associated element of w(*) is negligible.

    do k = 1, nrank
       i      = p(k)
       j      = q(k)
       t      = w(j)
       if (abs(t) <= small) then
          v(i) = zero
          cycle
       end if

       l1     = locr(i)
       t      = t/a(l1)
       v(i)   = t
       l2     = l1 + lenr(i) - 1
       l1     = l1 + 1

       !***** This loop could be coded specially.
       do l = l1, l2
          j    = indr(l)
          w(j) = w(j) - t*a(l)
       end do
    end do

    ! Compute residual for overdetermined systems.

    do k = nrank1, n
       j     = q(k)
       resid = resid + abs(w(j))
    end do

    ! Exit.

    if (resid > zero) inform = 1
    luparm(10) = inform
    parmlu(20) = resid

  end subroutine lu6Ut

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu6LD ( inform, mode, m, n, v, &
                     lena, luparm, parmlu, a, indc, indr, lenc, locr )

    integer(ip),   intent(in)    :: mode, m, n, lena
    integer(ip),   intent(in)    :: indc(lena), indr(lena), lenc(n), locr(m)
    real(rp),      intent(in)    :: a(lena)
    integer(ip),   intent(inout) :: luparm(30)
    real(rp),      intent(inout) :: parmlu(30), v(m)
    integer(ip),   intent(out)   :: inform

    !-------------------------------------------------------------------
    ! lu6LD  assumes lu1fac has computed factors A = LU of a
    ! symmetric definite or quasi-definite matrix A,
    ! using Threshold Symmetric Pivoting (TSP),   luparm(6) = 3,
    ! or    Threshold Diagonal  Pivoting (TDP),   luparm(6) = 4.
    ! It also assumes that no updates have been performed.
    ! In such cases,  U = D L', where D = diag(U).
    ! lu6LDL returns v as follows:
    !
    ! mode
    ! 1    v  solves   L D v = v(input).
    ! 2    v  solves   L|D|v = v(input).
    !
    ! 15 Dec 2002: First version of lu6LD.
    ! 15 Dec 2002: Current version.
    ! 13 Dec 2011: First f90 version.
    !-----------------------------------------------------------------------
    !
    ! Solve L D v(new) = v  or  L|D|v(new) = v, depending on mode.
    ! The code for L is the same as in lu6L,
    ! but when a nonzero entry of v arises, we divide by
    ! the corresponding entry of D or |D|.

    integer(ip)            :: ipiv, j, k, l, l1, ldummy, len, numL0
    real(rp)               :: diag, small, vpiv


    numL0  = luparm(20)
    small  = parmlu(3)
    inform = 0
    l1     = lena + 1

    do k = 1, numL0
       len   = lenc(k)
       l     = l1
       l1    = l1 - len
       ipiv  = indr(l1)
       vpiv  = v(ipiv)

       if (abs(vpiv) > small) then
          !***** This loop could be coded specially.
          do ldummy = 1, len
             l    = l - 1
             j    = indc(l)
             v(j) = v(j) + a(l)*vpiv
          end do

          ! Find diag = U(ipiv,ipiv) and divide by diag or |diag|.

          l    = locr(ipiv)
          diag = a(l)
          if (mode == 2) diag = abs(diag)
          v(ipiv) = vpiv/diag
       end if
    end do

  end subroutine lu6LD

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu6chk( mode, m, n, nslack, w, lena, luparm, parmlu, &
                     a, indc, indr, p, q, lenc, lenr, locc, locr, inform )

    integer(ip),   intent(in)    :: mode, m, n, nslack, lena
    integer(ip),   intent(in)    :: indc(lena), indr(lena), p(m), q(n), &
                                    lenc(n), lenr(m), locc(n), locr(m)
    real(rp),      intent(in)    :: a(lena)

    integer(ip),   intent(inout) :: inform
    integer(ip),   intent(inout) :: luparm(30)
    real(rp),      intent(inout) :: parmlu(30)

    real(rp),      intent(inout) :: w(n)

    !-----------------------------------------------------------------
    ! lu6chk  looks at the LU factorization  A = L*U.
    !
    ! If mode = 1, lu6chk is being called by lu1fac.
    ! (Other modes not yet implemented.)
    ! The important input parameters are
    !
    ! lprint = luparm(2)
    ! luparm(6) = 1 if TRP
    ! keepLU = luparm(8)
    ! Utol1  = parmlu(4)
    ! Utol2  = parmlu(5)
    !
    ! and the significant output parameters are
    !
    ! inform = luparm(10)
    ! nsing  = luparm(11)
    ! jsing  = luparm(12)
    ! jumin  = luparm(19)
    ! Lmax   = parmlu(11)
    ! Umax   = parmlu(12)
    ! DUmax  = parmlu(13)
    ! DUmin  = parmlu(14)
    ! and      w(*).
    !
    ! Lmax  and Umax  return the largest elements in L and U.
    ! DUmax and DUmin return the largest and smallest diagonals of U
    ! (excluding diagonals that are exactly zero).
    !
    ! In general, w(j) is set to the maximum absolute element in
    ! the j-th column of U.  However, if the corresponding diagonal
    ! of U is small in absolute terms or relative to w(j)
    ! (as judged by the parameters Utol1, Utol2 respectively),
    ! then w(j) is changed to - w(j).
    !
    ! Thus, if w(j) is not positive, the j-th column of A
    ! appears to be dependent on the other columns of A.
    ! The number of such columns, and the position of the last one,
    ! are returned as nsing and jsing.
    !
    ! Note that nrank is assumed to be set already, and is not altered.
    ! Typically, nsing will satisfy      nrank + nsing = n,  but if
    ! Utol1 and Utol2 are rather large,  nsing > n - nrank   may occur.
    !
    ! If keepLU = 0,
    !              Lmax  and Umax  are already set by lu1fac.
    !              The diagonals of U are in the top of A.
    !              Only Utol1 is used in the singularity test to set w(*).
    !
    ! inform = 0  if A appears to have full column rank (nsing = 0).
    ! inform = 1  otherwise (nsing > 0).
    !
    ! 00 Jul 1987: Early version.
    ! 09 May 1988: f77 version.
    ! 11 Mar 2001: Allow for keepLU = 0.
    ! 17 Nov 2001: Briefer output for singular factors.
    ! 05 May 2002: Comma needed in format 1100 (via Kenneth Holmstrom).
    ! 06 May 2002: With keepLU = 0, diags of U are in natural order.
    !              They were not being extracted correctly.
    ! 23 Apr 2004: TRP can judge singularity better by comparing
    !              all diagonals to DUmax.
    ! 27 Jun 2004: (PEG) Allow write only if nout > 0 .
    ! 13 Dec 2011: First f90 version.
    ! 12 Dec 2015: nslack ensures slacks are kept with w(j) > 0.
    !------------------------------------------------------------------

    character(1)        :: mnkey
    logical             :: keepLU, TRP
    integer(ip)         :: i, j, jsing, jumin, k, l, l1, l2, ldiagU, lenL, &
                           lprint, ndefic, nout, nrank, nsing
    real(rp)            :: aij, diag, DUmax, DUmin, Lmax, Umax, Utol1, Utol2


    nout   = luparm(1)
    lprint = luparm(2)
    TRP    = luparm(6) == 1  ! Threshold Rook Pivoting
    keepLU = luparm(8) /= 0
    nrank  = luparm(16)
    lenL   = luparm(23)
    Utol1  = parmlu(4)
    Utol2  = parmlu(5)

    inform = 0
    Lmax   = zero
    Umax   = zero
    nsing  = 0
    jsing  = 0
    jumin  = 0
    DUmax  = zero
    DUmin  = 1.0d+30

    ! w(j) is already set by lu1slk.
    ! w(1:n) = zero

    if (keepLU) then
       !--------------------------------------------------------------
       ! Find  Lmax.
       !--------------------------------------------------------------
       do l = lena + 1 - lenL, lena
          Lmax  = max( Lmax, abs(a(l)) )
       end do

       !--------------------------------------------------------------
       ! Find Umax and set w(j) = maximum element in j-th column of U.
       !--------------------------------------------------------------
       do k = nslack + 1, nrank   ! 12 Dec 2015: Allow for nslack.
          i     = p(k)
          l1    = locr(i)
          l2    = l1 + lenr(i) - 1

          do l = l1, l2
             j     = indr(l)
             aij   = abs( a(l) )
             w(j)  = max( w(j), aij )
             Umax  = max( Umax, aij )
          end do
       end do

       parmlu(11) = Lmax
       parmlu(12) = Umax

       !--------------------------------------------------------------
       ! Find DUmax and DUmin, the extreme diagonals of U.
       !--------------------------------------------------------------
       do k = nslack + 1, nrank   ! 12 Dec 2015: Allow for nslack.
          j      = q(k)
          i      = p(k)
          l1     = locr(i)
          diag   = abs( a(l1) )
          DUmax  = max( DUmax, diag )
          if (DUmin > diag) then
             DUmin  =   diag
             jumin  =   j
          end if
       end do

    else
       !--------------------------------------------------------------
       ! keepLU = 0.
       ! Only diag(U) is stored.  Set w(*) accordingly.
       ! Find DUmax and DUmin, the extreme diagonals of U.
       !--------------------------------------------------------------
       ldiagU = lena - n

       do k = nslack + 1, nrank   ! 12 Dec 2015: Allow for nslack.
          j      = q(k)
        ! diag   = abs( a(ldiagU + k) ) ! 06 May 2002: Diags
          diag   = abs( a(ldiagU + j) ) ! are in natural order
          w(j)   = diag
          DUmax  = max( DUmax, diag )
          if (DUmin > diag) then
             DUmin  =   diag
             jumin  =   j
          end if
       end do
    end if


    !--------------------------------------------------------------
    ! Negate w(j) if the corresponding diagonal of U is
    ! too small in absolute terms or relative to the other elements
    ! in the same column of  U.
    !
    ! 23 Apr 2004: TRP ensures that diags are NOT small relative to
    !              other elements in their own column.
    !              Much better, we can compare all diags to DUmax.
    ! 13 Nov 2015: This causes slacks to replace slacks when DUmax
    !              is big.  It seems better to leave Utol1 alone.
    ! 12 Dec 2015: Allow for nslack.
    !              DUmax now excludes slack rows, so we can
    !              reset Utol1 again for TRP.
    !--------------------------------------------------------------
    if (mode == 1  .and.  TRP) then
       Utol1 = max( Utol1, Utol2*DUmax )
    end if

    if (keepLU) then
       do k = nslack + 1, n   ! 12 Dec 2015: Allow for nslack.
          j     = q(k)
          if (k > nrank) then
             diag   = zero
          else
             i      = p(k)
             l1     = locr(i)
             diag   = abs( a(l1) )
          end if

          if (diag <= Utol1  .or.  diag <= Utol2*w(j)) then
             nsing  =   nsing + 1
             jsing  =   j
             w(j)   = - w(j)
          end if
       end do

    else ! keepLU = 0

       do k = nslack + 1, n   ! 12 Dec 2015: Allow for nslack.
          j      = q(k)
          diag   = w(j)

          if (diag <= Utol1) then
             nsing  =   nsing + 1
             jsing  =   j
             w(j)   = - w(j)
          end if
       end do
    end if

    !-----------------------------------------------------------------
    ! Set output parameters.
    !-----------------------------------------------------------------
    if (jumin == 0) DUmin = zero
    luparm(11) = nsing
    luparm(12) = jsing
    luparm(19) = jumin
    parmlu(13) = DUmax
    parmlu(14) = DUmin

    if (nsing > 0) then  ! The matrix has been judged singular.
       inform = 1
       ndefic = n - nrank
       if (nout > 0  .and.  lprint >= 0) then
          if (m > n) then
             mnkey  = '>'
          else if (m == n) then
             mnkey  = '='
          else
             mnkey  = '<'
          end if
          write(nout, 1100) mnkey, nrank, ndefic, nsing
       end if
    end if

    ! Exit.

    luparm(10) = inform
    return

1100 format(' Singular(m', a, 'n)', '  rank', i9, '  n-rank', i8, '  nsing', i9)

  end subroutine lu6chk

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ! File lusol7a.f90
  !
  ! lu7add   lu7cyc   lu7elm   lu7for   lu7rnk   lu7zap
  ! Utilities for LUSOL's update routines.
  ! lu7for is the most important -- the forward sweep.
  !
  ! 01 May 2002: Derived from LUSOL's original lu7a.f file.
  ! 13 Dec 2011: First f90 version.
  ! 20 Jan 2016: Current version of lusol7a.f90.
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu7add( m, n, jadd, v, lena, luparm, parmlu, &
                     lenL, lenU, lrow, nrank,             &
                     a, indr, p, lenr, locr,              &
                     inform, klast, vnorm )

    integer(ip),   intent(in)    :: m, n, jadd, lena, nrank, &
                                    p(m)
    integer(ip),   intent(inout) :: luparm(30), lenL, lenU, lrow, &
                                    indr(lena), lenr(m), locr(m)
    real(rp),      intent(inout) :: parmlu(30), a(lena), v(m)
    integer(ip),   intent(out)   :: inform, klast
    real(rp),      intent(out)   :: vnorm

    !------------------------------------------------------------------
    ! lu7add  inserts the first nrank elements of the vector v(*)
    ! as column jadd of U.  We assume that U does not yet have any
    ! entries in this column.
    ! Elements no larger than parmlu(3) are treated as zero.
    ! klast  will be set so that the last row to be affected
    ! (in pivotal order) is row p(klast).
    !
    ! 09 May 1988: First f77 version.
    ! 13 Dec 2011: First f90 version.
    ! 20 Dec 2015: ilast is now output by lu1rec.
    !------------------------------------------------------------------

    integer(ip)         :: i, ilast, j, k, leni, l, lr1, lr2, minfre, nfree
    real(rp)            :: small

    small  = parmlu(3)
    vnorm  = zero
    klast  = 0

    do k = 1, nrank
       i      = p(k)
       if (abs(v(i)) <= small) cycle
       klast  = k
       vnorm  = vnorm + abs(v(i))
       leni   = lenr(i)

       ! Compress row file if necessary.

       minfre = leni + 1
       nfree  = lena - lenL - lrow
       if (nfree < minfre) then
          call lu1rec( m, .true., luparm, lrow, ilast, &
                       lena, a, indr, lenr, locr )
          nfree  = lena - lenL - lrow
          if (nfree < minfre) go to 970
       end if

       ! Move row i to the end of the row file,
       ! unless it is already there.
       ! No need to move if there is a gap already.

       if (leni == 0) locr(i) = lrow + 1
       lr1    = locr(i)
       lr2    = lr1 + leni - 1
       if (lr2    ==   lrow) go to 150
       if (indr(lr2+1) == 0) go to 180
       locr(i) = lrow + 1

       do l = lr1, lr2
          lrow       = lrow + 1
          a(lrow)    = a(l)
          j          = indr(l)
          indr(l)    = 0
          indr(lrow) = j
       end do

150    lr2     = lrow
       lrow    = lrow + 1

       ! Add the element of  v.

180    lr2       = lr2 + 1
       a(lr2)    = v(i)
       indr(lr2) = jadd
       lenr(i)   = leni + 1
       lenU      = lenU + 1
    end do

    ! Normal exit.

    inform = 0
    go to 990

    ! Not enough storage.

970 inform = 7

990 return

  end subroutine lu7add

  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu7cyc( kfirst, klast, p )

    integer(ip),   intent(in)    :: kfirst, klast
    integer(ip),   intent(inout) :: p(klast)

    !------------------------------------------------------------------
    ! lu7cyc performs a cyclic permutation on the row or column ordering
    ! stored in p, moving entry kfirst down to klast.
    ! If kfirst .ge. klast, lu7cyc should not be called.
    ! Sometimes klast = 0 and nothing should happen.
    !
    ! 09 May 1988: First f77 version.
    ! 13 Dec 2011: First f90 version.
    !------------------------------------------------------------------

    integer(ip)      :: ifirst, k

    if (kfirst < klast) then
       ifirst = p(kfirst)

       do k = kfirst, klast - 1
          p(k) = p(k+1)
       end do

       p(klast) = ifirst
    end if

  end subroutine lu7cyc

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu7elm( m, n, jelm, v, lena, luparm, parmlu,     &
                     lenL, lenU, lrow, nrank,                 &
                     a, indc, indr, p, q, lenr, locc, locr,   &
                     inform, diag )

    integer(ip),   intent(in)    :: m, n, jelm, lena, nrank
    integer(ip),   intent(in)    :: lenU, q(n)   ! not used
    real(rp),      intent(in)    :: v(m)
    integer(ip),   intent(inout) :: luparm(30), lenL, lrow,       &
                                    indc(lena), indr(lena), p(m), &
                                    lenr(m), locc(n), locr(m)
    real(rp),      intent(inout) :: parmlu(30), a(lena)

    integer(ip),   intent(out)   :: inform
    real(rp),      intent(out)   :: diag

    !------------------------------------------------------------------
    ! lu7elm  eliminates the subdiagonal elements of a vector  v(*),
    ! where  L*v = y  for some vector y.
    ! If  jelm > 0,  y  has just become column  jelm  of the matrix  A.
    ! lu7elm  should not be called unless  m  is greater than  nrank.
    !
    ! inform = 0 if y contained no subdiagonal nonzeros to eliminate.
    ! inform = 1 if y contained at least one nontrivial subdiagonal.
    ! inform = 7 if there is insufficient storage.
    !
    ! 09 May 1988: First f77 version.
    !              No longer calls lu7for at end.  lu8rpc, lu8mod do so.
    ! 13 Dec 2011: First f90 version.
    ! 20 Dec 2015: ilast is now output by lu1rec.
    !------------------------------------------------------------------

    integer(ip)            :: i, ilast, imax, k, kmax, l, l1, l2, lmax, &
                              minfre, nfree, nrank1
    real(rp)               :: small, vi, vmax

    small  = parmlu(3)
    nrank1 = nrank + 1
    diag   = zero

    ! Compress row file if necessary.

    minfre = m - nrank
    nfree  = lena - lenL - lrow
    if (nfree < minfre) then
       call lu1rec( m, .true., luparm, lrow, ilast, &
                    lena, a, indr, lenr, locr )
       nfree  = lena - lenL - lrow
       if (nfree < minfre) go to 970
    end if

    ! Pack the subdiagonals of  v  into  L,  and find the largest.

    vmax   = zero
    kmax   = 0
    l      = lena - lenL + 1

    do k = nrank1, m
       i       = p(k)
       vi      = abs(v(i))
       if (vi <= small) cycle
       l       = l - 1
       a(l)    = v(i)
       indc(l) = i
       if (vmax >= vi ) cycle
       vmax    = vi
       kmax    = k
       lmax    = l
    end do

    if (kmax == 0) go to 900

    !------------------------------------------------------------------
    ! Remove  vmax  by overwriting it with the last packed  v(i).
    ! Then set the multipliers in  L  for the other elements.
    !------------------------------------------------------------------
    imax       = p(kmax)
    vmax       = a(lmax)
    a(lmax)    = a(l)
    indc(lmax) = indc(l)
    l1         = l + 1
    l2         = lena - lenL
    lenL       = lenL + (l2 - l)

    do l = l1, l2
       a(l)    = - a(l) / vmax
       indr(l) =   imax
    end do

    ! Move the row containing vmax to pivotal position nrank + 1.

    p(kmax  ) = p(nrank1)
    p(nrank1) = imax
    diag      = vmax

    !------------------------------------------------------------------
    ! If jelm is positive, insert  vmax  into a new row of  U.
    ! This is now the only subdiagonal element.
    !------------------------------------------------------------------

    if (jelm > 0) then
       lrow       = lrow + 1
       locr(imax) = lrow
       lenr(imax) = 1
       a(lrow)    = vmax
       indr(lrow) = jelm
    end if

    inform = 1
    go to 990

    ! No elements to eliminate.

900 inform = 0
    go to 990

    ! Not enough storage.

970 inform = 7

990 return

  end subroutine lu7elm

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu7for( m, n, kfirst, klast, lena, luparm, parmlu, &
                     lenL, lenU, lrow,                          &
                     a, indc, indr, p, q, lenr, locc, locr,     &
                     inform, diag )

    integer(ip),   intent(in)    :: m, n, kfirst, klast, lena
    integer(ip),   intent(in)    :: q(n)
    integer(ip),   intent(inout) :: luparm(30), lenL, lenU, lrow
    integer(ip),   intent(inout) :: indc(lena), indr(lena),     &
                                    p(m), lenr(m), locc(n), locr(m)
    real(rp),      intent(inout) :: parmlu(30), a(lena)

    integer(ip),   intent(out)   :: inform
    real(rp),      intent(out)   :: diag

    !------------------------------------------------------------------
    ! lu7for  (forward sweep) updates the LU factorization A = L*U
    ! when row iw = p(klast) of U is eliminated by a forward
    ! sweep of stabilized row operations, leaving p*U*q upper triangular.
    !
    ! The row permutation p is updated to preserve stability and/or
    ! sparsity.  The column permutation q is not altered.
    !
    ! kfirst  is such that row p(kfirst) is the first row involved
    ! in eliminating row  iw.  (Hence,  kfirst  marks the first nonzero
    ! in row  iw  in pivotal order.)  If  kfirst  is unknown it may be
    ! input as  1.
    !
    ! klast   is such that row p(klast) is the row being eliminated.
    ! klast   is not altered.
    !
    ! lu7for  should be called only if  kfirst .le. klast.
    ! If  kfirst = klast,  there are no nonzeros to eliminate, but the
    ! diagonal element of row p(klast) may need to be moved to the
    ! front of the row.
    !
    ! On entry,  locc(*)  must be zero.
    !
    ! On exit:
    ! inform = 0  if row iw has a nonzero diagonal (could be small).
    ! inform = 1  if row iw has no diagonal.
    ! inform = 7  if there is not enough storage to finish the update.
    !
    ! On a successful exit (inform le 1),  locc(*)  will again be zero.
    !
    !    Jan 1985: Final f66 version.
    ! 09 May 1988: First f77 version.
    ! 13 Dec 2011: First f90 version.
    ! 20 Dec 2015: ilast is now output by lu1rec.
    !------------------------------------------------------------------

    logical               :: swappd
    integer(ip)           :: ilast, iv, iw, j, jfirst, jlast, jv, &
                             k, kbegin, kstart, kstop,            &
                             l, ldiag, lenv, lenw, lfirst, limit, &
                             lv, lv1, lv2, lv3, lw, lw1, lw2,     &
                             minfre, nfree
    real(rp)              :: amult, Ltol, Uspace, small, vj, wj


    Ltol   = parmlu(2)
    small  = parmlu(3)
    Uspace = parmlu(6)
    kbegin = kfirst
    swappd = .false.

    ! We come back here from below if a row interchange is performed.

100 iw     = p(klast)
    lenw   = lenr(iw)
    if (lenw   ==   0  ) go to 910
    lw1    = locr(iw)
    lw2    = lw1 + lenw - 1
    jfirst = q(kbegin)
    if (kbegin >= klast) go to 700

    ! Make sure there is room at the end of the row file
    ! in case row  iw  is moved there and fills in completely.

    minfre = n + 1
    nfree  = lena - lenL - lrow
    if (nfree < minfre) then
       call lu1rec( m, .true., luparm, lrow, ilast, &
                    lena, a, indr, lenr, locr )
       lw1    = locr(iw)
       lw2    = lw1 + lenw - 1
       nfree  = lena - lenL - lrow
       if (nfree < minfre) go to 970
    end if

    ! Set markers on row iw.

    do l = lw1, lw2
       j       = indr(l)
       locc(j) = l
    end do

    !==================================================================
    ! Main elimination loop.
    !==================================================================
    kstart = kbegin
    kstop  = min( klast, n )

    do k = kstart, kstop
       jfirst = q(k)
       lfirst = locc(jfirst)
       if (lfirst == 0) go to 490

       ! Row  iw  has its first element in column  jfirst.

       wj     = a(lfirst)
       if (k == klast) go to 490

       !---------------------------------------------------------------
       ! We are about to use the first element of row iv
       ! to eliminate the first element of row iw.
       ! However, we may wish to interchange the rows instead,
       ! to preserve stability and/or sparsity.
       !---------------------------------------------------------------
       iv     = p(k)
       lenv   = lenr(iv)
       lv1    = locr(iv)
       vj     = zero
       if (lenv      ==   0   ) go to 150
       if (indr(lv1) /= jfirst) go to 150
       vj     = a(lv1)
       if (         swappd          ) go to 200
       if (Ltol * abs(wj) <  abs(vj)) go to 200
       if (Ltol * abs(vj) <  abs(wj)) go to 150
       if (          lenv <= lenw   ) go to 200

       !---------------------------------------------------------------
       ! Interchange rows iv and iw.
       !---------------------------------------------------------------
150    p(klast) = iv
       p(k)     = iw
       kbegin   = k
       swappd   = .true.
       go to 600

       !---------------------------------------------------------------
       ! Delete the eliminated element from row iw
       ! by overwriting it with the last element.
       !---------------------------------------------------------------
200    a(lfirst)    = a(lw2)
       jlast        = indr(lw2)
       indr(lfirst) = jlast
       indr(lw2)    = 0
       locc(jlast)  = lfirst
       locc(jfirst) = 0
       lenw         = lenw - 1
       lenU         = lenU - 1
       if (lrow == lw2) lrow = lrow - 1
       lw2          = lw2  - 1

       !---------------------------------------------------------------
       ! Form the multiplier and store it in the L file.
       !---------------------------------------------------------------
       if (abs(wj) <= small) go to 490
       amult   = - wj/vj
       l       = lena - lenL
       a(l)    = amult
       indr(l) = iv
       indc(l) = iw
       lenL    = lenL + 1

       !---------------------------------------------------------------
       ! Add the appropriate multiple of row iv to row iw.
       ! We use two different inner loops.  The first one is for the
       ! case where row iw is not at the end of storage.
       !---------------------------------------------------------------
       if (lenv == 1) go to 490
       lv2    = lv1 + 1
       lv3    = lv1 + lenv - 1
       if (lw2 == lrow) go to 400

       !...............................................................
       ! This inner loop will be interrupted only if
       ! fill-in occurs enough to bump into the next row.
       !...............................................................
       do lv = lv2, lv3
          jv = indr(lv)
          lw = locc(jv)

          if (lw > 0) then         ! No fill-in.
             a(lw) = a(lw) + amult*a(lv)
             if (abs(a(lw)) <= small) then  ! Delete small computed element.
                a(lw)     = a(lw2)
                j         = indr(lw2)
                indr(lw)  = j
                indr(lw2) = 0
                locc(j)   = lw
                locc(jv)  = 0
                lenU      = lenU - 1
                lenw      = lenw - 1
                lw2       = lw2  - 1
             end if

          else    ! Row iw doesn't have an element in column jv yet
                  ! so there is a fill-in.
             if (indr(lw2+1) /= 0) go to 360
             lenU      = lenU + 1
             lenw      = lenw + 1
             lw2       = lw2  + 1
             a(lw2)    = amult * a(lv)
             indr(lw2) = jv
             locc(jv)  = lw2
          end if
       end do

       go to 490

       ! Fill-in interrupted the previous loop.
       ! Move row  iw  to the end of the row file.

360    lv2      = lv
       locr(iw) = lrow + 1

       do l = lw1, lw2
          lrow       = lrow + 1
          a(lrow)    = a(l)
          j          = indr(l)
          indr(l)    = 0
          indr(lrow) = j
          locc(j)    = lrow
       end do

       lw1    = locr(iw)
       lw2    = lrow

       !...............................................................
       ! Inner loop with row iw at the end of storage.
       !...............................................................
400    do lv = lv2, lv3
          jv     = indr(lv)
          lw     = locc(jv)

          if (lw > 0) then       ! No fill-in
             a(lw) = a(lw) + amult*a(lv)
             if (abs(a(lw)) <= small) then    ! Delete small computed element
                a(lw)     = a(lw2)
                j         = indr(lw2)
                indr(lw)  = j
                indr(lw2) = 0
                locc(j)   = lw
                locc(jv)  = 0
                lenU      = lenU - 1
                lenw      = lenw - 1
                lw2       = lw2  - 1
             end if

          else           ! Row iw doesn't have an element in column jv yet
                         ! so there is a fill-in
             lenU      = lenU + 1
             lenw      = lenw + 1
             lw2       = lw2  + 1
             a(lw2)    = amult * a(lv)
             indr(lw2) = jv
             locc(jv)  = lw2
          end if
       end do

       lrow   = lw2

       ! The k-th element of row iw has been processed.
       ! Reset swappd before looking at the next element.

490    swappd = .false.
    end do

    !=================================================================
    ! End of main elimination loop.
    !==================================================================

    ! Cancel markers on row iw.

600 lenr(iw) = lenw
    if (lenw == 0) go to 910
    do l = lw1, lw2
       j       = indr(l)
       locc(j) = 0
    end do

    ! Move the diagonal element to the front of row iw.
    ! At this stage, lenw > 0 and klast <= n.

700 do l = lw1, lw2
       ldiag = l
       if (indr(l) == jfirst) go to 730  ! not exit !!!
    end do
    go to 910

730 diag        = a(ldiag)
    a(ldiag)    = a(lw1)
    a(lw1)      = diag
    indr(ldiag) = indr(lw1)
    indr(lw1)   = jfirst

    ! If an interchange is needed, repeat from the beginning with the
    ! new row iw, knowing that the opposite interchange cannot occur.

    if (swappd) go to 100
    inform = 0
    go to 950

    ! Singular

910 diag   = zero
    inform = 1

    ! Force a compression if the file for U is much longer than the
    ! no. of nonzeros in U (i.e. if lrow is much bigger than lenU).
    ! This should prevent memory fragmentation when there is far more
    ! memory than necessary (i.e. when lena is huge).

950 limit  = int(Uspace*real(lenU)) + m + n + 1000
    if (lrow > limit) then
       call lu1rec( m, .true., luparm, lrow, ilast, &
                    lena, a, indr, lenr, locr )
    end if
    go to 990

    ! Not enough storage.

970 inform = 7

    ! Exit.

990 return

  end subroutine lu7for

  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu7rnk( m, n, jsing, lena, parmlu,             &
                     lenL, lenU, lrow, nrank,               &
                     a, indc, indr, p, q, lenr, locc, locr, &
                     inform, diag )

    integer(ip),   intent(in)    :: m, n, jsing, lena,      &
                                    p(m)
    integer(ip),   intent(inout) :: lenL, lenU, lrow, nrank, &
                                    indc(lena), indr(lena), q(n),        &
                                    lenr(m), locc(n), locr(m)
    real(rp),      intent(inout) :: parmlu(30)  ! not used
    real(rp),      intent(inout) :: a(lena)
    integer(ip),   intent(out)   :: inform
    real(rp),      intent(out)   :: diag

    !------------------------------------------------------------------
    ! lu7rnk (check rank) assumes U is currently nrank by n
    ! and determines if row nrank contains an acceptable pivot.
    ! If not, the row is deleted and nrank is decreased by 1.

    ! jsing is an input parameter (not altered).  If jsing is positive,
    ! column jsing has already been judged dependent.  A substitute
    ! (if any) must be some other column.
    !
    ! On exit,
    ! inform = -1 if nrank decreases by 1
    !        =  0 if nrank stays the same
    !        =  1 if there's a fatal error.  (Currently we stop.)
    !
    ! -- Jul 1987: First version.
    ! 09 May 1988: First f77 version.
    ! 13 Dec 2011: First f90 version.
    ! 01 Jan 2012: luparm not used.
    !------------------------------------------------------------------

    integer(ip)             :: iw, jmax, kmax, l, l1, l2, lenw, lmax
    real(rp)                :: Umax, Utol1

    Utol1    = parmlu(4)
    diag     = zero

    ! Find Umax, the largest element in row nrank.

    iw       = p(nrank)
    lenw     = lenr(iw)
    if (lenw == 0) go to 400
    l1       = locr(iw)
    l2       = l1 + lenw - 1
    Umax     = zero
    lmax     = l1

    do l = l1, l2
       if (Umax < abs(a(l))) then
          Umax  = abs(a(l))
          lmax  = l
       end if
    end do

    ! Find which column that guy is in (in pivotal order).
    ! Interchange him with column nrank, then move him to be
    ! the new diagonal at the front of row nrank.

    diag   = a(lmax)
    jmax   = indr(lmax)
    l      = 0

    do kmax = nrank, n
       if (q(kmax) == jmax) then
          l = kmax   ! l allows check below for fatal error
          exit
       end if
    end do

    if (l == 0) go to 800   ! Fatal error

    q(kmax)    = q(nrank)
    q(nrank)   = jmax
    a(lmax)    = a(l1)
    a(l1)      = diag
    indr(lmax) = indr(l1)
    indr(l1)   = jmax

    ! See if the new diagonal is big enough.

    if (Umax <= Utol1) go to 400
    if (jmax == jsing) go to 400

    !------------------------------------------------------------------
    ! The rank stays the same.
    !------------------------------------------------------------------
    inform = 0
    go to 900

    !------------------------------------------------------------------
    ! The rank decreases by one.
    !------------------------------------------------------------------
400 inform = -1
    nrank  = nrank - 1

    if (lenw > 0) then       ! Delete row nrank from U.
       lenU     = lenU - lenw
       lenr(iw) = 0
       do l = l1, l2
          indr(l) = 0
       end do

       if (l2 == lrow) then
          ! This row was at the end of the data structure.
          ! We have to reset lrow.
          ! Preceding rows might already have been deleted, so we
          ! have to be prepared to go all the way back to 1.

          do l = 1, l2
             if (indr(lrow) > 0) go to 900
             lrow  = lrow - 1
          end do
       end if
    end if
    go to 900

    ! 15 Dec 2011: Fatal error (should never happen!).
    ! This is a safeguard during work on the f90 version.

800 inform = 1
    write(*,*) 'Fatal error in LUSOL lu7rnk.  Stopping now'
    stop

900 return

  end subroutine lu7rnk

  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu7zap( m, n, jzap, kzap, lena, lenU, lrow, nrank, &
                     a, indr, p, q, lenr, locr )

    integer(ip),   intent(in)    :: m, n, jzap, lena, nrank, &
                                    p(m)
    integer(ip),   intent(inout) :: lenU, lrow, &
                                    indr(lena), q(n), lenr(m), locr(m)
    real(rp),      intent(inout) :: a(lena)
    integer(ip),   intent(out)   :: kzap

    !------------------------------------------------------------------
    ! lu7zap  eliminates all nonzeros in column  jzap  of  U.
    ! It also sets  kzap  to the position of  jzap  in pivotal order.
    ! Thus, on exit we have  q(kzap) = jzap.
    !
    ! -- Jul 1987: nrank added.
    ! 10 May 1988: First f77 version.
    ! 13 Dec 2011: First f90 version.
    !------------------------------------------------------------------

    integer(ip)            :: i, k, leni, l, lr1, lr2

    do k = 1, nrank
       i      = p(k)
       leni   = lenr(i)
       if (leni == 0) go to 90
       lr1    = locr(i)
       lr2    = lr1 + leni - 1
       do l = lr1, lr2
          if (indr(l) == jzap) go to 60
       end do
       go to 90

       ! Delete the old element.

60     a(l)      = a(lr2)
       indr(l)   = indr(lr2)
       indr(lr2) = 0
       lenr(i)   = leni - 1
       lenU      = lenU - 1

       ! Stop if we know there are no more rows containing  jzap.

90     kzap   = k
       if (q(k) == jzap) go to 800
    end do

    ! nrank must be smaller than n because we haven't found kzap yet.

    do k = nrank+1, n
       kzap  = k
       if (q(k) == jzap) exit
    end do

    ! See if we zapped the last element in the file.

800 if (lrow > 0) then
       if (indr(lrow) == 0) lrow = lrow - 1
    end if

  end subroutine lu7zap

  !*********************************************************************
  ! File lusol8a.f90
  !
  ! lu8rpc
  !
  ! Sparse LU update: Replace Column
  ! LUSOL's sparse implementation of the Bartels-Golub update.
  !
  ! 01 May 2002: Derived from LUSOL's original lu8a.f file.
  ! 01 May 2002: Current version of lusol8a.f.
  ! 15 Sep 2004: Test nout. gt. 0 to protect write statements.
  ! 13 Dec 2011: First f90 version.
  ! 20 Jan 2016: Current version of lusol8a.f90.
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lu8rpc( mode1, mode2, m, n, jrep, v, w, &
                     lena, luparm, parmlu,           &
                     a, indc, indr, p, q,            &
                     lenc, lenr, locc, locr,         &
                     inform, diag, vnorm )

    integer(ip),   intent(in)    :: mode1, mode2, m, n, jrep, lena
    integer(ip),   intent(inout) :: luparm(30), &
                                    indc(lena), indr(lena), p(m), q(n), &
                                    lenc(n), lenr(m), locc(n), locr(m)
    real(rp),      intent(inout) :: parmlu(30), a(lena), v(m), &
                                    w(n)  ! not used
    integer(ip),   intent(out)   :: inform
    real(rp),      intent(out)   :: diag, vnorm

    !------------------------------------------------------------------
    ! lu8rpc  updates the LU factorization A = L*U when column jrep
    ! is replaced by some vector v = a(new).
    !
    ! lu8rpc  is an implementation of the Bartels-Golub update,
    ! designed for the case where A is rectangular and/or singular.
    ! L is a product of stabilized eliminations (m x m, nonsingular).
    ! P U Q is upper trapezoidal (m x n, rank nrank).
    !
    ! If  mode1 = 0,  the old column is taken to be zero
    ! (so it does not have to be removed from U).
    !
    ! If  mode1 = 1,  the old column need not have been zero.
    !
    ! If  mode2 = 0,  the new column is taken to be zero.
    !                 v(*) is not used or altered.
    !
    ! If  mode2 = 1,  v(*) must contain the new column a(new).
    ! On exit,  v(*)  will satisfy L*v = a(new).
    !
    ! If  mode2 = 2,  v(*) must satisfy L*v = a(new).
    !
    ! The array w(*) is not used or altered.
    !
    ! On entry, all elements of locc are assumed to be zero.
    ! On a successful exit (inform /= 7), this will again be true.
    !
    ! On exit:
    ! inform = -1  if the rank of U decreased by 1.
    ! inform =  0  if the rank of U stayed the same.
    ! inform =  1  if the rank of U increased by 1.
    ! inform =  2  if the update seemed to be unstable
    !              (diag much bigger than vnorm).
    ! inform =  7  if the update was not completed (lack of storage).
    ! inform =  8  if jrep is not between 1 and n.
    !
    ! -- Jan 1985: Original F66 version.
    ! -- Jul 1987: Modified to maintain U in trapezoidal form.
    ! 10 May 1988: First f77 version.
    ! 16 Oct 2000: Added test for instability (inform = 2).
    ! 13 Dec 2011: First f90 version.
    !------------------------------------------------------------------

    logical                :: singlr
    integer(ip)            :: iw, j1, jsing, klast, krep, &
                              l1, lenL, lenU, lprint, lrow, nout, nrank, nrank0
    real(rp)               :: Utol1, Utol2

    nout   = luparm(1)
    lprint = luparm(2)
    nrank  = luparm(16)
    lenL   = luparm(23)
    lenU   = luparm(24)
    lrow   = luparm(25)
    Utol1  = parmlu(4)
    Utol2  = parmlu(5)
    nrank0 = nrank
    diag   = zero
    vnorm  = zero
    if (jrep < 1) go to 980
    if (jrep > n) go to 980

    !------------------------------------------------------------------
    ! If mode1 = 0, there are no elements to be removed from  U
    ! but we still have to set  krep  (using a backward loop).
    ! Otherwise, use lu7zap to remove column  jrep  from  U
    ! and set  krep  at the same time.
    !------------------------------------------------------------------
    if (mode1 == 0) then
       krep   = n + 1

10     krep   = krep - 1
       if (q(krep) /= jrep) go to 10
    else
       call lu7zap( m, n, jrep, krep, lena, lenU, lrow, nrank, &
                    a, indr, p, q, lenr, locr )
    end if

    !------------------------------------------------------------------
    ! Insert a new column of u and find klast.
    !------------------------------------------------------------------

    if (mode2 == 0) then
       klast  = 0
    else
       if (mode2 == 1) then

          ! Transform v = a(new) to satisfy  L*v = a(new).

          call lu6sol( i1, m, n, v, w, lena, luparm, parmlu, &
                       a, indc, indr, p, q,                  &
                       lenc, lenr, locc, locr, inform )
       end if

       ! Insert into U any nonzeros in the top of v.
       ! row p(klast) will contain the last nonzero in pivotal order.
       ! Note that klast will be in the range ( 0, nrank ).

       call lu7add( m, n, jrep, v,           &
                    lena, luparm, parmlu,    &
                    lenL, lenU, lrow, nrank, &
                    a, indr, p, lenr, locr,  &
                    inform, klast, vnorm )
       if (inform == 7) go to 970
    end if

    !------------------------------------------------------------------
    ! In general, the new column causes U to look like this:
    !
    !                 krep        n                 krep  n
    !
    !                ....a.........          ..........a...
    !                 .  a        .           .        a  .
    !                  . a        .            .       a  .
    !                   .a        .             .      a  .
    !        P U Q =     a        .    or        .     a  .
    !                    b.       .               .    a  .
    !                    b .      .                .   a  .
    !                    b  .     .                 .  a  .
    !                    b   ......                  ..a...  nrank
    !                    c                             c
    !                    c                             c
    !                    c                             c     m
    !
    !     klast points to the last nonzero "a" or "b".
    !     klast = 0 means all "a" and "b" entries are zero.
    !------------------------------------------------------------------

    if (mode2 == 0) then
       if (krep > nrank) go to 900
    else if (nrank < m) then

       ! Eliminate any "c"s (in either case).
       ! Row nrank + 1 may end up containing one nonzero.

       call lu7elm( m, n, jrep, v, lena, luparm, parmlu,     &
                    lenL, lenU, lrow, nrank,                 &
                    a, indc, indr, p, q, lenr, locc, locr,   &
                    inform, diag )
       if (inform == 7) go to 970

       if (inform == 1) then

          ! The nonzero is apparently significant.
          ! Increase nrank by 1 and make klast point to the bottom.

          nrank = nrank + 1
          klast = nrank
       end if
    end if

    if (nrank < n) then

       ! The column rank is low.
       !
       ! In the first case, we want the new column to end up in
       ! position nrank, so the trapezoidal columns will have a chance
       ! later on (in lu7rnk) to pivot in that position.
       !
       ! Otherwise the new column is not part of the triangle.  We
       ! swap it into position nrank so we can judge it for singularity.
       ! lu7rnk might choose some other trapezoidal column later.

       if (krep < nrank) then
          klast    = nrank
       else
          q(krep ) = q(nrank)
          q(nrank) = jrep
          krep     = nrank
       end if
    end if

    !------------------------------------------------------------------
    ! If krep < klast, there are some "b"s to eliminate:
    !
    !                  krep
    !
    !                ....a.........
    !                 .  a        .
    !                  . a        .
    !                   .a        .
    !        P U Q =     a        .  krep
    !                    b.       .
    !                    b .      .
    !                    b  .     .
    !                    b   ......  nrank
    !
    !     If krep == klast, there are no "b"s, but the last "a" still
    !     has to be moved to the front of row krep (by lu7for).
    !------------------------------------------------------------------

    if (krep <= klast) then

       ! Perform a cyclic permutation on the current pivotal order,
       ! and eliminate the resulting row spike.  krep becomes klast.
       ! The final diagonal (if any) will be correctly positioned at
       ! the front of the new krep-th row.  nrank stays the same.

       call lu7cyc( krep, klast, p )
       call lu7cyc( krep, klast, q )

       call lu7for( m, n, krep, klast,    &
                    lena, luparm, parmlu, &
                    lenL, lenU, lrow,     &
                    a, indc, indr, p, q, lenr, locc, locr, &
                    inform, diag )
       if (inform == 7) go to 970
       krep   = klast

       ! Test for instability (diag much bigger than vnorm).

       singlr = vnorm < Utol2 * abs(diag)
       if ( singlr ) go to 920
    end if

    !------------------------------------------------------------------
    ! Test for singularity in column krep (where krep .le. nrank).
    !------------------------------------------------------------------

    diag   = zero
    iw     = p(krep)
    singlr = lenr(iw) == 0

    if (.not. singlr) then
       l1     = locr(iw)
       j1     = indr(l1)
       singlr = j1 /= jrep

       if (.not. singlr) then
          diag   = a(l1)
          singlr = abs( diag ) <= Utol1          .or. &
                   abs( diag ) <= Utol2 * vnorm
       end if
    end if

    if (singlr  .and.  krep < nrank) then

       ! Perform cyclic permutations to move column jrep to the end.
       ! Move the corresponding row to position nrank
       ! then eliminate the resulting row spike.

       call lu7cyc( krep, nrank, p )
       call lu7cyc( krep, n    , q )

       call lu7for( m, n, krep, nrank,    &
                    lena, luparm, parmlu, &
                    lenL, lenU, lrow,     &
                    a, indc, indr, p, q, lenr, locc, locr, &
                    inform, diag )
       if (inform == 7) go to 970
    end if

    ! Find the best column to be in position nrank.
    ! If singlr, it can't be the new column, jrep.
    ! If nothing satisfactory exists, nrank will be decreased.

    if (singlr  .or.  nrank < n) then
       jsing  = 0
       if ( singlr ) jsing = jrep

       call lu7rnk( m, n, jsing, lena, parmlu,             &
                    lenL, lenU, lrow, nrank,               &
                    a, indc, indr, p, q, lenr, locc, locr, &
                    inform, diag )
    end if

    !------------------------------------------------------------------
    ! Set inform for exit.
    !------------------------------------------------------------------
900 if (nrank == nrank0) then
       inform =  0
    else if (nrank < nrank0) then
       inform = -1
       if (nrank0 == n) then
          if (nout > 0  .and.  lprint >= 0) write(nout, 1100) jrep, diag
       end if
    else
       inform =  1
    end if
    go to 990

    ! Instability.

920 inform = 2
    if (nout > 0  .and.  lprint >= 0) write(nout, 1200) jrep, diag
    go to 990

    ! Not enough storage.

970 inform = 7
    if (nout > 0  .and.  lprint >= 0) write(nout, 1700) lena
    go to 990

    ! jrep  is out of range.

980 inform = 8
    if (nout > 0  .and.  lprint >= 0) write(nout, 1800) m, n, jrep

    ! Exit.

990 luparm(10) = inform
    luparm(15) = luparm(15) + 1
    luparm(16) = nrank
    luparm(23) = lenL
    luparm(24) = lenU
    luparm(25) = lrow
    return

1100 format(/ ' lu8rpc  warning.  Singularity after replacing column.', &
              '    jrep =', i8, '    diag =', es12.2 )
1200 format(/ ' lu8rpc  warning.  Instability after replacing column.', &
              '    jrep =', i8, '    diag =', es12.2 )
1700 format(/ ' lu8rpc  error...  Insufficient storage.', &
              '    lena =', i8)
1800 format(/ ' lu8rpc  error...  jrep  is out of range.', &
              '    m =', i8, '    n =', i8, '    jrep =', i8)

  end subroutine lu8rpc

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  function jdamax( n, x, incx )  result(iAmax)

    integer(ip), intent(in)    :: n, incx
    real(rp),    intent(in)    :: x(:)
    integer(ip)                :: iAmax

    !===========================================================================
    ! jdamax does the same as idamax in most cases.
    ! jdamax > 0 if x contains normal values.
    ! jdamax = 0 if n = 0.
    ! jdamax < 0 means x(-jdamax) contains the first NaN or Inf.
    !
    ! 29 Jul 2003: First version of jdamax implemented for s5setx.
    ! 29 Jul 2003: Current version of jdamax
    ! 15 Mar 2008: First f90 version.
    !===========================================================================

    intrinsic           :: huge
    integer(ip)         :: i, ix, kmax
    real(rp)            :: dmax, xi
    real(rp), parameter :: realmax = huge(realmax)

    if (n < 1) then
       iAmax = 0
       return
    end if

    dmax  = zero
    ix    = 1
    kmax  = 1

    do  i = 1, n
       xi = abs( x(ix) )
       if (xi <= realmax) then  ! false if xi = Nan or Inf
          if (dmax < xi) then
             dmax   = xi
             kmax   = ix
          end if
       else
          go to 800
       end if
       ix = ix + incx
    end do

    iAmax = kmax
    return

800 iAmax = -ix

  end function jdamax

end module lusol
