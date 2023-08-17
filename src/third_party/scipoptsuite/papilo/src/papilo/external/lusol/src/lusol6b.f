!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!
!     File  lusol6b.f
!
!     lu6mul   lu6prt   lu6set
!
! 03 Mar 2004: lusol6b.f is essentially lu6b.for from VMS days.
!              integer*4 changed to integer  .
! 14 Jul 2011: lu6mul's v = L'*v fixed.
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine lu6mul( mode, m, n, v, w,
     $                   lena, luparm, parmlu,
     $                   a, indc, indr, ip, iq,
     $                   lenc, lenr, locc, locr )

      implicit           double precision (a-h,o-z)
      integer            luparm(30)
      double precision   parmlu(30), a(lena), v(m), w(n)
      integer            indc(lena), indr(lena), ip(m), iq(n)
      integer            lenc(n), lenr(m)
      integer            locc(n), locr(m)

!     ------------------------------------------------------------------
!     lu6mul  uses the factorization   A = L*U   as follows...
!
!     mode
!     ----
!      1    v  is changed to  L*v.             w  is not touched.
!      2    v  is changed to  L(t)*v.          w  is not touched.
!      3    v  is set to      U*w.             w  is not altered.
!      4    w  is set to      U(t)*v.          v  is not altered.
!      5    v  is set to      A*w.             w  is not altered.
!      6    w  is set to      A(t)*v.          v  is changed to  L(t)*v.
!
!     If  mode .ge. 3,  v  and  w  must not be the same arrays.
!
!     lenc(*)  and  locc(*)  are not used.
!
!     09 May 1988: First F77 version.
!     30 Jan 1996: Converted to lower case (finally!).
!     03 Mar 2004: Time to abolish the computed go to!!
!     ------------------------------------------------------------------

      parameter        ( zero = 0.0d+0 )

      nrank  = luparm(16)
      lenl   = luparm(23)
      !!!! go to (100, 200, 300, 400, 300, 200), mode
      if (mode .eq. 1) go to 100
      if (mode .eq. 2) go to 200
      if (mode .eq. 3) go to 300
      if (mode .eq. 4) go to 400
      if (mode .eq. 5) go to 300
      if (mode .eq. 6) go to 200

!     ==================================================================
!     mode = 1 or 5.    Set  v = L*v.
!     ==================================================================
  100 l1     = lena + 1 - lenl
      do l = l1, lena
         i     = indr(l)
         if (v(i) .ne. zero) then
            j     = indc(l)
            v(j)  = v(j) - a(l)*v(i)
         end if
      end do

      return

!     ==================================================================
!     mode = 2 or 6.    Set  v = L(t)*v.
!     14 Jul 2011: We have to run forward thru the columns of L.
!                  The first column is at the end of memory.
!     ==================================================================
  200 l1 = lena + 1 - lenl
      do l = lena, l1, -1
         j     = indc(l)
         if (v(j) .ne. zero) then
            i     = indr(l)
            v(i)  = v(i)  -  a(l) * v(j)
         end if
      end do

      if (mode .eq. 6) go to 400
      return

!     ==================================================================
!     mode = 3 or 5.    set  v = U*w.
!     ==================================================================

      ! Find the last nonzero in  w(*).

  300 do klast = n, 1, -1
         j     = iq(klast)
         if (w(j) .ne. zero) go to 320
      end do

  320 klast  = min( klast, nrank )
      do k = klast + 1, m
         i     = ip(k)
         v(i)  = zero
      end do

      ! Form U*w, using rows 1 to klast of U.

      do k = 1, klast
         t     = zero
         i     = ip(k)
         l1    = locr(i)
         l2    = l1 + lenr(i) - 1

         do l = l1, l2
            j     = indr(l)
            t     = t  +  a(l) * w(j)
         end do

         v(i)  = t
      end do

      if (mode .eq. 5) go to 100
      return

!     ==================================================================
!     mode = 4.    set  w = U(transpose)*v.
!     ==================================================================

      ! Find the last nonzero in  v(*).

  400 do klast = m, 1, -1
         i     = ip(klast)
         if (v(i) .ne. zero) go to 420
      end do

  420 klast  = min( klast, nrank )

      do j = 1, n
         w(j)  = zero
      end do

      do 480 k = 1, klast
         i     = ip(k)
         t     = v(i)
         if (t .eq. zero) go to 480
         l1    = locr(i)
         l2    = l1 + lenr(i) - 1

         do l = l1, l2
            j     = indr(l)
            w(j)  = w(j)  +  a(l) * t
         end do
  480 continue

      end ! subroutine lu6mul

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine lu6prt( m, n, v, w,
     $                   lena, luparm, parmlu,
     $                   a, indc, indr, ip, iq,
     $                   lenc, lenr, locc, locr )

      implicit           double precision (a-h,o-z)
      integer            luparm(30)
      double precision   parmlu(30), a(lena), v(m), w(n)
      integer            indc(lena), indr(lena), ip(m), iq(n)
      integer            lenc(n), lenr(m)
      integer            locc(n), locr(m)

!     ------------------------------------------------------------------
!     lu6prt  prints details of the current LU factorization, and
!     prints the matrix A = L*U row by row.  The amount of output
!     is controlled by lprint = luparm(2).
!
!     If  lprint = 0,  nothing is printed.
!     If  lprint = 1,  current LU statistics are printed.
!     If  lprint = 2,  the leading 10 by 10 submatrix is printed.
!     If  lprint = 3   or more, all rows are printed.
!
!     lenc(*), locc(*)  are not used.
!
!     09 May 1988: First F77 version.
!     03 Mar 2004: Current version.
!     ------------------------------------------------------------------

      parameter        ( zero = 0.0d+0,  one = 1.0d+0 )

      nout   = luparm(1)
      lprint = luparm(2)
      imax   = m
      jmax   = n
      if (lprint .le. 0) return
      if (lprint .le. 2) imax = min( imax, 10 )
      if (lprint .le. 2) jmax = min( jmax, 10 )
      write(nout, 1000) m, n, lena

!     --------------------------------
!     Print LU statistics.
!     --------------------------------
      lamin  = luparm(13)
      nrank  = luparm(16)
      lenl   = luparm(23)
      lenu   = luparm(24)
      lrow   = luparm(25)
      ncp    = luparm(26)
      mersum = luparm(27)
      amax   = parmlu(10)
      elmax  = parmlu(11)
      umax   = parmlu(12)
      dumin  = parmlu(14)

      avgmer = mersum
      floatm = m
      avgmer = avgmer / floatm
      growth = umax / (amax + 1.0d-20)
      write(nout, 2000) ncp, avgmer, lenl, lenu, nrank,
     $                  elmax, amax, umax, dumin, growth
      write(nout, 2100) (ip(i), i = 1, imax)
      write(nout, 2200) (iq(j), j = 1, jmax)
      if (lprint .le. 1) return

!     -------------------------------------------------------
!     lprint = 2 or more.    Print the first imax rows of  A.
!     -------------------------------------------------------
      do i = 1, imax
         do k = 1, m
            v(k) = zero
         end do
         v(i)   = one  ! v = i-th unit vector

         ! Set  w = A(t)*v = U(t)*L(t)*v.

         call lu6mul( 6, m, n, v, w,
     $                lena, luparm, parmlu,
     $                a, indc, indr, ip, iq, lenc, lenr, locc, locr )

         write(nout, 1100) (w(j), j = 1, jmax)
      end do
      return

 1000 format(/ ' -----------------------------------------------------'
     $       / ' lu6prt.      m =', i6, '     n =', i6, '   lena =', i8
     $       / ' -----------------------------------------------------')
 1100 format(/ (1x, 10g13.5))
 2000 format(/ ' -----------------------------------------------------'
     $       / ' LU factorization statistics.'
     $       / ' -----------------------------------------------------'
     $       //' compressns', i5, '    merit', 1p, e10.1,
     $         '    lenl', i11, '    lenu', i11, '    rank', i11
     $       / ' lmax', e11.1, '    bmax', e11.1,
     $         '    umax', e11.1, '    umin', e11.1,
     $         '    growth', e9.1)
 2100 format(//' Row    permutation  ip' / (1x, 10i7))
 2200 format(//' Column permutation  iq' / (1x, 10i7))

      end ! subroutine lu6prt

!     ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine lu6set( mode, m, n, w,
     $                   lena, luparm, parmlu,
     $                   a, indc, indr, ip, iq,
     $                   lenc, lenr, locc, locr,
     $                   inform )

      implicit           double precision (a-h,o-z)
      integer            luparm(30)
      double precision   parmlu(30), a(lena), w(n)
      integer            indc(lena), indr(lena), ip(m), iq(n)
      integer            lenc(n), lenr(m)
      integer            locc(n), locr(m)

!     ------------------------------------------------------------------
!     lu6set  initializes various quantities for the LU routines.
!
!     If mode = 1,
!     lu6set  gives default values to relevant elements of the
!     arrays  luparm(*)  and  parmlu(*).  This mode is appropriate
!     if a factorization routine (e.g. lu1fac) is called later.
!
!     If mode = 2,
!     lu6set  also sets the arrays a(*), indc(*), indr(*), etc.
!     to correspond to an LU factorization of the diagonal matrix
!
!                      A = Diag( w(i) )
!
!     for the given n-vector w(*).  The matrix A will have
!     m rows and n columns, where m and n must be at least 1.
!
!     w(*) may be completely zero, or contain some zero elements.
!     Any zero elements will be excluded from the data structure.
!     If m is less than n, only the first m entries of w(*)
!     will be used.
!
!     The LU factorization returned will be L = I, U = Diag(w),
!     with identities for the row and column permutations ip, iq.
!
!     On exit, inform = 0 unless there is some blatant error in
!     the input parameters.
!
!     09 May 1988: First F77 version.
!     03 Mar 2004: Current version.
!     ------------------------------------------------------------------

      parameter        ( zero = 0.0d+0 )

      luparm( 1) = 6
      luparm( 2) = 0
      luparm( 3) = 10

      parmlu( 1) = 10.0d+0
      parmlu( 2) = 10.0d+0
      parmlu( 3) = 1.0d-12
      parmlu( 4) = 1.0d-10
      parmlu( 5) = parmlu(4)
      parmlu( 6) = 3.0d+0
      inform     = 0
      nrank      = 0

!     Check for obvious errors.

      if (m    .le. 0    ) go to 910
      if (n    .le. 0    ) go to 910
      if (lena .le. m + n) go to 910
      if (mode .le. 1    ) go to 990

!     =================================================================
!     mode = 2.      Set  U = Diag( w(i) ) using nonzero elements w(i).
!     =================================================================
      nsing  = 0
      jsing  = 0
      dumax  = zero
      dumin  = abs( w(1) )
      l      = 0
      minmn  = min( m, n )

      do 100 i = 1, minmn
         if (w(i) .eq. zero) go to 100
         l       = l + 1
         ip(l)   = i
         iq(l)   = i
         lenr(i) = 1
         locr(i) = l
         a(l)    = w(i)
         indr(l) = i
         dumax   = max( dumax, abs( w(i) ) )
         dumin   = min( dumin, abs( w(i) ) )
  100 continue

      nrank  = l

      ! The permutations ip and iq point to the nonsingular part of w.
      ! Fix up the remaining parts of ip.

      do 200 i = 1, m
         if (  i  .gt.  n  ) go to 150
         if (w(i) .ne. zero) go to 200
  150    l       = l + 1
         ip(l)   = i
         lenr(i) = 0
         nsing   = nsing + 1
         jsing   = i
  200 continue

      ! Fix up the remaining parts of iq.

      l      = nrank
      do 300 j = 1, n
         locc(j) = 0
         if (w(j) .ne. zero) go to 300
         l       = l + 1
         iq(l)   = j
  300 continue

      !-----------------------------------
      ! Set output parameters for  mode 2.
      !-----------------------------------
      nupdat = 0
      numl0  = 0
      lenl   = 0
      lenu   = nrank
      lrow   = nrank
      ncp    = 0
      amax   = dumax
      elmax  = zero
      umax   = dumax

      luparm(11) = nsing
      luparm(12) = jsing
      luparm(15) = nupdat
      luparm(16) = nrank

      luparm(20) = numl0
      luparm(21) = lenl
      luparm(22) = lenu
      luparm(23) = lenl
      luparm(24) = lenu
      luparm(25) = lrow
      luparm(26) = ncp

      parmlu(10) = amax
      parmlu(11) = elmax
      parmlu(12) = umax
      parmlu(13) = dumax
      parmlu(14) = dumin
      go to 990

      ! Error exit.

  910 inform = 1

      ! Exit.

  990 luparm(10) = inform

      end ! subroutine lu6set
