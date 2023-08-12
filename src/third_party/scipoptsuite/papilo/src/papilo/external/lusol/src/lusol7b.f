*************************************************************************
!
!     File  lusol7b.f
!
!     lu7asv   lu7bak
!
!     These routines are used by the update routines in lusol8b.f.
!
! 08 Jun 2004: lusol7b.f is essentially lu7b.for from VMS days.
!              integer*4 changed to integer  .
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine lu7asv( m, n, iv, lenw, lw1, lw2, marker, wmult,
     $                   lena, luparm, parmlu,
     $                   lenu, lrow,
     $                   a, indc, indr, lenr, locc, locr )

      implicit           double precision (a-h,o-z)
      integer            luparm(30)
      double precision   parmlu(30), a(lena)
      integer            indc(lena), indr(lena), lenr(m)
      integer            locc(n), locr(m)

!     ------------------------------------------------------------------
!     lu7asv (Add Sparse Vectors) sets  v  =  v  +  wmult * w,
!     where  v  and  w  are sparse vectors and  wmult  is a scalar.
!     v  is row  iv  of an  m by n  matrix  U  stored in  a(*), indr(*).
!     w  is stored in locations  lw1  thru  lw2  of  a(*), indr(*).
!
!     Assumptions...
!
!  1. There are at least  n  free locations of storage in  a(*), indr(*)
!     beyond the current point  lrow.  This means that  v  can be moved
!     to the end of storage if need be, and there is room for fill-in.
!
!  2. The nonzeros of  w  are marked by  locc(*)  in the usual way.
!
!  3. The array  indc(*)  is parallel to  indr(*),  and has been
!     initialized in the locations corresponding to  w  to some value
!     other than the input parameter  marker.
!
!     -- Feb 1985: Last  F66 version.
!     10 May 1988: First F77 version.
!     ------------------------------------------------------------------

      logical            atend

      if (lenw .eq. 0) return
      small  = parmlu(3)
      lenv   = lenr(iv)
      if (lenv .eq. 0) locr(iv) = lrow + 1
      lv1    = locr(iv)
      lv2    = lv1 + lenv - 1
      last   = lv2
      lenu   = lenu - lenv
      atend  = last .eq. lrow

!     ------------------------------------------------------------------
!     First, modify existing elements  v(j)  if  w(j)  is nonzero,
!     and flag those elements of  w  by setting  indc(*) = marker.
!     To facilitate removal of computed zeros, we work backwards
!     through the elements of  v  and replace negligible new elements
!     by the current last element of  v  (which will always be in the
!     spot pointed to by the integer  last).
!     ------------------------------------------------------------------
      if (lenv .eq. 0) go to 420
      nw     = 0

      do lv = lv2, lv1, -1
         jv     = indr(lv)
         lw     = locc(jv)
         if (lw .gt. 0) then
            nw       = nw + 1
            indc(lw) = marker
            a(lv)    = a(lv)  +  wmult * a(lw)
            if (abs( a(lv) ) .le. small) then

               ! Delete small element by moving the last element of  v.

               a(lv)      = a(last)
               indr(lv)   = indr(last)
               indr(last) = 0
               last       = last - 1
            end if
         end if
      end do

!     ------------------------------------------------------------------
!     If all elements of  w  have just been accessed, we are done.
!     Otherwise, see if there is enough room at the end of  v
!     to accommodate the fill-in.
!     ------------------------------------------------------------------
      if (nw .eq. lenw) go to 500
      if (   atend    ) go to 420
      l1     = last + 1
      l2     = last + (lenw - nw)

      do l = l1, l2
         if (indr(l) .gt. 0) go to 400
      end do
      go to 420

!     ------------------------------------------------------------------
!     We must move  v  to the end of storage.
!     ------------------------------------------------------------------
  400 atend    = .true.
      l1       = lv1
      l2       = last
      lenv     = last - lv1 + 1
      lv1      = lrow + 1
      last     = lrow + lenv
      locr(iv) = lv1

      do l = l1, l2
         lrow       = lrow + 1
         a(lrow)    = a(l)
         indr(lrow) = indr(l)
         indr(l)    = 0
      end do

!     ------------------------------------------------------------------
!     Now generate the fill-ins in  v
!     using the elements of  w  that were previously skipped.
!     ------------------------------------------------------------------
  420 do lw = lw1, lw2
         if (indc(lw) .ne. marker) then
            last       = last + 1
            a(last)    = wmult * a(lw)
            indr(last) = indr(lw)
         end if
      end do

!     ------------------------------------------------------------------
!     Reset the length of  v  and the total nonzeros in  U.
!     ------------------------------------------------------------------
  500 lenv       = last  - lv1 + 1
      lenr(iv)   = lenv
      lenu       = lenu + lenv
      if (atend) lrow = last

      end ! subroutine lu7asv

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine lu7bak( m, n, kfirst, klast, c,
     $                   lena, luparm, parmlu,
     $                   lenl, lenu, lrow,
     $                   a, indc, indr, ip, iq, lenr, locc, locr,
     $                   inform )

      implicit           double precision (a-h,o-z)
      integer            luparm(30)
      double precision   parmlu(30), a(lena), c(m)
      integer            indc(lena), indr(lena), ip(m), iq(n), lenr(m)
      integer            locc(n), locr(m)

!     ------------------------------------------------------------------
!     lu7bak  (Backward Sweep) updates the LU factorization  A = L*U
!     when part of the column vector  c  is reduced to a multiple of
!     a unit vector by a backward sweep of stabilized row operations
!     of the form
!
!        LSWEEP * c  =  c(iw) * e(iw).
!
!     Elements  ip(kfirst+1),  ip(kfirst+2),  ...,  ip(klast)  of  c
!     are involved.  It is assumed that  (kfirst + 1) < klast.
!     The matrices  L  and  U  are updated to become
!
!        L(new)  =  L * LSWEEP(inverse)    and    U(new)  =  LSWEEP * U
!
!     where  ip * U(new) * iq  is upper triangular except for the
!     row  iw = ip(klast).  The row permutation  ip  is updated
!     accordingly.  The column permutation  iq  is not altered.
!
!     Row  iw  is held in compact form like the other rows of  U.
!
!     On entry, all elements of  locc  are assumed to be zero.
!     On a successful exit (inform = 0) they will again be zero.
!
!     -- Feb 1985: First version.
!     10 May 1988: First F77 version.
!     ------------------------------------------------------------------

      double precision   lmax
      logical            cmprss, first, swap

      lmax   = parmlu(2)
      small  = parmlu(3)
      lfree  = lena - lenl
      iw     = ip(klast)
      cw     = c(iw)
      lenw   = lenr(iw)

!     ==================================================================
!     Eliminate the nonzeros of  c  backwards.
!     ==================================================================
      first  = .true.

      do 500 k  = klast-1, kfirst+1, -1
         iv     = ip(k)
         cv     = c(iv)
         if (abs( cv ) .le. small) go to 500

         ! cv  has to be eliminated.

         cmprss = .false.
         swap   = .false.

!        ---------------------------------------------------------------
!        Compress storage if necessary, so there will be room if
!        v  has to be moved to the end.  (If we waited until  v  is
!        actually moved, the markers on  w  would have to be reset.)
!        ---------------------------------------------------------------
         minfre = n + 1
         nfree  = lfree - lrow
         if (nfree .ge. minfre) go to 200
         call lu1rec( m, .true., luparm, lrow, lena, a,indr,lenr,locr )
         cmprss = .true.
         nfree  = lfree - lrow
         if (nfree .lt. minfre) go to 970

!        ---------------------------------------------------------------
!        We are about to set  v  =  v  +  ( - cv/cw ) * w.
!        See if rows  v  and  w  should be interchanged first.
!        ---------------------------------------------------------------
  200    if (lmax * abs( cv )  .lt.  abs( cw )) go to 210
         if (lmax * abs( cw )  .lt.  abs( cv )) go to 220
         if (lenr(iv)          .lt.  lenw     ) go to 220

  210    if (first             .or.  cmprss   ) go to 250
         go to 300

!        ---------------------------------------------------------------
!        Interchange rows  iv  and  iw.
!        ---------------------------------------------------------------
  220    swap      = .true.
         i         = iv
         iv        = iw
         iw        = i
         w         = cv
         cv        = cw
         cw        = w
         ip(k)     = iv
         ip(klast) = iw
         if (first  .or.  lenw .eq. 0) go to 250

         ! Cancel the markers on the old row  iw.

         do 230 l = lw1, lw2
            jw       = indr(l)
            locc(jw) = 0
  230    continue

!        ---------------------------------------------------------------
!        Set markers on the new row  iw.
!        This happens if  first,  cmprss,  or  swap.
!        ---------------------------------------------------------------
  250    lenw   = lenr(iw)
         if (lenw .eq. 0) go to 300
         lw1    = locr(iw)
         lw2    = lw1 + lenw - 1

         do 260 lw = lw1, lw2
            jw       = indr(lw)
            locc(jw) = lw
            indc(lw) = 0
  260    continue

!        ---------------------------------------------------------------
!        Form the multiplier and store it in the L file.
!        ---------------------------------------------------------------
  300    first       = .false.
         wmult       = - cv / cw
         a(lfree)    = wmult
         indr(lfree) = iw
         indc(lfree) = iv
         lenl        = lenl  + 1
         lfree       = lfree - 1

!        ---------------------------------------------------------------
!        Set  v  =  v  +  wmult * w.
!        ---------------------------------------------------------------
         if (lenw .gt. 0) then
            call lu7asv( m, n, iv, lenw, lw1, lw2, k, wmult,
     $                   lena, luparm, parmlu,
     $                   lenu, lrow,
     $                   a, indc, indr, lenr, locc, locr )
         end if

!        ---------------------------------------------------------------
!        If there was a swap, make sure the diagonal of  v  is first.
!        ---------------------------------------------------------------
         if (.not. swap ) go to 500
         lenv   = lenr(iv)
         if (lenv .eq. 0) go to 500
         if ( k   .gt. n) go to 500
         jdiag  = iq(k)
         lv1    = locr(iv)
         lv2    = lv1 + lenv - 1

         do l = lv1, lv2
            if (indr(l) .eq. jdiag) go to 470
         end do
         go to 500

  470    indr(l)   = indr(lv1)
         indr(lv1) = jdiag
         diag      = a(l)
         a(l)      = a(lv1)
         a(lv1)    = diag
  500 continue

!     ==================================================================
!     End of main elimination loop.
!     ==================================================================

      ! Cancel the markers on  w,  unless they were never set.

      if (first  .or.  lenw .eq. 0) go to 900
      do lw = lw1, lw2
         jw       = indr(lw)
         locc(jw) = 0
      end do

      ! Successful update.

  900 inform = 0
      go to 990

      ! Not enough storage.

  970 inform = 7

      ! Exit.

  990 return

      end ! subroutine lu7bak
