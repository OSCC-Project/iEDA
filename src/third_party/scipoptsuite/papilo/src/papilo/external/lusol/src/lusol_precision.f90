!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! File lusol_recision.f90
!
! SNOPT module for defining integer(ip), real(rp).
! snPrecision.f90 should be one of the following 3 files:
!
! snPrecision32.f90   sets  ip=4, rp = 8
! snPrecision64.f90   sets  ip=8, rp = 8
! snPrecision128.f90  sets  ip=8, rp = 16
!
! ip  huge
!  4  2147483647
!  8  9223372036854775807
!
! rp  huge
!  4  3.40282347E+38
!  8  1.79769313486231571E+308
! 16  1.18973149535723176508575932662800702E+4932
!
! rp  eps
!  4  1.19209290E-07
!  8  2.22044604925031308E-016
! 16  1.92592994438723585305597794258492732E-0034
!
! We don't need selected_int_kind or selected_real_kind now.
! Previously we used these values:
! ip = integer precision    int_kind( 7) = integer(4)
!                           int_kind(15) = integer(8)
! rp = real precision      real_kind( 6) = real(4)   (not used in SNOPT)
!                          real_kind(15) = real(8)
!                          real_kind(30) = real(16)
!
! 11 Mar 2008: First version.
! 20 Apr 2012: First quad version.
! 22 Apr 2012: Made three versions of snPrecision.f90.
!              See README.QUAD for use with configure and make.
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

module lusol_precision
  use  iso_fortran_env
  implicit none
  public

  integer(4),   parameter :: ip = int64, rp = real64

end module lusol_precision
