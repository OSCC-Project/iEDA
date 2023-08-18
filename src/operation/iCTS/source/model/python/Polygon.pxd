# distutils: include_dirs = ../../solver/polygon

from libcpp.vector cimport vector

cdef extern from "pgl.h" namespace "icts":
    cdef cppclass CtsPoint[double]:
        double x()
        double y()
    cdef cppclass CtsSegment[double]:
        CtsPoint[double] low()
        CtsPoint[double] high()
    cdef cppclass CtsPolygon[double]:
        vector[CtsPoint[double]] get_points()