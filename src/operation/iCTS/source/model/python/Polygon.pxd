# distutils: include_dirs = ../../solver/polygon

from libcpp.vector cimport vector

cdef extern from "pgl.h" namespace "icts":
    cdef cppclass Point:
        int x()
        int y()
    cdef cppclass Segment:
        Point low()
        Point high()
    cdef cppclass Polygon:
        vector[Point] get_points()