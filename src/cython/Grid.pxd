#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

cdef class Cut:
    cdef:
        Py_ssize_t k

cdef class Dual:
    cdef:
        Py_ssize_t k

cdef class DualCut:
    cdef:
        Py_ssize_t k

cdef class Mid:
    cdef:
        Py_ssize_t k

cdef class ZLocation:
    pass

cdef class Zmin(ZLocation):
    pass

cdef class Zmax(ZLocation):
    pass

cdef class DataLocation:
    pass

cdef class Center(DataLocation):
    pass

cdef class Node(DataLocation):
    pass

cdef class Grid:
    cdef:
        double dz
        double dzi
        Py_ssize_t gw
        Py_ssize_t nz
        Py_ssize_t nzg
        double [:] z_half
        double [:] z

        cdef Py_ssize_t n_hat(self, ZLocation b)
        cdef Py_ssize_t binary(self, ZLocation b)
        cdef Py_ssize_t first_interior(self, ZLocation b)
        cdef Py_ssize_t boundary(self, ZLocation b)
        cdef int[:] slice_real(self, DataLocation loc)
        cdef int[:] slice_all(self, DataLocation loc)
        cdef int[:] slice_ghost(self, DataLocation loc, ZLocation b)
        cdef Py_ssize_t[:] over_elems_ghost(self, DataLocation loc, ZLocation b)
        cdef Py_ssize_t[:] over_elems(self, DataLocation loc)
        cdef Py_ssize_t[:] over_elems_real(self, DataLocation loc)
