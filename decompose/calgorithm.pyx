#!/usr/bin/env python
# encoding: utf-8

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from cpython cimport bool

cdef np.float64_t NEGINF = -np.inf
cdef np.float64_t INF = np.inf
cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline np.float64_t float64_max(np.float64_t a, np.float64_t b): return a if a >= b else b
cdef inline np.float64_t float64_min(np.float64_t a, np.float64_t b): return a if a <= b else b


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.npy_intp, ndim=1] parse_proj(np.ndarray[np.float64_t, ndim=2] scores):
    cdef int nr, nc, N, i, k, s, t, r, maxidx
    cdef np.float64_t tmp, cand
    cdef np.ndarray[np.float64_t, ndim=2] complete_0
    cdef np.ndarray[np.float64_t, ndim=2] complete_1
    cdef np.ndarray[np.float64_t, ndim=2] incomplete_0
    cdef np.ndarray[np.float64_t, ndim=2] incomplete_1
    cdef np.ndarray[np.npy_intp, ndim=3] complete_backtrack
    cdef np.ndarray[np.npy_intp, ndim=3] incomplete_backtrack
    cdef np.ndarray[np.npy_intp, ndim=1] heads

    nr, nc = np.shape(scores)

    N = nr - 1 # Number of words (excluding root).

    complete_0 = np.zeros((nr, nr)) # s, t, direction (right=1).
    complete_1 = np.zeros((nr, nr)) # s, t, direction (right=1).
    incomplete_0 = np.zeros((nr, nr)) # s, t, direction (right=1).
    incomplete_1 = np.zeros((nr, nr)) # s, t, direction (right=1).

    complete_backtrack = -np.ones((nr, nr, 2), dtype=int) # s, t, direction (right=1).
    incomplete_backtrack = -np.ones((nr, nr, 2), dtype=int) # s, t, direction (right=1).

    for i in range(nr):
        incomplete_0[i, 0] = NEGINF

    for k in range(1, nr):
        for s in range(nr - k):
            t = s + k
            tmp = NEGINF
            maxidx = s
            for r in range(s, t):
                cand = complete_1[s, r] + complete_0[r+1, t]
                if cand > tmp:
                    tmp = cand
                    maxidx = r
                if s == 0 and r == 0:
                    break
            incomplete_0[t, s] = tmp + scores[t, s]
            incomplete_1[s, t] = tmp + scores[s, t]
            incomplete_backtrack[s, t, 0] = maxidx
            incomplete_backtrack[s, t, 1] = maxidx

            tmp = NEGINF
            maxidx = s
            for r in range(s, t):
                cand = complete_0[s, r] + incomplete_0[t, r]
                if cand > tmp:
                    tmp = cand
                    maxidx = r
            complete_0[s, t] = tmp
            complete_backtrack[s, t, 0] = maxidx

            tmp = NEGINF
            maxidx = s + 1
            for r in range(s+1, t+1):
                cand = incomplete_1[s, r] + complete_1[r, t]
                if cand > tmp:
                    tmp = cand
                    maxidx = r
            complete_1[s, t] = tmp
            complete_backtrack[s, t, 1] = maxidx

    heads = -np.ones(N + 1, dtype=int)
    backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)

    return heads


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void backtrack_eisner(np.ndarray[np.npy_intp, ndim=3] incomplete_backtrack,
        np.ndarray[np.npy_intp, ndim=3]complete_backtrack,
        int s, int t, int direction, int complete, np.ndarray[np.npy_intp, ndim=1] heads):
    cdef int r
    if s == t:
        return
    if complete:
        r = complete_backtrack[s, t, direction]
        if direction:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
            return
        else:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
            return
    else:
        r = incomplete_backtrack[s, t, direction]
        if direction:
            heads[t] = s
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, 1, heads)
            return
        else:
            heads[s] = t
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, 1, heads)
            return

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bool is_projective(np.ndarray[np.npy_intp, ndim=1] heads):
    cdef int n_len, i, j, cur
    cdef int edge1_0, edge1_1, edge2_0, edge2_1
    n_len = heads.shape[0]
    for i in range(n_len):
        if heads[i] < 0:
            continue
        for j in range(i + 1, n_len):
            if heads[j] < 0:
                continue
            edge1_0 = int_min(i, heads[i])
            edge1_1 = int_max(i, heads[i])
            edge2_0 = int_min(j, heads[j])
            edge2_1 = int_max(j, heads[j])
            if edge1_0 == edge2_0:
                if edge1_1 == edge2_1:
                    return False
                else:
                    continue
            if edge1_0 < edge2_0 and not (edge2_0 >= edge1_1 or edge2_1 <= edge1_1):
                return False
            if edge1_0 > edge2_0 and not (edge1_0 >= edge2_1 or edge1_1 <= edge2_1):
                return False
    return True

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.npy_intp, ndim=1] projectivize(np.ndarray[np.npy_intp, ndim=1] heads):
    if is_projective(heads):
        return heads

    cdef int n_len, h, m
    cdef np.ndarray[np.float64_t, ndim=2] scores

    n_len = heads.shape[0]
    scores = np.zeros((n_len, n_len))
    for m in range(1, n_len):
        h = heads[m]
        scores[h, m] = 1.
    return parse_proj(scores)
