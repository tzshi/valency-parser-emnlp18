# distutils: language = c++
# encoding: utf-8

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from cpython cimport bool
from libcpp.queue cimport priority_queue
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.set cimport set
from libcpp.unordered_map cimport unordered_map as umap
from libc.stdlib cimport malloc, free
from cython.operator cimport dereference as deref, preincrement as inc
import sys

cdef np.float64_t NEGINF = -np.inf
cdef np.float64_t INF = np.inf
cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline np.float64_t float64_max(np.float64_t a, np.float64_t b): return a if a >= b else b
cdef inline np.float64_t float64_min(np.float64_t a, np.float64_t b): return a if a <= b else b
ctypedef vector[int] *int_vec_p
ctypedef unsigned long long ull

cdef int LEFT = 0
cdef int RIGHT = 1
cdef int TRAP = 0
cdef int TRI = 1
cdef int FLTRI = 2
cdef int FRTRI = 3
cdef int TAIL = 0
cdef int HEAD = 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.npy_intp, ndim=1] chart_hsel(np.ndarray[np.float64_t, ndim=2] probs):
    cdef int nr, nc, N, i, k, s, t, r, maxidx, total
    cdef np.float64_t tmp, cand

    cdef np.ndarray[np.float64_t, ndim=2] triangle
    cdef np.ndarray[np.float64_t, ndim=2] trapezoid
    cdef np.ndarray[np.npy_intp, ndim=2] triangle_backtrack
    cdef np.ndarray[np.npy_intp, ndim=2] trapezoid_backtrack
    cdef np.ndarray[np.npy_intp, ndim=1] heads

    total = 0

    nr, nc = np.shape(probs)
    N = nr - 1

    triangle = np.full((nr, nr), 0., dtype=np.float)
    trapezoid = np.full((nr, nr), 0., dtype=np.float)

    triangle_backtrack = -np.ones((nr, nr), dtype=int)
    trapezoid_backtrack = -np.ones((nr, nr), dtype=int)

    for i in range(nr):
        trapezoid[i, 0] = NEGINF

    for k in range(1, nr):
        for s in range(nr - k):
            t = s + k
            tmp = NEGINF
            maxidx = s

            for r in range(s, t):
                cand = triangle[s, r] + triangle[t, r+1]
                if cand > tmp:
                    tmp = cand
                    maxidx = r
                if s == 0 and r == 0:
                    break
            trapezoid[t, s] = tmp + probs[t, s]
            trapezoid_backtrack[t, s] = maxidx
            trapezoid[s, t] = tmp + probs[s, t]
            trapezoid_backtrack[s, t] = maxidx

            tmp = NEGINF
            maxidx = s
            for r in range(s, t):
                cand = triangle[r, s] + trapezoid[t, r]
                if cand > tmp:
                    tmp = cand
                    maxidx = r
            triangle[t, s] = tmp
            triangle_backtrack[t, s] = maxidx

            tmp = NEGINF
            maxidx = s + 1
            for r in range(s+1, t+1):
                cand = trapezoid[s, r] + triangle[r, t]
                if cand > tmp:
                    tmp = cand
                    maxidx = r
            triangle[s, t] = tmp
            triangle_backtrack[s, t] = maxidx

    heads = -np.ones(nr, dtype=int)
    backtrack_hsel(trapezoid_backtrack, triangle_backtrack, 0, N, True, True, heads)

    return heads


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void push_queue(priority_queue[pair[float, pair[float, vector[int]]]]& queue, float inside_score, float outside_score, int s, int t, int is_triangle, int backtrack):
    cdef vector[int] vec
    cdef pair[float, vector[int]] item
    cdef pair[float, pair[float, vector[int]]] p
    vec.push_back(s)
    vec.push_back(t)
    vec.push_back(is_triangle)
    vec.push_back(backtrack)
    p.first = inside_score + outside_score
    item.first = inside_score
    item.second = vec
    p.second = item
    queue.push(p)


@cython.boundscheck(False)
@cython.wraparound(False)
def astar_hsel(np.ndarray[np.float64_t, ndim=2] probs):
    cdef int nr, nc, N, i, j, k, s, t, r, maxidx, total
    cdef int is_triangle
    cdef int start, end
    cdef np.float64_t tmp, cand

    cdef np.ndarray[np.float64_t, ndim=2] heuristics
    cdef np.ndarray[np.float64_t, ndim=2] triangle
    cdef np.ndarray[np.float64_t, ndim=2] trapezoid
    cdef np.ndarray[np.float64_t, ndim=1] maximum
    cdef np.ndarray[np.npy_intp, ndim=2] triangle_backtrack
    cdef np.ndarray[np.npy_intp, ndim=2] trapezoid_backtrack
    cdef np.ndarray[np.npy_intp, ndim=1] heads

    cdef priority_queue[pair[float, pair[float, vector[int]]]] queue
    cdef pair[float, pair[float, vector[int]]] p
    cdef pair[float, vector[int]] item
    cdef vector[int] vec


    total = 0

    nr, nc = np.shape(probs)
    N = nr - 1

    triangle = np.full((nr, nr), NEGINF, dtype=np.float)
    trapezoid = np.full((nr, nr), NEGINF, dtype=np.float)

    triangle_backtrack = -np.ones((nr, nr), dtype=int)
    trapezoid_backtrack = -np.ones((nr, nr), dtype=int)

    # disable self attachment
    for i in range(nr):
        probs[i, i] = NEGINF

    # pre-compute the heuristics, O(n^2)
    maximum = np.amax(probs, axis=0)
    heuristics = np.zeros((nr, nr), dtype=np.float)
    for i in range(nr):
        for j in range(1, i):
            for k in range(nr):
                heuristics[i, k] += maximum[j]
    for i in range(nr):
        for j in range(i + 1, nr):
            for k in range(nr):
                heuristics[k, i] += maximum[j]

    for i in range(nr):
        push_queue(queue, 0., heuristics[i, i] + maximum[i], i, i, 1, -1)

    while triangle[0, N] == NEGINF and queue.size() > 0:
        p = queue.top()
        item = p.second
        vec = item.second
        s, t, is_triangle, maxidx = vec[0], vec[1], vec[2], vec[3]
        queue.pop()

        if is_triangle:
            if triangle[s, t] != NEGINF:
                continue
            else:
                triangle[s, t] = item.first
                triangle_backtrack[s, t] = maxidx

            if s >= t:
                for r in range(t):
                    if triangle[r, t - 1] == NEGINF:
                        continue

                    push_queue(queue, triangle[r, t - 1] + triangle[s, t] + probs[r, s],
                            heuristics[r, s] + maximum[r], r, s, 0, t - 1)
                    total += 1

                    push_queue(queue, triangle[r, t - 1] + triangle[s, t] + probs[s, r],
                            heuristics[r, s] + maximum[s], s, r, 0, t - 1)
                    total += 1

                for r in range(s + 1, nr):
                    if trapezoid[r, s] == NEGINF:
                        continue

                    push_queue(queue, triangle[s, t] + trapezoid[r, s],
                            heuristics[t, r] + maximum[r], r, t, 1, s)
                    total += 1

            if s <= t:
                for r in range(t + 1, nr):
                    if triangle[r, t + 1] == NEGINF:
                        continue

                    push_queue(queue, triangle[r, t + 1] + triangle[s, t] + probs[r, s],
                            heuristics[s, r] + maximum[r], r, s, 0, t)
                    total += 1

                    push_queue(queue, triangle[r, t + 1] + triangle[s, t] + probs[s, r],
                            heuristics[s, r] + maximum[s], s, r, 0, t)
                    total += 1

                for r in range(s):
                    if trapezoid[r, s] == NEGINF:
                        continue

                    if r == 0 and t != N:
                        continue

                    push_queue(queue, triangle[s, t] + trapezoid[r, s],
                            heuristics[r, t] + maximum[r], r, t, 1, s)
                    total += 1

        else:
            if trapezoid[s, t] != NEGINF:
                continue
            else:
                trapezoid[s, t] = item.first
                trapezoid_backtrack[s, t] = maxidx

            if s > t:
                for r in range(t + 1):
                    if triangle[t, r] == NEGINF:
                        continue

                    push_queue(queue, triangle[t, r] + trapezoid[s, t],
                            heuristics[r, s] + maximum[s], s, r, 1, t)
                    total += 1

            elif s < t:
                for r in range(t, nr):
                    if triangle[t, r] == NEGINF:
                        continue

                    if s == 0 and r != N:
                        continue

                    push_queue(queue, triangle[t, r] + trapezoid[s, t],
                            heuristics[s, r] + maximum[s], s, r, 1, t)
                    total += 1

    heads = -np.ones(nr, dtype=int)
    backtrack_hsel(trapezoid_backtrack, triangle_backtrack, 0, N, True, True, heads)

    return total, heads


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void backtrack_hsel(np.ndarray[np.npy_intp, ndim=2] trapezoid_backtrack,
        np.ndarray[np.npy_intp, ndim=2] triangle_backtrack,
        int s, int t, int direction, int triangle, np.ndarray[np.npy_intp, ndim=1] heads):
    cdef int r
    if s == t:
        return
    if triangle:
        r = triangle_backtrack[s, t]
        backtrack_hsel(trapezoid_backtrack, triangle_backtrack, s, r, direction, False, heads)
        backtrack_hsel(trapezoid_backtrack, triangle_backtrack, r, t, direction, True, heads)
        return
    else:
        heads[t] = s
        r = trapezoid_backtrack[s, t]
        backtrack_hsel(trapezoid_backtrack, triangle_backtrack, int_min(s, t), r, True, True, heads)
        backtrack_hsel(trapezoid_backtrack, triangle_backtrack, int_max(s, t), r + 1, False, True, heads)
        return


@cython.boundscheck(False)
@cython.wraparound(False)
def chart_hsel_supertag(
        np.ndarray[np.float64_t, ndim=2] hsel, np.ndarray[np.float64_t, ndim=2] supertag, np.ndarray[np.float64_t, ndim=3] label,
        np.ndarray[np.npy_intp, ndim=2] supertag_mapping, np.ndarray[np.npy_intp, ndim=2] reverse_mapping,
        np.ndarray[np.npy_intp, ndim=3] cat2rel, int noncore_cat):
    cdef int nr, nc, nt, npt, _, N, i, k, s, t, r, g, g1, g2, new_g, maxidx, maxg, total
    cdef np.float64_t tmp, cand

    cdef np.ndarray[np.float64_t, ndim=3] partial_triangle
    cdef np.ndarray[np.float64_t, ndim=3] full_ltriangle
    cdef np.ndarray[np.float64_t, ndim=3] full_rtriangle
    cdef np.ndarray[np.float64_t, ndim=4] trapezoid
    cdef np.ndarray[np.npy_intp, ndim=4] triangle_backtrack
    cdef np.ndarray[np.npy_intp, ndim=5] trapezoid_backtrack
    cdef np.ndarray[np.npy_intp, ndim=1] heads
    cdef np.ndarray[np.npy_intp, ndim=1] rels
    supertag[0] = 0.

    total = 0

    nr, nc = np.shape(hsel)
    N = nr - 1

    _, nt = np.shape(supertag)
    npt, _ = np.shape(supertag_mapping)

    full_ltriangle = np.full((nr, nr, nt), NEGINF, dtype=np.float)
    full_rtriangle = np.full((nr, nr, nt), NEGINF, dtype=np.float)
    partial_triangle = np.full((nr, nr, npt), NEGINF, dtype=np.float)
    trapezoid = np.full((nr, nr, npt, npt), NEGINF, dtype=np.float)

    triangle_backtrack = -np.ones((nr, nr, npt, 2), dtype=int)
    trapezoid_backtrack = -np.ones((nr, nr, npt, nt, 3), dtype=int)

    for i in range(nr):
        for t in range(nt):
            partial_triangle[i, i, t] = supertag[i, t] / 2
            if supertag_mapping[t, 0] < 0:
                full_ltriangle[i, i, t] = supertag[i, t] / 2
            if supertag_mapping[t, 1] < 0:
                full_rtriangle[i, i, t] = supertag[i, t] / 2
        for t in range(nt, npt):
            partial_triangle[i, i, t] = NEGINF

    for k in range(1, nr):
        for s in range(nr - k):
            t = s + k

            for g1 in range(npt):
                for g2 in range(nt):
                    tmp = NEGINF
                    maxidx = s

                    for r in range(s, t):
                        cand = full_rtriangle[s, r, g2] + partial_triangle[t, r+1, g1]
                        if cand > tmp:
                            tmp = cand
                            maxidx = r
                        if s == 0 and r == 0:
                            break

                    new_g = supertag_mapping[g1, 0]
                    if new_g >= 0:
                        cand = tmp + hsel[t, s] + label[t, s, supertag_mapping[g1, 2]]
                        if cand > trapezoid[t, s, new_g, g2]:
                            trapezoid[t, s, new_g, g2] = cand
                            trapezoid_backtrack[t, s, new_g, g2, 0] = maxidx
                            trapezoid_backtrack[t, s, new_g, g2, 1] = g1
                            trapezoid_backtrack[t, s, new_g, g2, 2] = supertag_mapping[g1, 2]
                    cand = tmp + hsel[t, s] + label[t, s, noncore_cat]
                    if cand > trapezoid[t, s, g1, g2]:
                        trapezoid[t, s, g1, g2] = cand
                        trapezoid_backtrack[t, s, g1, g2, 0] = maxidx
                        trapezoid_backtrack[t, s, g1, g2, 1] = g1
                        trapezoid_backtrack[t, s, g1, g2, 2] = noncore_cat

                    tmp = NEGINF
                    maxidx = s

                    for r in range(s, t):
                        cand = partial_triangle[s, r, g1] + full_ltriangle[t, r+1, g2]
                        if cand > tmp:
                            tmp = cand
                            maxidx = r
                        if s == 0 and r == 0:
                            break

                    new_g = supertag_mapping[g1, 1]
                    if new_g >= 0:
                        cand = tmp + hsel[s, t] + label[s, t, supertag_mapping[g1, 3]]
                        if cand > trapezoid[s, t, new_g, g2]:
                            trapezoid[s, t, new_g, g2] = cand
                            trapezoid_backtrack[s, t, new_g, g2, 0] = maxidx
                            trapezoid_backtrack[s, t, new_g, g2, 1] = g1
                            trapezoid_backtrack[s, t, new_g, g2, 2] = supertag_mapping[g1, 3]
                    cand = tmp + hsel[s, t] + label[s, t, noncore_cat]
                    if cand > trapezoid[s, t, g1, g2]:
                        trapezoid[s, t, g1, g2] = cand
                        trapezoid_backtrack[s, t, g1, g2, 0] = maxidx
                        trapezoid_backtrack[s, t, g1, g2, 1] = g1
                        trapezoid_backtrack[s, t, g1, g2, 2] = noncore_cat

            for g1 in range(npt):
                tmp = NEGINF
                maxidx = s
                maxg = -1
                for g2 in range(nt):
                    for r in range(s, t):
                        cand = full_ltriangle[r, s, g2] + trapezoid[t, r, g1, g2]
                        if cand > tmp:
                            tmp = cand
                            maxidx = r
                            maxg = g2
                partial_triangle[t, s, g1] = tmp
                triangle_backtrack[t, s, g1, 0] = maxidx
                triangle_backtrack[t, s, g1, 1] = maxg

                if supertag_mapping[g1, 0] < 0:
                    g = -supertag_mapping[g1, 0] - 1
                    if tmp > full_ltriangle[t, s, g]:
                        full_ltriangle[t, s, g] = tmp

                tmp = NEGINF
                maxidx = s + 1
                maxg = -1
                for g2 in range(nt):
                    for r in range(s+1, t+1):
                        cand = trapezoid[s, r, g1, g2] + full_rtriangle[r, t, g2]
                        if cand > tmp:
                            tmp = cand
                            maxidx = r
                            maxg = g2
                partial_triangle[s, t, g1] = tmp
                triangle_backtrack[s, t, g1, 0] = maxidx
                triangle_backtrack[s, t, g1, 1] = maxg

                if supertag_mapping[g1, 1] < 0:
                    g = -supertag_mapping[g1, 1] - 1
                    if tmp > full_rtriangle[s, t, g]:
                        full_rtriangle[s, t, g] = tmp

    # print(partial_triangle[0, N, 0], full_rtriangle[0, N, 0])
    # print(partial_triangle[0, N, 1], full_rtriangle[0, N, 1])
    heads = -np.ones(nr, dtype=int)
    rels = -np.ones(nr, dtype=int)
    backtrack_hsel_supertag(trapezoid_backtrack, triangle_backtrack, reverse_mapping, 0, N, 1, 0, True, True, heads, rels)

    for i in range(1, nr):
        rels[i] = cat2rel[heads[i], i, rels[i]]

    return heads, rels


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void backtrack_hsel_supertag(np.ndarray[np.npy_intp, ndim=5] trapezoid_backtrack,
        np.ndarray[np.npy_intp, ndim=4] triangle_backtrack,
        np.ndarray[np.npy_intp, ndim=2] reverse_mapping,
        int s, int t, int gt, int g, int direction, int triangle,
        np.ndarray[np.npy_intp, ndim=1] heads, np.ndarray[np.npy_intp, ndim=1] rels):
    cdef int r, m
    if s == t:
        return
    if triangle:
        r = triangle_backtrack[s, t, gt, 0]
        m = triangle_backtrack[s, t, gt, 1]
        backtrack_hsel_supertag(trapezoid_backtrack, triangle_backtrack, reverse_mapping, s, r, gt, m, direction, False, heads, rels)
        if s < t:
            backtrack_hsel_supertag(trapezoid_backtrack, triangle_backtrack, reverse_mapping, r, t, reverse_mapping[m, 1], 0, direction, True, heads, rels)
        else:
            backtrack_hsel_supertag(trapezoid_backtrack, triangle_backtrack, reverse_mapping, r, t, reverse_mapping[m, 0], 0, direction, True, heads, rels)
        return
    else:
        heads[t] = s
        r = trapezoid_backtrack[s, t, gt, g, 0]
        m = trapezoid_backtrack[s, t, gt, g, 1]
        rels[t] = trapezoid_backtrack[s, t, gt, g, 2]
        if s < t:
            backtrack_hsel_supertag(trapezoid_backtrack, triangle_backtrack, reverse_mapping, s, r, m, 0, True, True, heads, rels)
            backtrack_hsel_supertag(trapezoid_backtrack, triangle_backtrack, reverse_mapping, t, r + 1, reverse_mapping[g, 0], 0, False, True, heads, rels)
        else:
            backtrack_hsel_supertag(trapezoid_backtrack, triangle_backtrack, reverse_mapping, t, r, reverse_mapping[g, 1], 0, True, True, heads, rels)
            backtrack_hsel_supertag(trapezoid_backtrack, triangle_backtrack, reverse_mapping, s, r + 1, m, 0, False, True, heads, rels)
        return


@cython.boundscheck(False)
@cython.wraparound(False)
def best_label_per_cat(np.ndarray[np.float64_t, ndim=3] label_score, np.ndarray[np.npy_intp, ndim=1] label2cat, int num_cats, int noncore_cat):
    cdef int nr, nc, nl, i, j, k, c
    cdef np.ndarray[np.float64_t, ndim=3] best_per_cat
    cdef np.ndarray[np.npy_intp, ndim=3] best_labels

    nr, nc, nl = np.shape(label_score)

    best_per_cat = np.full((nr, nc, num_cats), NEGINF, dtype=np.float)
    best_labels = np.full((nr, nc, num_cats), -1, dtype=int)

    for i in range(nr):
        for j in range(nc):
            for k in range(nl):
                c = label2cat[k]
                if label_score[i, j, k] > best_per_cat[i, j, c]:
                    best_per_cat[i, j, c] = label_score[i, j, k]
                    best_labels[i, j, c] = k
                if label_score[i, j, k] > best_per_cat[i, j, 0]:
                    best_per_cat[i, j, 0] = label_score[i, j, k]
                    best_labels[i, j, 0] = k

    # for i in range(nr):
        # for j in range(nc):
            # for k in range(num_cats):
                # best_per_cat[i, j, k] = 0.

    return best_per_cat, best_labels

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void push_queue_new(priority_queue[pair[float, pair[float, vector[int]]]]& queue, float inside_score, float outside_score, int s, int t, int is_triangle, int comp1, int comp2):
    cdef vector[int] vec
    cdef pair[float, vector[int]] item
    cdef pair[float, pair[float, vector[int]]] p
    vec.push_back(s)
    vec.push_back(t)
    vec.push_back(is_triangle)
    vec.push_back(comp1)
    vec.push_back(comp2)
    p.first = inside_score + outside_score
    item.first = inside_score
    item.second = vec
    p.second = item
    queue.push(p)

cdef inline ull get_idx(ull direction, ull triangle, ull head, ull idx, ull nr):
    return direction * 4 * nr + triangle * 2 * nr + head * nr + idx

cdef inline ull get_set_id(ull triangle, ull head, ull tail, ull nr):
    return triangle * nr * nr + head * nr + tail

@cython.boundscheck(False)
@cython.wraparound(False)
def astar_hsel_new(np.ndarray[np.float64_t, ndim=2] probs):
    cdef int nr, nc, N, i, j, k, s, t, r, u, total, _
    cdef ull idx
    cdef int is_triangle
    cdef int start, end
    cdef int data_i, data_cur
    cdef np.float64_t tmp, cand
    cdef float score, score2

    cdef np.ndarray[np.float64_t, ndim=2] heuristics
    cdef np.ndarray[np.float64_t, ndim=1] maximum
    cdef np.ndarray[np.npy_intp, ndim=1] heads

    cdef priority_queue[pair[float, pair[float, vector[int]]]] queue
    cdef pair[float, pair[float, vector[int]]] p
    cdef pair[float, vector[int]] item
    cdef vector[int] vec
    cdef int_vec_p v_p
    cdef vector[int].iterator v_iter
    cdef set[int] structure_set

    data_cur = -1
    total = 0

    nr, nc = np.shape(probs)
    N = nr - 1

    cdef int total_p = 2 * 2 * 2 * nr
    # cdef vector[int] **pointers = <vector[int] **>malloc(total_p * sizeof(int_vec_p))
    # for i in range(total_p):
        # pointers[i] = new vector[int]()
    cdef umap[ull, vector[int]] pointers
    cdef vector[vector[int]] data
    cdef vector[float] inside_score

    # disable self attachment
    for i in range(nr):
        probs[i, i] = NEGINF

    # pre-compute the heuristics, O(n^2)
    maximum = np.amax(probs, axis=0)
    heuristics = np.zeros((nr, nr), dtype=np.float)
    for i in range(nr):
        for j in range(1, i):
            for k in range(nr):
                heuristics[i, k] += maximum[j]
    for i in range(nr):
        for j in range(i + 1, nr):
            for k in range(nr):
                heuristics[k, i] += maximum[j]

    for i in range(nr):
        push_queue_new(queue, 0., heuristics[i, i] + maximum[i], i, i, TRI, -1, -1)

    while queue.size() > 0:
        p = queue.top()
        item = p.second
        score = item.first
        vec = item.second
        s, t, is_triangle, comp1, comp2 = vec[0], vec[1], vec[2], vec[3], vec[4]
        queue.pop()

        idx = get_set_id(is_triangle, s, t, nr)
        if structure_set.count(idx):
            continue

        structure_set.insert(idx)
        data.push_back(vec)
        inside_score.push_back(score)
        data_cur += 1

        if is_triangle:
            if s >= t:
                idx = get_idx(LEFT, TRI, HEAD, s, nr)
                pointers[idx].push_back(data_cur)
                idx = get_idx(LEFT, TRI, TAIL, t, nr)
                pointers[idx].push_back(data_cur)
            if s <= t:
                idx = get_idx(RIGHT, TRI, HEAD, s, nr)
                pointers[idx].push_back(data_cur)
                idx = get_idx(RIGHT, TRI, TAIL, t, nr)
                pointers[idx].push_back(data_cur)

            if s == 0 and t == N:
                break

            if s >= t:
                if t - 1 >= 0:
                    idx = get_idx(RIGHT, TRI, TAIL, t - 1, nr)
                    if pointers.count(idx):
                        v_p = &pointers[idx]
                        v_iter = v_p.begin()
                        while v_iter != v_p.end():
                            u = deref(v_iter)
                            score2 = inside_score[u]
                            r = data[u][0]
                            push_queue_new(queue, score + score2 + probs[r, s], heuristics[r, s] + maximum[r], r, s, TRAP, u, data_cur)
                            push_queue_new(queue, score + score2 + probs[s, r], heuristics[r, s] + maximum[s], s, r, TRAP, data_cur, u)
                            total += 2
                            inc(v_iter)

                idx = get_idx(LEFT, TRAP, TAIL, s, nr)
                if pointers.count(idx):
                    v_p = &pointers[idx]
                    v_iter = v_p.begin()
                    while v_iter != v_p.end():
                        u = deref(v_iter)
                        score2 = inside_score[u]
                        r = data[u][0]
                        push_queue_new(queue, score + score2, heuristics[t, r] + maximum[r], r, t, TRI, u, data_cur)
                        total += 1
                        inc(v_iter)

            if s <= t:
                if t + 1 < nr:
                    idx = get_idx(LEFT, TRI, TAIL, t + 1, nr)
                    if pointers.count(idx):
                        v_p = &pointers[idx]
                        v_iter = v_p.begin()
                        while v_iter != v_p.end():
                            u = deref(v_iter)
                            score2 = inside_score[u]
                            r = data[u][0]
                            push_queue_new(queue, score + score2 + probs[r, s], heuristics[s, r] + maximum[r], r, s, TRAP, u, data_cur)
                            push_queue_new(queue, score + score2 + probs[s, r], heuristics[s, r] + maximum[s], s, r, TRAP, data_cur, u)
                            total += 2
                            inc(v_iter)

                idx = get_idx(RIGHT, TRAP, TAIL, s, nr)
                if pointers.count(idx):
                    v_p = &pointers[idx]
                    v_iter = v_p.begin()
                    while v_iter != v_p.end():
                        u = deref(v_iter)
                        score2 = inside_score[u]
                        r = data[u][0]
                        if r == 0 and t != N:
                            inc(v_iter)
                            continue
                        push_queue_new(queue, score + score2, heuristics[r, t] + maximum[r], r, t, TRI, u, data_cur)
                        total += 1
                        inc(v_iter)

        else:
            # idx = get_idx(RIGHT if s < t else LEFT, TRAP, HEAD, s, nr)
            # pointers[idx].push_back(data_cur)
            idx = get_idx(RIGHT if s < t else LEFT, TRAP, TAIL, t, nr)
            pointers[idx].push_back(data_cur)

            if s > t:
                idx = get_idx(LEFT, TRI, HEAD, t, nr)
                if pointers.count(idx):
                    v_p = &pointers[idx]
                    v_iter = v_p.begin()
                    while v_iter != v_p.end():
                        u = deref(v_iter)
                        score2 = inside_score[u]
                        r = data[u][1]
                        push_queue_new(queue, score + score2, heuristics[r, s] + maximum[s], s, r, TRI, data_cur, u)
                        total += 1
                        inc(v_iter)
            elif s < t:
                idx = get_idx(RIGHT, TRI, HEAD, t, nr)
                if pointers.count(idx):
                    v_p = &pointers[idx]
                    v_iter = v_p.begin()
                    while v_iter != v_p.end():
                        u = deref(v_iter)
                        score2 = inside_score[u]
                        r = data[u][1]
                        if s == 0 and r != N:
                            inc(v_iter)
                            continue
                        push_queue_new(queue, score + score2, heuristics[s, r] + maximum[s], s, r, TRI, data_cur, u)
                        total += 1
                        inc(v_iter)

    heads = -np.ones(nr, dtype=int)
    backtrack_hsel_new(data, data_cur, heads)

    # for i in range(total_p):
        # del pointers[i]
    # free(pointers)

    return total, heads


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void backtrack_hsel_new(vector[vector[int]] data, int data_cur, np.ndarray[np.npy_intp, ndim=1] heads):
    cdef int s, t, is_triangle, comp1, comp2
    if data_cur < 0:
        return
    s, t, is_triangle, comp1, comp2 = data[data_cur][0], data[data_cur][1], data[data_cur][2], data[data_cur][3], data[data_cur][4]
    if is_triangle == TRAP:
        heads[t] = s
    backtrack_hsel_new(data, comp1, heads)
    backtrack_hsel_new(data, comp2, heads)

cdef inline ull get_idx_supertag(ull direction, ull triangle, ull head, ull idx, ull tag, ull nr):
    return (((tag * nr + idx) * 2 + head) * 2 + direction) * 4 + triangle

cdef inline ull get_set_id_supertag(ull triangle, ull head, ull tail, ull tag1, ull tag2, ull nr, ull nt):
    return (((tag1 * nt + tag2) * nr + head) * nr + tail) * 4 + triangle

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void push_queue_supertag_new(priority_queue[pair[float, pair[float, vector[int]]]]& queue, float inside_score, float outside_score, int s, int t, int triangle, int tag1, int tag2, int tt1, int comp1, int comp2, int label=-1):
    cdef vector[int] vec
    cdef pair[float, vector[int]] item
    cdef pair[float, pair[float, vector[int]]] p
    vec.push_back(s)
    vec.push_back(t)
    vec.push_back(triangle)
    vec.push_back(tag1)
    vec.push_back(tag2)
    vec.push_back(tt1)
    vec.push_back(comp1)
    vec.push_back(comp2)
    vec.push_back(label)
    p.first = inside_score + outside_score
    item.first = inside_score
    item.second = vec
    p.second = item
    # print("push", p.first, item.first, item.second)
    queue.push(p)


@cython.boundscheck(False)
@cython.wraparound(False)
def astar_hsel_supertag_new(
        np.ndarray[np.float64_t, ndim=2] hsel, np.ndarray[np.float64_t, ndim=2] supertag,
        np.ndarray[np.float64_t, ndim=3] label, np.ndarray[np.npy_intp, ndim=2] supertag_mapping,
        np.ndarray[np.npy_intp, ndim=2] reverse_mapping, np.ndarray[np.npy_intp, ndim=3] cat2rel,
        int noncore_cat):
    cdef int nr, nc, nt, npt, N, i, j, k, s, t, r, u, tag1, tag2, tt1, tt2, t1, t2, m, total, _
    cdef ull idx
    cdef int triangle
    cdef int start, end
    cdef int data_i, data_cur
    cdef np.float64_t tmp, cand
    cdef float score, score2

    cdef np.ndarray[np.float64_t, ndim=2] heuristics
    cdef np.ndarray[np.float64_t, ndim=1] maximum_supertag
    cdef np.ndarray[np.float64_t, ndim=1] maximum_attachment
    cdef np.ndarray[np.npy_intp, ndim=1] heads
    cdef np.ndarray[np.npy_intp, ndim=1] rels

    cdef priority_queue[pair[float, pair[float, vector[int]]]] queue
    cdef pair[float, pair[float, vector[int]]] p
    cdef pair[float, vector[int]] item
    cdef vector[int] vec
    cdef int_vec_p v_p
    cdef vector[int].iterator v_iter
    cdef set[int] structure_set

    supertag[0] = 0.

    # print(supertag)

    data_cur = -1
    total = 0

    nr, nc = np.shape(hsel)
    N = nr - 1

    _, nt = np.shape(supertag)
    npt, _ = np.shape(supertag_mapping)

    cdef umap[ull, vector[int]] pointers
    cdef vector[vector[int]] data
    cdef vector[float] inside_score

    # disable self attachment
    for i in range(nr):
        hsel[i, i] = NEGINF

    # pre-compute the heuristics, O(n^2)
    maximum_attachment = np.amax(hsel + np.amax(label, axis=2), axis=0)
    maximum_attachment[0] = 0.
    maximum_supertag = np.amax(supertag, axis=1)
    maximum_supertag[0] = 0.
    heuristics = np.zeros((nr, nr), dtype=np.float)
    for i in range(nr):
        for j in range(1, i):
            for k in range(nr):
                heuristics[i, k] += maximum_attachment[j] + maximum_supertag[j]
    for i in range(nr):
        for j in range(i + 1, nr):
            for k in range(nr):
                heuristics[k, i] += maximum_attachment[j] + maximum_supertag[j]

    for i in range(1, nr):
        for t in range(nt):
            push_queue_supertag_new(queue, supertag[i, t] / 2, heuristics[i, i] + maximum_attachment[i] + supertag[i, t] / 2, i, i, TRI, t, 0, t, -1, -1)
    for t in range(nt):
        push_queue_supertag_new(queue, supertag[0, t] / 2, heuristics[0, 0] + supertag[i, t] / 2, 0, 0, TRI, t, 0, t, -1, -1)

    while queue.size() > 0:
        p = queue.top()
        item = p.second
        score = item.first
        vec = item.second
        s, t, triangle, tag1, tag2, tt1, comp1, comp2 = vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7]
        idx = get_set_id_supertag(triangle, s, t, tag1, tag2, nr, nt)
        # print(p.first, score, vec, idx)
        queue.pop()

        idx = get_set_id_supertag(triangle, s, t, tag1, tag2, nr, nt)
        if structure_set.count(idx):
            continue

        structure_set.insert(idx)
        data.push_back(vec)
        inside_score.push_back(score)
        data_cur += 1
        # print(data_cur)

        if triangle == FLTRI:
            idx = get_idx_supertag(LEFT, FLTRI, HEAD, s, tag1, nr)
            pointers[idx].push_back(data_cur)
            idx = get_idx_supertag(LEFT, FLTRI, TAIL, t, 0, nr)
            pointers[idx].push_back(data_cur)

            if t - 1 >= 0:
                idx = get_idx_supertag(RIGHT, TRI, TAIL, t - 1, 0, nr)
                if pointers.count(idx):
                    v_p = &pointers[idx]
                    v_iter = v_p.begin()
                    while v_iter != v_p.end():
                        u = deref(v_iter)
                        score2 = inside_score[u]
                        r = data[u][0]
                        t1 = data[u][3]
                        tt2 = data[u][5]
                        m = supertag_mapping[t1, RIGHT]
                        if m >= 0:
                            push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, supertag_mapping[t1, 2 + RIGHT]], heuristics[r, s] + maximum_attachment[r] + supertag[r, tt2] / 2 + supertag[s, tt1] / 2, r, s, TRAP, m, tag1, tt2, u, data_cur, supertag_mapping[t1, 2 + RIGHT])
                            total += 1
                        elif t1 == 0:
                            push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, 0], heuristics[r, s] + maximum_attachment[r] + supertag[r, tt2] / 2 + supertag[s, tt1] / 2, r, s, TRAP, t1, tag1, tt2, u, data_cur, 0)
                            total += 1
                        push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, noncore_cat], heuristics[r, s] + maximum_attachment[r] + supertag[r, tt2] / 2 + supertag[s, tt1] / 2, r, s, TRAP, t1, tag1, tt2, u, data_cur, noncore_cat)
                        total += 1
                        inc(v_iter)

            idx = get_idx_supertag(LEFT, TRAP, TAIL, s, tag1, nr)
            if pointers.count(idx):
                v_p = &pointers[idx]
                v_iter = v_p.begin()
                while v_iter != v_p.end():
                    u = deref(v_iter)
                    score2 = inside_score[u]
                    r = data[u][0]
                    t1 = data[u][3]
                    tt2 = data[u][5]
                    push_queue_supertag_new(queue, score + score2, heuristics[t, r] + maximum_attachment[r] + supertag[r, tt2] / 2, r, t, TRI, t1, 0, tt2, u, data_cur)
                    total += 1
                    inc(v_iter)

        elif triangle == FRTRI:
            idx = get_idx_supertag(RIGHT, FRTRI, HEAD, s, tag1, nr)
            pointers[idx].push_back(data_cur)
            idx = get_idx_supertag(RIGHT, FRTRI, TAIL, t, 0, nr)
            pointers[idx].push_back(data_cur)

            if t + 1 < nr:
                idx = get_idx_supertag(LEFT, TRI, TAIL, t + 1, 0, nr)
                if pointers.count(idx):
                    v_p = &pointers[idx]
                    v_iter = v_p.begin()
                    while v_iter != v_p.end():
                        u = deref(v_iter)
                        score2 = inside_score[u]
                        r = data[u][0]
                        t1 = data[u][3]
                        tt2 = data[u][5]
                        m = supertag_mapping[t1, LEFT]
                        if m >= 0:
                            push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, supertag_mapping[t1, 2 + LEFT]], heuristics[s, r] + maximum_attachment[r] + supertag[s, tt1] / 2 + supertag[r, tt2] / 2, r, s, TRAP, m, tag1, tt2, u, data_cur, supertag_mapping[t1, 2 + LEFT])
                            total += 1
                        elif t1 == 0:
                            push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, 0], heuristics[s, r] + maximum_attachment[r] + supertag[s, tt1] / 2 + supertag[r, tt2] / 2, r, s, TRAP, t1, tag1, tt2, u, data_cur, 0)
                            total += 1
                        push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, noncore_cat], heuristics[s, r] + maximum_attachment[r] + supertag[s, tt1] / 2 + supertag[r, tt2] / 2, r, s, TRAP, t1, tag1, tt2, u, data_cur, noncore_cat)
                        total += 1
                        inc(v_iter)

            idx = get_idx_supertag(RIGHT, TRAP, TAIL, s, tag1, nr)
            if pointers.count(idx):
                v_p = &pointers[idx]
                v_iter = v_p.begin()
                while v_iter != v_p.end():
                    u = deref(v_iter)
                    score2 = inside_score[u]
                    r = data[u][0]
                    t1 = data[u][3]
                    tt2 = data[u][5]
                    if r == 0 and t != N:
                        inc(v_iter)
                        continue
                    push_queue_supertag_new(queue, score + score2, heuristics[r, t] + maximum_attachment[r] + supertag[r, tt2] / 2, r, t, TRI, t1, 0, tt2, u, data_cur)
                    total += 1
                    inc(v_iter)

        elif triangle == TRI:
            if s >= t:
                idx = get_idx_supertag(LEFT, TRI, HEAD, s, tag1, nr)
                pointers[idx].push_back(data_cur)
                idx = get_idx_supertag(LEFT, TRI, TAIL, t, 0, nr)
                pointers[idx].push_back(data_cur)
                m = supertag_mapping[tag1, LEFT]
                if m < 0:
                    push_queue_supertag_new(queue, score, heuristics[t, s] + maximum_attachment[s] + supertag[s, tt1] / 2, s, t, FLTRI, -m - 1, 0, tt1, data_cur, -1)
            if s <= t:
                idx = get_idx_supertag(RIGHT, TRI, HEAD, s, tag1, nr)
                pointers[idx].push_back(data_cur)
                idx = get_idx_supertag(RIGHT, TRI, TAIL, t, 0, nr)
                pointers[idx].push_back(data_cur)
                m = supertag_mapping[tag1, RIGHT]
                if m < 0:
                    push_queue_supertag_new(queue, score, heuristics[s, t] + maximum_attachment[s] + supertag[s, tt1] / 2, s, t, FRTRI, -m - 1, 0, tt1, data_cur, -1)

            if s == 0 and t == N:
                break

            if s >= t:
                if t - 1 >= 0:
                    idx = get_idx_supertag(RIGHT, FRTRI, TAIL, t - 1, 0, nr)
                    if pointers.count(idx):
                        v_p = &pointers[idx]
                        v_iter = v_p.begin()
                        m = supertag_mapping[tag1, LEFT]
                        while v_iter != v_p.end():
                            u = deref(v_iter)
                            score2 = inside_score[u]
                            r = data[u][0]
                            t1 = data[u][3]
                            tt2 = data[u][5]
                            if m >= 0:
                                push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, supertag_mapping[tag1, 2 + LEFT]], heuristics[r, s] + maximum_attachment[s] + supertag[s, tt1] / 2 + supertag[r, tt2] / 2, s, r, TRAP, m, t1, tt1, data_cur, u, supertag_mapping[tag1, 2 + LEFT])
                                total += 1
                            elif tag1 == 0:
                                push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, 0], heuristics[r, s] + maximum_attachment[s] + supertag[s, tt1] / 2 + supertag[r, tt2] / 2, s, r, TRAP, tag1, t1, tt1, data_cur, u, 0)
                                total += 1
                            push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, noncore_cat], heuristics[r, s] + maximum_attachment[s] + supertag[s, tt1] / 2 + supertag[r, tt2] / 2, s, r, TRAP, tag1, t1, tt1, data_cur, u, noncore_cat)
                            total += 1
                            inc(v_iter)

            if s <= t:
                if t + 1 < nr:
                    idx = get_idx_supertag(LEFT, FLTRI, TAIL, t + 1, 0, nr)
                    if pointers.count(idx):
                        v_p = &pointers[idx]
                        v_iter = v_p.begin()
                        m = supertag_mapping[tag1, RIGHT]
                        while v_iter != v_p.end():
                            u = deref(v_iter)
                            score2 = inside_score[u]
                            r = data[u][0]
                            t1 = data[u][3]
                            tt2 = data[u][5]
                            if m >= 0:
                                push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, supertag_mapping[tag1, 2 + RIGHT]], heuristics[s, r] + maximum_attachment[s] + supertag[s, tt1] / 2 + supertag[r, tt2] / 2, s, r, TRAP, m, t1, tt1, data_cur, u, supertag_mapping[tag1, 2 + RIGHT])
                                total += 1
                            elif tag1 == 0:
                                push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, 0], heuristics[s, r] + maximum_attachment[s] + supertag[s, tt1] / 2 + supertag[r, tt2] / 2, s, r, TRAP, tag1, t1, tt1, data_cur, u, 0)
                                total += 1
                            push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, noncore_cat], heuristics[s, r] + maximum_attachment[s] + supertag[s, tt1] / 2 + supertag[r, tt2] / 2, s, r, TRAP, tag1, t1, tt1, data_cur, u, noncore_cat)
                            total += 1
                            inc(v_iter)

        elif triangle == TRAP:
            idx = get_idx_supertag(RIGHT if s < t else LEFT, TRAP, TAIL, t, tag2, nr)
            pointers[idx].push_back(data_cur)

            if s > t:
                idx = get_idx_supertag(LEFT, FLTRI, HEAD, t, tag2, nr)
                if pointers.count(idx):
                    v_p = &pointers[idx]
                    v_iter = v_p.begin()
                    while v_iter != v_p.end():
                        u = deref(v_iter)
                        score2 = inside_score[u]
                        r = data[u][1]
                        push_queue_supertag_new(queue, score + score2, heuristics[r, s] + maximum_attachment[s] + supertag[s, tt1] / 2, s, r, TRI, tag1, 0, tt1, data_cur, u)
                        total += 1
                        inc(v_iter)
            elif s < t:
                idx = get_idx_supertag(RIGHT, FRTRI, HEAD, t, tag2, nr)
                if pointers.count(idx):
                    v_p = &pointers[idx]
                    v_iter = v_p.begin()
                    while v_iter != v_p.end():
                        u = deref(v_iter)
                        score2 = inside_score[u]
                        r = data[u][1]
                        if s == 0 and r != N:
                            inc(v_iter)
                            continue
                        push_queue_supertag_new(queue, score + score2, heuristics[s, r] + maximum_attachment[s] + supertag[s, tt1] / 2, s, r, TRI, tag1, 0, tt1, data_cur, u)
                        total += 1
                        inc(v_iter)

    # print(heuristics[0, 0])
    # print(score)
    heads = -np.ones(nr, dtype=int)
    rels = -np.ones(nr, dtype=int)
    backtrack_hsel_supertag_new(data, data_cur, heads, rels)

    for i in range(1, nr):
        rels[i] = cat2rel[heads[i], i, rels[i]]

    return total, heads, rels


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void backtrack_hsel_supertag_new(
        vector[vector[int]] data, int data_cur,
        np.ndarray[np.npy_intp, ndim=1] heads, np.ndarray[np.npy_intp, ndim=1] rels):
    cdef int s, t, triangle, comp1, comp2, tag1, tag2, tt1, label
    if data_cur < 0:
        return
    s, t, triangle, tag1, tag2, tt1, comp1, comp2, label = data[data_cur][0], data[data_cur][1], data[data_cur][2], data[data_cur][3], data[data_cur][4], data[data_cur][5], data[data_cur][6], data[data_cur][7], data[data_cur][8]
    if triangle == TRAP:
        heads[t] = s
        rels[t] = label
    backtrack_hsel_supertag_new(data, comp1, heads, rels)
    backtrack_hsel_supertag_new(data, comp2, heads, rels)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int layer_id_t(int t, int[::1] layer_index, int num_layers, int layer_id):
    cdef int i, cur
    cur = 1
    for i in range(layer_id + 1, num_layers):
        cur *= layer_index[i + 1] - layer_index[i]
    t = t / cur
    return t % (layer_index[layer_id + 1] - layer_index[layer_id]) + layer_index[layer_id]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int layer_id_pt(int t, int npt, int num_layers, int layer_id):
    cdef int i, cur
    cur = 1
    for i in range(layer_id + 1, num_layers):
        cur *= npt
    t = t / cur
    return t % npt

cdef inline int t2pt(int t, int[::1] layer_index, int npt, int num_layers):
    cdef int i, cur, ret
    ret = 0
    for i in range(num_layers):
        ret *= npt
        ret += layer_id_t(t, layer_index, num_layers, i)
    # print(t, ret)
    return ret

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double supertag_score_t(int n, int t, double[:, ::1] supertag, int[::1] layer_index, int num_layers):
    cdef int i
    cdef double ret
    ret = 0.
    for i in range(num_layers):
        ret += supertag[n, layer_id_t(t, layer_index, num_layers, i)]
    return ret / 2

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int next_cat_layer(int pt, int npt, int num_layers, int[:,::1] supertag_mapping, int direction, int layer):
    return supertag_mapping[layer_id_pt(pt, npt, num_layers, layer), 2 + direction]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int advance_pt(int pt, int npt, int num_layers, int cat, int[:,::1] supertag_mapping, int direction):
    cdef int i, cur, ret, tmp
    ret = 0
    for i in range(num_layers):
        ret *= npt
        cur = layer_id_pt(pt, npt, num_layers, i)
        tmp = supertag_mapping[cur, 2 + direction]
        if tmp == cat and cat > 0:
            ret += supertag_mapping[cur, direction]
        else:
            ret += cur
    # print(pt, cat, ret)
    return ret

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int raise_to_full(int pt, int npt, int[::1] layer_index, int num_layers, int[:,::1] supertag_mapping, int direction):
    cdef int i, cur, ret, tmp
    ret = 0
    for i in range(num_layers):
        ret *= layer_index[i + 1] - layer_index[i]
        cur = layer_id_pt(pt, npt, num_layers, i)
        tmp = supertag_mapping[cur, direction]
        if tmp >= 0:
            return -1
        else:
            ret += -tmp - 1 - layer_index[i]
    return ret

@cython.boundscheck(False)
@cython.wraparound(False)
def astar_hsel_supertag_multi(
        np.ndarray[np.float64_t, ndim=2] hsel, double[:,::1] supertag,
        np.ndarray[np.float64_t, ndim=3] label, int[:,::1] supertag_mapping,
        np.ndarray[np.npy_intp, ndim=3] cat2rel, int[::1] layer_index,
        int noncore_cat, int max_queue=-1):
    cdef int nr, nc, nt, npt, n_layers, N, i, j, k, s, t, r, u, tag1, tag2, tt1, tt2, t1, t2, m, total, _
    cdef int next_cat
    cdef ull idx
    cdef int triangle
    cdef int start, end
    cdef int data_i, data_cur
    cdef np.float64_t tmp, cand
    cdef float score, score2

    cdef np.ndarray[np.float64_t, ndim=2] heuristics
    cdef np.ndarray[np.float64_t, ndim=1] maximum_supertag
    cdef np.ndarray[np.float64_t, ndim=1] maximum_attachment
    cdef np.ndarray[np.npy_intp, ndim=1] heads
    cdef np.ndarray[np.npy_intp, ndim=1] rels

    cdef priority_queue[pair[float, pair[float, vector[int]]]] queue
    cdef pair[float, pair[float, vector[int]]] p
    cdef pair[float, vector[int]] item
    cdef vector[int] vec
    cdef int_vec_p v_p
    cdef vector[int].iterator v_iter
    cdef set[int] structure_set

    data_cur = -1
    total = 0
    nr, nc = np.shape(hsel)
    N = nr - 1

    npt, _ = np.shape(supertag_mapping)

    n_layers = len(layer_index) - 1
    # n_layers = 1

    nt = 1
    for i in range(n_layers):
        nt *= layer_index[i + 1] - layer_index[i]

    supertag[0,:] = 0.

    cdef umap[ull, vector[int]] pointers
    cdef vector[vector[int]] data
    cdef vector[float] inside_score

    # disable self attachment
    for i in range(nr):
        hsel[i, i] = NEGINF

    # pre-compute the heuristics, O(n^2)
    maximum_attachment = np.amax(hsel + np.amax(label, axis=2), axis=0)
    maximum_attachment[0] = 0.
    maximum_supertag = np.zeros((nr,), dtype=np.float)
    for i in range(n_layers):
        maximum_supertag += np.amax(supertag[:, layer_index[i]:layer_index[i+1]], axis=1)
    maximum_supertag[0] = 0.
    heuristics = np.zeros((nr, nr), dtype=np.float)
    for i in range(nr):
        for j in range(1, i):
            for k in range(i, nr):
                heuristics[i, k] += maximum_attachment[j] + maximum_supertag[j]
    for i in range(nr):
        for j in range(i + 1, nr):
            for k in range(0, i + 1):
                heuristics[k, i] += maximum_attachment[j] + maximum_supertag[j]

    for i in range(1, nr):
        for t in range(1, nt):
            push_queue_supertag_new(queue, supertag_score_t(i, t, supertag, layer_index, n_layers), heuristics[i, i] + maximum_attachment[i] + supertag_score_t(i, t, supertag, layer_index, n_layers), i, i, TRI, t2pt(t, layer_index, npt, n_layers), 0, t, -1, -1)
    # for t in range(nt):
    for t in range(1):
        push_queue_supertag_new(queue, supertag_score_t(0, t, supertag, layer_index, n_layers), heuristics[0, 0] + supertag_score_t(0, t, supertag, layer_index, n_layers), 0, 0, TRI, t2pt(t, layer_index, npt, n_layers), 0, t, -1, -1)

    while queue.size() > 0:
        if queue.size() > max_queue:
            return -1, None, None
        p = queue.top()
        item = p.second
        score = item.first
        vec = item.second
        s, t, triangle, tag1, tag2, tt1, comp1, comp2 = vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7]
        # idx = get_set_id_supertag(triangle, s, t, tag1, tag2, nr, nt)
        # print(p.first, score, vec)
        queue.pop()
        # print(s, t, triangle, tag1, tag2, tt1, comp1, comp2)

        idx = get_set_id_supertag(triangle, s, t, tag1, tag2, nr, nt)
        if structure_set.count(idx):
            continue

        structure_set.insert(idx)
        data.push_back(vec)
        inside_score.push_back(score)
        data_cur += 1
        # print(data_cur)

        if triangle == FLTRI:
            idx = get_idx_supertag(LEFT, FLTRI, HEAD, s, tag1, nr)
            pointers[idx].push_back(data_cur)
            idx = get_idx_supertag(LEFT, FLTRI, TAIL, t, 0, nr)
            pointers[idx].push_back(data_cur)

            if t - 1 >= 0:
                idx = get_idx_supertag(RIGHT, TRI, TAIL, t - 1, 0, nr)
                if pointers.count(idx):
                    v_p = &pointers[idx]
                    v_iter = v_p.begin()
                    while v_iter != v_p.end():
                        u = deref(v_iter)
                        score2 = inside_score[u]
                        r = data[u][0]
                        t1 = data[u][3]
                        tt2 = data[u][5]
                        # m = supertag_mapping[t1, RIGHT]
                        # if m >= 0:
                            # push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, supertag_mapping[t1, 2 + RIGHT]], heuristics[r, s] + maximum_attachment[r] + supertag_score_t(r, tt2, supertag, layer_index, n_layers) + supertag_score_t(s, tt1, supertag, layer_index, n_layers), r, s, TRAP, m, tag1, tt2, u, data_cur, supertag_mapping[t1, 2 + RIGHT])
                            # total += 1
                        # elif t1 == 0:
                            # push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, 0], heuristics[r, s] + maximum_attachment[r] + supertag_score_t(r, tt2, supertag, layer_index, n_layers) + supertag_score_t(s, tt1, supertag, layer_index, n_layers), r, s, TRAP, t1, tag1, tt2, u, data_cur, 0)
                            # total += 1
                        for i in range(n_layers):
                            next_cat =  next_cat_layer(t1, npt, n_layers, supertag_mapping, RIGHT, i)
                            if next_cat < 0:
                                continue
                            m = advance_pt(t1, npt, n_layers, next_cat, supertag_mapping, RIGHT)
                            # print("1", m)
                            push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, next_cat], heuristics[r, s] + maximum_attachment[r] + supertag_score_t(r, tt2, supertag, layer_index, n_layers) + supertag_score_t(s, tt1, supertag, layer_index, n_layers), r, s, TRAP, m, tag1, tt2, u, data_cur, next_cat)
                            total += 1
                        push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, noncore_cat], heuristics[r, s] + maximum_attachment[r] + supertag_score_t(r, tt2, supertag, layer_index, n_layers) + supertag_score_t(s, tt1, supertag, layer_index, n_layers), r, s, TRAP, t1, tag1, tt2, u, data_cur, noncore_cat)
                        total += 1
                        inc(v_iter)

            idx = get_idx_supertag(LEFT, TRAP, TAIL, s, tag1, nr)
            if pointers.count(idx):
                v_p = &pointers[idx]
                v_iter = v_p.begin()
                while v_iter != v_p.end():
                    u = deref(v_iter)
                    score2 = inside_score[u]
                    r = data[u][0]
                    t1 = data[u][3]
                    tt2 = data[u][5]
                    push_queue_supertag_new(queue, score + score2, heuristics[t, r] + maximum_attachment[r] + supertag_score_t(r, tt2, supertag, layer_index, n_layers), r, t, TRI, t1, 0, tt2, u, data_cur)
                    total += 1
                    inc(v_iter)

        elif triangle == FRTRI:
            idx = get_idx_supertag(RIGHT, FRTRI, HEAD, s, tag1, nr)
            pointers[idx].push_back(data_cur)
            idx = get_idx_supertag(RIGHT, FRTRI, TAIL, t, 0, nr)
            pointers[idx].push_back(data_cur)

            if t + 1 < nr:
                idx = get_idx_supertag(LEFT, TRI, TAIL, t + 1, 0, nr)
                if pointers.count(idx):
                    v_p = &pointers[idx]
                    v_iter = v_p.begin()
                    while v_iter != v_p.end():
                        u = deref(v_iter)
                        score2 = inside_score[u]
                        r = data[u][0]
                        t1 = data[u][3]
                        tt2 = data[u][5]
                        # m = supertag_mapping[t1, LEFT]
                        # if m >= 0:
                            # push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, supertag_mapping[t1, 2 + LEFT]], heuristics[s, r] + maximum_attachment[r] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), r, s, TRAP, m, tag1, tt2, u, data_cur, supertag_mapping[t1, 2 + LEFT])
                            # total += 1
                        # elif t1 == 0:
                            # push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, 0], heuristics[s, r] + maximum_attachment[r] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), r, s, TRAP, t1, tag1, tt2, u, data_cur, 0)
                            # total += 1
                        for i in range(n_layers):
                            next_cat =  next_cat_layer(t1, npt, n_layers, supertag_mapping, LEFT, i)
                            if next_cat < 0:
                                continue
                            m = advance_pt(t1, npt, n_layers, next_cat, supertag_mapping, LEFT)
                            # print("2", m)
                            push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, next_cat], heuristics[s, r] + maximum_attachment[r] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), r, s, TRAP, m, tag1, tt2, u, data_cur, next_cat)
                            total += 1
                        push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, noncore_cat], heuristics[s, r] + maximum_attachment[r] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), r, s, TRAP, t1, tag1, tt2, u, data_cur, noncore_cat)
                        total += 1
                        inc(v_iter)

            idx = get_idx_supertag(RIGHT, TRAP, TAIL, s, tag1, nr)
            if pointers.count(idx):
                v_p = &pointers[idx]
                v_iter = v_p.begin()
                while v_iter != v_p.end():
                    u = deref(v_iter)
                    score2 = inside_score[u]
                    r = data[u][0]
                    t1 = data[u][3]
                    tt2 = data[u][5]
                    if r == 0 and t != N:
                        inc(v_iter)
                        continue
                    push_queue_supertag_new(queue, score + score2, heuristics[r, t] + maximum_attachment[r] + supertag_score_t(r, tt2, supertag, layer_index, n_layers), r, t, TRI, t1, 0, tt2, u, data_cur)
                    total += 1
                    inc(v_iter)

        elif triangle == TRI:
            if s >= t:
                idx = get_idx_supertag(LEFT, TRI, HEAD, s, tag1, nr)
                pointers[idx].push_back(data_cur)
                idx = get_idx_supertag(LEFT, TRI, TAIL, t, 0, nr)
                pointers[idx].push_back(data_cur)
                # m = supertag_mapping[tag1, LEFT]
                m = raise_to_full(tag1, npt, layer_index, n_layers, supertag_mapping, LEFT)
                # if m < 0:
                    # push_queue_supertag_new(queue, score, heuristics[t, s] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers), s, t, FLTRI, -m - 1, 0, tt1, data_cur, -1)
                if m >= 0:
                    push_queue_supertag_new(queue, score, heuristics[t, s] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers), s, t, FLTRI, m, 0, tt1, data_cur, -1)
            if s <= t:
                # print("a")
                idx = get_idx_supertag(RIGHT, TRI, HEAD, s, tag1, nr)
                pointers[idx].push_back(data_cur)
                idx = get_idx_supertag(RIGHT, TRI, TAIL, t, 0, nr)
                pointers[idx].push_back(data_cur)
                # m = supertag_mapping[tag1, RIGHT]
                m = raise_to_full(tag1, npt, layer_index, n_layers, supertag_mapping, RIGHT)
                # if m < 0:
                    # push_queue_supertag_new(queue, score, heuristics[s, t] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers), s, t, FRTRI, -m - 1, 0, tt1, data_cur, -1)
                if m >= 0:
                    push_queue_supertag_new(queue, score, heuristics[s, t] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers), s, t, FRTRI, m, 0, tt1, data_cur, -1)
                # print("b")

            if s == 0 and t == N:
                # print("c")
                break

            if s >= t:
                if t - 1 >= 0:
                    idx = get_idx_supertag(RIGHT, FRTRI, TAIL, t - 1, 0, nr)
                    if pointers.count(idx):
                        v_p = &pointers[idx]
                        v_iter = v_p.begin()
                        # m = supertag_mapping[tag1, LEFT]
                        while v_iter != v_p.end():
                            u = deref(v_iter)
                            score2 = inside_score[u]
                            r = data[u][0]
                            t1 = data[u][3]
                            tt2 = data[u][5]
                            # if m >= 0:
                                # push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, supertag_mapping[tag1, 2 + LEFT]], heuristics[r, s] + maximum_attachment[s] + supertag[s, tt1] / 2 + supertag[r, tt2] / 2, s, r, TRAP, m, t1, tt1, data_cur, u, supertag_mapping[tag1, 2 + LEFT])
                                # total += 1
                            # elif tag1 == 0:
                                # push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, 0], heuristics[r, s] + maximum_attachment[s] + supertag[s, tt1] / 2 + supertag[r, tt2] / 2, s, r, TRAP, tag1, t1, tt1, data_cur, u, 0)
                                # total += 1
                            for i in range(n_layers):
                                next_cat =  next_cat_layer(tag1, npt, n_layers, supertag_mapping, LEFT, i)
                                if next_cat < 0:
                                    continue
                                m = advance_pt(tag1, npt, n_layers, next_cat, supertag_mapping, LEFT)
                                # print("3", m)
                                push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, next_cat], heuristics[r, s] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), s, r, TRAP, m, t1, tt1, data_cur, u, next_cat)
                                total += 1
                                # print("d")

                            push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, noncore_cat], heuristics[r, s] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), s, r, TRAP, tag1, t1, tt1, data_cur, u, noncore_cat)
                            total += 1
                            inc(v_iter)
                            # print("d2")

            if s <= t:
                # print("d")
                if t + 1 < nr:
                    idx = get_idx_supertag(LEFT, FLTRI, TAIL, t + 1, 0, nr)
                    if pointers.count(idx):
                        v_p = &pointers[idx]
                        v_iter = v_p.begin()
                        # m = supertag_mapping[tag1, RIGHT]
                        while v_iter != v_p.end():
                            u = deref(v_iter)
                            score2 = inside_score[u]
                            r = data[u][0]
                            t1 = data[u][3]
                            tt2 = data[u][5]
                            # if m >= 0:
                                # push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, supertag_mapping[tag1, 2 + RIGHT]], heuristics[s, r] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), s, r, TRAP, m, t1, tt1, data_cur, u, supertag_mapping[tag1, 2 + RIGHT])
                                # total += 1
                            # elif tag1 == 0:
                                # push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, 0], heuristics[s, r] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), s, r, TRAP, tag1, t1, tt1, data_cur, u, 0)
                                # total += 1
                            for i in range(n_layers):
                                next_cat =  next_cat_layer(tag1, npt, n_layers, supertag_mapping, RIGHT, i)
                                if next_cat < 0:
                                    continue
                                m = advance_pt(tag1, npt, n_layers, next_cat, supertag_mapping, RIGHT)
                                # print("4", m)
                                push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, next_cat], heuristics[s, r] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), s, r, TRAP, m, t1, tt1, data_cur, u, next_cat)
                                total += 1
                            push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, noncore_cat], heuristics[s, r] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), s, r, TRAP, tag1, t1, tt1, data_cur, u, noncore_cat)
                            total += 1
                            inc(v_iter)
                # print("e")
            # print("d3")

        elif triangle == TRAP:
            idx = get_idx_supertag(RIGHT if s < t else LEFT, TRAP, TAIL, t, tag2, nr)
            pointers[idx].push_back(data_cur)

            if s > t:
                idx = get_idx_supertag(LEFT, FLTRI, HEAD, t, tag2, nr)
                if pointers.count(idx):
                    v_p = &pointers[idx]
                    v_iter = v_p.begin()
                    while v_iter != v_p.end():
                        u = deref(v_iter)
                        score2 = inside_score[u]
                        r = data[u][1]
                        push_queue_supertag_new(queue, score + score2, heuristics[r, s] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers), s, r, TRI, tag1, 0, tt1, data_cur, u)
                        total += 1
                        inc(v_iter)
            elif s < t:
                idx = get_idx_supertag(RIGHT, FRTRI, HEAD, t, tag2, nr)
                if pointers.count(idx):
                    v_p = &pointers[idx]
                    v_iter = v_p.begin()
                    while v_iter != v_p.end():
                        u = deref(v_iter)
                        score2 = inside_score[u]
                        r = data[u][1]
                        if s == 0 and r != N:
                            inc(v_iter)
                            continue
                        push_queue_supertag_new(queue, score + score2, heuristics[s, r] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers), s, r, TRI, tag1, 0, tt1, data_cur, u)
                        total += 1
                        inc(v_iter)

    # print(heuristics[0, 0])
    # print(score)
    # print("f")
    heads = -np.ones(nr, dtype=int)
    rels = -np.ones(nr, dtype=int)
    backtrack_hsel_supertag_new(data, data_cur, heads, rels)
    # print("g")

    for i in range(1, nr):
        rels[i] = cat2rel[heads[i], i, rels[i]]
    # print("h")
    # print(total)
    # sys.stdout.flush()
    # print(heads)
    # sys.stdout.flush()
    # print(rels)
    # sys.stdout.flush()

    return total, heads, rels


@cython.boundscheck(False)
@cython.wraparound(False)
def astar_hsel_supertag_multi_lexlimit(
        np.ndarray[np.float64_t, ndim=2] hsel, double[:,::1] supertag,
        np.ndarray[np.float64_t, ndim=3] label, int[:,::1] supertag_mapping,
        np.ndarray[np.npy_intp, ndim=3] cat2rel, int[::1] layer_index,
        int noncore_cat, int max_queue=-1, int max_lex=-1):
    cdef int nr, nc, nt, npt, n_layers, N, i, j, k, s, t, r, u, tag1, tag2, tt1, tt2, t1, t2, m, total, _
    cdef int cur_lex_no
    cdef int next_cat
    cdef ull idx
    cdef int triangle
    cdef int start, end
    cdef int data_i, data_cur
    cdef np.float64_t tmp, cand
    cdef float score, score2

    cdef np.ndarray[np.float64_t, ndim=2] heuristics
    cdef np.ndarray[np.float64_t, ndim=1] maximum_supertag
    cdef np.ndarray[np.float64_t, ndim=1] maximum_attachment
    cdef np.ndarray[np.npy_intp, ndim=1] heads
    cdef np.ndarray[np.npy_intp, ndim=1] rels

    cdef priority_queue[pair[float, pair[float, vector[int]]]] queue
    cdef priority_queue[pair[float, int]] lex_queue
    cdef pair[float, pair[float, vector[int]]] p
    cdef pair[float, vector[int]] item
    cdef vector[int] vec
    cdef int_vec_p v_p
    cdef vector[int].iterator v_iter
    cdef set[int] structure_set

    data_cur = -1
    total = 0
    nr, nc = np.shape(hsel)
    N = nr - 1

    npt, _ = np.shape(supertag_mapping)

    n_layers = len(layer_index) - 1
    # n_layers = 1

    nt = 1
    for i in range(n_layers):
        nt *= layer_index[i + 1] - layer_index[i]

    supertag[0,:] = 0.

    cdef umap[ull, vector[int]] pointers
    cdef vector[vector[int]] data
    cdef vector[float] inside_score

    # disable self attachment
    for i in range(nr):
        hsel[i, i] = NEGINF

    # pre-compute the heuristics, O(n^2)
    maximum_attachment = np.amax(hsel + np.amax(label, axis=2), axis=0)
    maximum_attachment[0] = 0.
    maximum_supertag = np.zeros((nr,), dtype=np.float)
    for i in range(n_layers):
        maximum_supertag += np.amax(supertag[:, layer_index[i]:layer_index[i+1]], axis=1)
    maximum_supertag[0] = 0.
    heuristics = np.zeros((nr, nr), dtype=np.float)
    for i in range(nr):
        for j in range(1, i):
            for k in range(i, nr):
                heuristics[i, k] += maximum_attachment[j] + maximum_supertag[j]
    for i in range(nr):
        for j in range(i + 1, nr):
            for k in range(0, i + 1):
                heuristics[k, i] += maximum_attachment[j] + maximum_supertag[j]

    cdef pair[float, int] l_pair
    for i in range(1, nr):
        cur_lex_no = 0
        for t in range(1, nt):
            l_pair = pair[float, int]()
            l_pair.first = supertag_score_t(i, t, supertag, layer_index, n_layers)
            l_pair.second = t
            lex_queue.push(l_pair)
        while not lex_queue.empty():
            l_pair = lex_queue.top()
            t = l_pair.second
            if max_lex <= 0 or cur_lex_no < max_lex:
                cur_lex_no += 1
                push_queue_supertag_new(queue, supertag_score_t(i, t, supertag, layer_index, n_layers), heuristics[i, i] + maximum_attachment[i] + supertag_score_t(i, t, supertag, layer_index, n_layers), i, i, TRI, t2pt(t, layer_index, npt, n_layers), 0, t, -1, -1)
            lex_queue.pop()
    # for t in range(nt):
    for t in range(1):
        push_queue_supertag_new(queue, supertag_score_t(0, t, supertag, layer_index, n_layers), heuristics[0, 0] + supertag_score_t(0, t, supertag, layer_index, n_layers), 0, 0, TRI, t2pt(t, layer_index, npt, n_layers), 0, t, -1, -1)

    while queue.size() > 0:
        if queue.size() > max_queue:
            return -1, None, None
        p = queue.top()
        item = p.second
        score = item.first
        vec = item.second
        s, t, triangle, tag1, tag2, tt1, comp1, comp2 = vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7]
        # idx = get_set_id_supertag(triangle, s, t, tag1, tag2, nr, nt)
        # print(p.first, score, vec)
        queue.pop()
        # print(s, t, triangle, tag1, tag2, tt1, comp1, comp2)

        idx = get_set_id_supertag(triangle, s, t, tag1, tag2, nr, nt)
        if structure_set.count(idx):
            continue

        structure_set.insert(idx)
        data.push_back(vec)
        inside_score.push_back(score)
        data_cur += 1
        # print(data_cur)

        if triangle == FLTRI:
            idx = get_idx_supertag(LEFT, FLTRI, HEAD, s, tag1, nr)
            pointers[idx].push_back(data_cur)
            idx = get_idx_supertag(LEFT, FLTRI, TAIL, t, 0, nr)
            pointers[idx].push_back(data_cur)

            if t - 1 >= 0:
                idx = get_idx_supertag(RIGHT, TRI, TAIL, t - 1, 0, nr)
                if pointers.count(idx):
                    v_p = &pointers[idx]
                    v_iter = v_p.begin()
                    while v_iter != v_p.end():
                        u = deref(v_iter)
                        score2 = inside_score[u]
                        r = data[u][0]
                        t1 = data[u][3]
                        tt2 = data[u][5]
                        # m = supertag_mapping[t1, RIGHT]
                        # if m >= 0:
                            # push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, supertag_mapping[t1, 2 + RIGHT]], heuristics[r, s] + maximum_attachment[r] + supertag_score_t(r, tt2, supertag, layer_index, n_layers) + supertag_score_t(s, tt1, supertag, layer_index, n_layers), r, s, TRAP, m, tag1, tt2, u, data_cur, supertag_mapping[t1, 2 + RIGHT])
                            # total += 1
                        # elif t1 == 0:
                            # push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, 0], heuristics[r, s] + maximum_attachment[r] + supertag_score_t(r, tt2, supertag, layer_index, n_layers) + supertag_score_t(s, tt1, supertag, layer_index, n_layers), r, s, TRAP, t1, tag1, tt2, u, data_cur, 0)
                            # total += 1
                        for i in range(n_layers):
                            next_cat =  next_cat_layer(t1, npt, n_layers, supertag_mapping, RIGHT, i)
                            if next_cat < 0:
                                continue
                            m = advance_pt(t1, npt, n_layers, next_cat, supertag_mapping, RIGHT)
                            # print("1", m)
                            push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, next_cat], heuristics[r, s] + maximum_attachment[r] + supertag_score_t(r, tt2, supertag, layer_index, n_layers) + supertag_score_t(s, tt1, supertag, layer_index, n_layers), r, s, TRAP, m, tag1, tt2, u, data_cur, next_cat)
                            total += 1
                        push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, noncore_cat], heuristics[r, s] + maximum_attachment[r] + supertag_score_t(r, tt2, supertag, layer_index, n_layers) + supertag_score_t(s, tt1, supertag, layer_index, n_layers), r, s, TRAP, t1, tag1, tt2, u, data_cur, noncore_cat)
                        total += 1
                        inc(v_iter)

            idx = get_idx_supertag(LEFT, TRAP, TAIL, s, tag1, nr)
            if pointers.count(idx):
                v_p = &pointers[idx]
                v_iter = v_p.begin()
                while v_iter != v_p.end():
                    u = deref(v_iter)
                    score2 = inside_score[u]
                    r = data[u][0]
                    t1 = data[u][3]
                    tt2 = data[u][5]
                    push_queue_supertag_new(queue, score + score2, heuristics[t, r] + maximum_attachment[r] + supertag_score_t(r, tt2, supertag, layer_index, n_layers), r, t, TRI, t1, 0, tt2, u, data_cur)
                    total += 1
                    inc(v_iter)

        elif triangle == FRTRI:
            idx = get_idx_supertag(RIGHT, FRTRI, HEAD, s, tag1, nr)
            pointers[idx].push_back(data_cur)
            idx = get_idx_supertag(RIGHT, FRTRI, TAIL, t, 0, nr)
            pointers[idx].push_back(data_cur)

            if t + 1 < nr:
                idx = get_idx_supertag(LEFT, TRI, TAIL, t + 1, 0, nr)
                if pointers.count(idx):
                    v_p = &pointers[idx]
                    v_iter = v_p.begin()
                    while v_iter != v_p.end():
                        u = deref(v_iter)
                        score2 = inside_score[u]
                        r = data[u][0]
                        t1 = data[u][3]
                        tt2 = data[u][5]
                        # m = supertag_mapping[t1, LEFT]
                        # if m >= 0:
                            # push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, supertag_mapping[t1, 2 + LEFT]], heuristics[s, r] + maximum_attachment[r] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), r, s, TRAP, m, tag1, tt2, u, data_cur, supertag_mapping[t1, 2 + LEFT])
                            # total += 1
                        # elif t1 == 0:
                            # push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, 0], heuristics[s, r] + maximum_attachment[r] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), r, s, TRAP, t1, tag1, tt2, u, data_cur, 0)
                            # total += 1
                        for i in range(n_layers):
                            next_cat =  next_cat_layer(t1, npt, n_layers, supertag_mapping, LEFT, i)
                            if next_cat < 0:
                                continue
                            m = advance_pt(t1, npt, n_layers, next_cat, supertag_mapping, LEFT)
                            # print("2", m)
                            push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, next_cat], heuristics[s, r] + maximum_attachment[r] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), r, s, TRAP, m, tag1, tt2, u, data_cur, next_cat)
                            total += 1
                        push_queue_supertag_new(queue, score + score2 + hsel[r, s] + label[r, s, noncore_cat], heuristics[s, r] + maximum_attachment[r] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), r, s, TRAP, t1, tag1, tt2, u, data_cur, noncore_cat)
                        total += 1
                        inc(v_iter)

            idx = get_idx_supertag(RIGHT, TRAP, TAIL, s, tag1, nr)
            if pointers.count(idx):
                v_p = &pointers[idx]
                v_iter = v_p.begin()
                while v_iter != v_p.end():
                    u = deref(v_iter)
                    score2 = inside_score[u]
                    r = data[u][0]
                    t1 = data[u][3]
                    tt2 = data[u][5]
                    if r == 0 and t != N:
                        inc(v_iter)
                        continue
                    push_queue_supertag_new(queue, score + score2, heuristics[r, t] + maximum_attachment[r] + supertag_score_t(r, tt2, supertag, layer_index, n_layers), r, t, TRI, t1, 0, tt2, u, data_cur)
                    total += 1
                    inc(v_iter)

        elif triangle == TRI:
            if s >= t:
                idx = get_idx_supertag(LEFT, TRI, HEAD, s, tag1, nr)
                pointers[idx].push_back(data_cur)
                idx = get_idx_supertag(LEFT, TRI, TAIL, t, 0, nr)
                pointers[idx].push_back(data_cur)
                # m = supertag_mapping[tag1, LEFT]
                m = raise_to_full(tag1, npt, layer_index, n_layers, supertag_mapping, LEFT)
                # if m < 0:
                    # push_queue_supertag_new(queue, score, heuristics[t, s] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers), s, t, FLTRI, -m - 1, 0, tt1, data_cur, -1)
                if m >= 0:
                    push_queue_supertag_new(queue, score, heuristics[t, s] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers), s, t, FLTRI, m, 0, tt1, data_cur, -1)
            if s <= t:
                # print("a")
                idx = get_idx_supertag(RIGHT, TRI, HEAD, s, tag1, nr)
                pointers[idx].push_back(data_cur)
                idx = get_idx_supertag(RIGHT, TRI, TAIL, t, 0, nr)
                pointers[idx].push_back(data_cur)
                # m = supertag_mapping[tag1, RIGHT]
                m = raise_to_full(tag1, npt, layer_index, n_layers, supertag_mapping, RIGHT)
                # if m < 0:
                    # push_queue_supertag_new(queue, score, heuristics[s, t] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers), s, t, FRTRI, -m - 1, 0, tt1, data_cur, -1)
                if m >= 0:
                    push_queue_supertag_new(queue, score, heuristics[s, t] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers), s, t, FRTRI, m, 0, tt1, data_cur, -1)
                # print("b")

            if s == 0 and t == N:
                # print("c")
                break

            if s >= t:
                if t - 1 >= 0:
                    idx = get_idx_supertag(RIGHT, FRTRI, TAIL, t - 1, 0, nr)
                    if pointers.count(idx):
                        v_p = &pointers[idx]
                        v_iter = v_p.begin()
                        # m = supertag_mapping[tag1, LEFT]
                        while v_iter != v_p.end():
                            u = deref(v_iter)
                            score2 = inside_score[u]
                            r = data[u][0]
                            t1 = data[u][3]
                            tt2 = data[u][5]
                            # if m >= 0:
                                # push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, supertag_mapping[tag1, 2 + LEFT]], heuristics[r, s] + maximum_attachment[s] + supertag[s, tt1] / 2 + supertag[r, tt2] / 2, s, r, TRAP, m, t1, tt1, data_cur, u, supertag_mapping[tag1, 2 + LEFT])
                                # total += 1
                            # elif tag1 == 0:
                                # push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, 0], heuristics[r, s] + maximum_attachment[s] + supertag[s, tt1] / 2 + supertag[r, tt2] / 2, s, r, TRAP, tag1, t1, tt1, data_cur, u, 0)
                                # total += 1
                            for i in range(n_layers):
                                next_cat =  next_cat_layer(tag1, npt, n_layers, supertag_mapping, LEFT, i)
                                if next_cat < 0:
                                    continue
                                m = advance_pt(tag1, npt, n_layers, next_cat, supertag_mapping, LEFT)
                                # print("3", m)
                                push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, next_cat], heuristics[r, s] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), s, r, TRAP, m, t1, tt1, data_cur, u, next_cat)
                                total += 1
                                # print("d")

                            push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, noncore_cat], heuristics[r, s] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), s, r, TRAP, tag1, t1, tt1, data_cur, u, noncore_cat)
                            total += 1
                            inc(v_iter)
                            # print("d2")

            if s <= t:
                # print("d")
                if t + 1 < nr:
                    idx = get_idx_supertag(LEFT, FLTRI, TAIL, t + 1, 0, nr)
                    if pointers.count(idx):
                        v_p = &pointers[idx]
                        v_iter = v_p.begin()
                        # m = supertag_mapping[tag1, RIGHT]
                        while v_iter != v_p.end():
                            u = deref(v_iter)
                            score2 = inside_score[u]
                            r = data[u][0]
                            t1 = data[u][3]
                            tt2 = data[u][5]
                            # if m >= 0:
                                # push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, supertag_mapping[tag1, 2 + RIGHT]], heuristics[s, r] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), s, r, TRAP, m, t1, tt1, data_cur, u, supertag_mapping[tag1, 2 + RIGHT])
                                # total += 1
                            # elif tag1 == 0:
                                # push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, 0], heuristics[s, r] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), s, r, TRAP, tag1, t1, tt1, data_cur, u, 0)
                                # total += 1
                            for i in range(n_layers):
                                next_cat =  next_cat_layer(tag1, npt, n_layers, supertag_mapping, RIGHT, i)
                                if next_cat < 0:
                                    continue
                                m = advance_pt(tag1, npt, n_layers, next_cat, supertag_mapping, RIGHT)
                                # print("4", m)
                                push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, next_cat], heuristics[s, r] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), s, r, TRAP, m, t1, tt1, data_cur, u, next_cat)
                                total += 1
                            push_queue_supertag_new(queue, score + score2 + hsel[s, r] + label[s, r, noncore_cat], heuristics[s, r] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers) + supertag_score_t(r, tt2, supertag, layer_index, n_layers), s, r, TRAP, tag1, t1, tt1, data_cur, u, noncore_cat)
                            total += 1
                            inc(v_iter)
                # print("e")
            # print("d3")

        elif triangle == TRAP:
            idx = get_idx_supertag(RIGHT if s < t else LEFT, TRAP, TAIL, t, tag2, nr)
            pointers[idx].push_back(data_cur)

            if s > t:
                idx = get_idx_supertag(LEFT, FLTRI, HEAD, t, tag2, nr)
                if pointers.count(idx):
                    v_p = &pointers[idx]
                    v_iter = v_p.begin()
                    while v_iter != v_p.end():
                        u = deref(v_iter)
                        score2 = inside_score[u]
                        r = data[u][1]
                        push_queue_supertag_new(queue, score + score2, heuristics[r, s] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers), s, r, TRI, tag1, 0, tt1, data_cur, u)
                        total += 1
                        inc(v_iter)
            elif s < t:
                idx = get_idx_supertag(RIGHT, FRTRI, HEAD, t, tag2, nr)
                if pointers.count(idx):
                    v_p = &pointers[idx]
                    v_iter = v_p.begin()
                    while v_iter != v_p.end():
                        u = deref(v_iter)
                        score2 = inside_score[u]
                        r = data[u][1]
                        if s == 0 and r != N:
                            inc(v_iter)
                            continue
                        push_queue_supertag_new(queue, score + score2, heuristics[s, r] + maximum_attachment[s] + supertag_score_t(s, tt1, supertag, layer_index, n_layers), s, r, TRI, tag1, 0, tt1, data_cur, u)
                        total += 1
                        inc(v_iter)

    # print(heuristics[0, 0])
    # print(score)
    # print("f")
    heads = -np.ones(nr, dtype=int)
    rels = -np.ones(nr, dtype=int)
    backtrack_hsel_supertag_new(data, data_cur, heads, rels)
    # print("g")

    for i in range(1, nr):
        rels[i] = cat2rel[heads[i], i, rels[i]]
    # print("h")
    # print(total)
    # sys.stdout.flush()
    # print(heads)
    # sys.stdout.flush()
    # print(rels)
    # sys.stdout.flush()

    return total, heads, rels
