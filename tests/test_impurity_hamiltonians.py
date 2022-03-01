#!/usr/bin/env/python
# coding: utf-8

import numpy as np
from numpy import conj, dot
from scipy.sparse.linalg import eigsh
import itertools

from qat.fermion.util import init_creation_ops, dag
from qat.fermion.impurity.hamiltonians import make_embedded_model, make_anderson_model


def test_anderson():
    U = 1
    mu = U / 2
    anderson_hamilt = make_anderson_model(U, mu, [0.4], [0.04])
    embedding_hamilt = make_embedded_model(
        U, mu, np.diag([0.4, 0.4]), -0.04 * np.eye(2), grouping="clusters"
    )
    embedding_hamilt.constant_coeff = 0
    anderson_mat = np.real(anderson_hamilt.get_matrix(sparse=True))
    embedding_mat = np.real(embedding_hamilt.get_matrix(sparse=True))
    assert np.max(anderson_mat - embedding_mat) < 1e-8


def test_risb_embedding():
    Nc = 2
    M = 2 * Nc

    U = 0.1
    mu = U / 2
    np.random.seed(0)
    D = np.diag(np.random.random(4))
    lambda_c = np.diag(np.random.random(4))
    t_pm = np.diag(np.random.random(4))

    d_dag = init_creation_ops(2 * M, sparse="True")
    d = {ind: dag(c_dag) for ind, c_dag in d_dag.items()}

    int_kernel = 1.0 * np.fromfunction(
        lambda i, j, k, l: (i == j) * (i % 2 == 0) * (k == l) * (k == i + 1) * 1,
        (2 * Nc, 2 * Nc, 2 * Nc, 2 * Nc),
        dtype=float,
    )
    int_kernel_sym = 1.0 * np.fromfunction(
        lambda i, j, k, l: (i == j) * (i % 2 == 1) * (k == l) * (k == i - 1) * 1,
        (2 * Nc, 2 * Nc, 2 * Nc, 2 * Nc),
        dtype=float,
    )
    int_kernel += int_kernel_sym

    # cluster ordering:
    int_kernel_fermion = 1.0 * np.fromfunction(
        lambda i, j, k, l: (i == k) * (i % 2 == 0) * (j == l) * (j == i + 1) * 1,
        (2 * Nc, 2 * Nc, 2 * Nc, 2 * Nc),
        dtype=float,
    )
    int_kernel_sym_fermion = 1.0 * np.fromfunction(
        lambda i, j, k, l: (i == k) * (i % 2 == 1) * (j == l) * (j == i - 1) * 1,
        (2 * Nc, 2 * Nc, 2 * Nc, 2 * Nc),
        dtype=float,
    )

    int_kernel_fermion += int_kernel_sym_fermion
    extended_int_kernel_fermion = np.zeros((2 * M, 2 * M, 2 * M, 2 * M))
    extended_int_kernel_fermion[
        : (2 * Nc), : (2 * Nc), : (2 * Nc), : (2 * Nc)
    ] = int_kernel_fermion

    int_term = (
        U
        / 2
        * sum(
            [
                int_kernel[i, j, k, l] * d_dag[i].dot(d[j]).dot(d_dag[k]).dot(d[l])
                for i, j, k, l in itertools.product(range(M), repeat=4)
            ]
        )
    )

    non_int_term = (
        sum(
            [
                lambda_c[i, j] * dot(d[M + j], d_dag[M + i])
                for i, j in itertools.product(list(range(M)), list(range(M)))
            ]
        )
        + sum(
            [
                D[i, j] * dot(d_dag[i], d[M + j])
                + conj(D[i, j]) * dot(d_dag[M + j], d[i])
                for i, j in itertools.product(list(range(M)), list(range(M)))
            ]
        )
        + sum(
            [
                t_pm[i, j] * dot(d_dag[i], d[j])
                for i, j in itertools.product(range(M), repeat=2)
            ]
        )
        - mu * sum([d_dag[i].dot(d[i]) for i in range(M)])
    )

    hamilt_emb_fermion = make_embedded_model(
        U,
        mu,
        D,
        lambda_c,
        t_loc=t_pm,
        int_kernel=extended_int_kernel_fermion,
        grouping="clusters",
    )
    h_emb_pm = hamilt_emb_fermion.get_matrix(sparse=True)

    H_emb = non_int_term + int_term

    eig, psi = eigsh(h_emb_pm, k=6, which="SA")
    Eig, Psi = eigsh(H_emb, k=6, which="SA")

    np.testing.assert_almost_equal(min(eig), min(Eig), decimal=7)
