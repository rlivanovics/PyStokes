"""
This library computes the mobility coefficients of N rigid spheres in Stokes flow, given the sphere coordinates,
velocities and angular velocities following the Filippov multipole expansion method described in:
Filippov, A. V. "Drag and Torque on Clusters of N Arbitrary Spheres at Low Reynolds Number."
Journal of colloid and interface science 229.1 (2000): 184-195.
DOI: 10.1006/jcis.2000.6981
"""
import numpy as np


def import_sphere_data(input_file='data.txt'):
    import csv

    reader = csv.reader(open(input_file, 'r'), delimeter=' ')
    x, y, z = [], [], []
    for row in reader:
        x.append(row[0])
        y.append(row[1])
        z.append(row[2])
    return x, y, z


def calculate_sphere_vectors(x, y, z):
    """
    Given 3 lists of sphere coordinates x, y and z, computes the vectors R_ij connecting the sphere centers:
    r_ij = (x_j, y_j, z_j) - (x_i, y_i, z_i)
    r_ij is a N*N*3 matrix for N spheres
    """
    n = len(x)
    r = np.zeros((n, n, 3))
    for i in xrange(n):
        for j in xrange(n):
            r[i, j] = (x[j] - x[i], y[j] - y[i], z[j] - z[i])
    return r


def kronecker_delta(i, j):
    if i == j:
        return 1
    else:
        return 0


def compute_velocity_coefficients(v_sphere, w_sphere, a):
    """
    Computes the coefficient matrices Xi_mn, Yi_mn, Zi_mn assuming that v_sphere and w_sphere are N*3 vectors
    representing the linear and angular velocities of N rigid spheres.

    Since m goes from -1 to +Inf, whereas n goes from +1 to +Inf, the second index (corresponding to m) is moved
    forward by two units, meaning that, for example, X[i, 2, 0] corresponds to X at i = i, m = 1 and n = 1.

    The coefficient matrices should in theory be of size N * n_max * (n_max + 2), but it seems (if I'm not mistaken)
    that n_max = 1 for rigid spheres, therefore each coefficient matrix is of size N * 3.
    """
    n = int(np.shape(v_sphere)[0])
    X = np.zeros((n, 3))
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)
    for i in range(n):
        X[i, 2] = 0.5 * (v_sphere[i, 0] - 1j * v_sphere[i, 1])
        X[i, 0] = -v_sphere[i, 0] - 1j * v_sphere[i, 1]
        X[i, 1] = v_sphere[i, 2]
        Z[i, 2] = a[i] * (w_sphere[i, 0] - 1j * w_sphere[i, 1])
        Z[i, 0] = -2 * a[i] * (w_sphere[i, 0] + 1j * w_sphere[i, 1])
        Z[i, 1] = 2 * a[i] * w_sphere[i, 2]
    return X, Y, Z


def compute_transformation_coefficients():
    """
    Computes the transformation coefficient matrix Cij_mnkl which transforms the regular solid harmonics from sphere
    origin j to sphere origin i in the form:
    uj-_mn(r_j, theta_j, phi_j) = Sum(l=0, L) Sum(k=-l, l) Cij_mnkl * ui+_kl(r_i, theta_i, phi_i),
    where ui+_kl = rl_i * Pk_l(cos(theta_i)) * exp(1j * m * phi_i).
    Note: not entirely clear if the harmonics require any normalization terms or not.
    Indices mathematically defined as:
    i = 1 ... N
    j = 1 ... N
    m = 1 ... 3 ?
    n = 1 ... 3 ?
    k = -l ... L
    l = 0 ... L
    """


if __name__ == "__main__":
    x, y, z = import_sphere_data('coordinates.txt')
    vx, vy, vz = import_sphere_data('velocities.txt')
    v_sphere = np.vstack((vx, vy, vz)).transpose()
    wx, wy, wz = import_sphere_data('angvel.txt')
    w_sphere = np.vstack((wx, wy, wz)).transpose()
    r = calculate_sphere_vectors(x, y, z)
    n = len(x)
    a = np.ones(n)

    """
        TODO:
        1. compute Xi_mn, Yi_mn, Zi_mn coefficient matrices from sphere velocities Ui                               DONE?
        2. compute the transformation coefficient matrix Cij_mnkl from the positive/negative spherical harmonics
        3. compute the auxiliary coefficient matrices Dij_klmn to Nij_klmn using Cij_mnkl from the appendix
        4. construct the linear system of 3*N*L*(L+2) equations for the coefficients ai_mn, bi_mn and ci_mn
        5. solve it
        6. ???
        7. profit!
    """
    X, Y, Z = compute_velocity_coefficients(v_sphere, w_sphere, a)
