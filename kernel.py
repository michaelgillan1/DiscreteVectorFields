import jax.numpy as jnp
import numpy as np
import scipy.linalg


def matern_scaling(l, k, v, dim):
    if v == np.inf:
        return np.exp(-((k ** 2) / 2.) * l)
    else:
        return np.power(((2 * v) / (k ** 2)) + l, - v - (dim / 2.))


def matern_scaling_ns(l, k, v, dim):
    if v == jnp.inf:
        return jnp.exp(-((k ** 2) / 2.) * l)
    else:
        return jnp.power(((2 * v) / (k[:, None].T ** 2)) + l[:, None], - v - (dim / 2.))


def vector_mesh_kernel(eigenvalues, d_eigenfields, c_eigenfields, points_1, points_2, dim, params, vol,
                       h_eigenfields=None):
    ms_d = matern_scaling(eigenvalues[1:], params["k_d"], params["v"], dim)
    ms_c = matern_scaling(eigenvalues[1:], params["k_c"], params["v"], dim)

    ms_d = ms_d / np.sqrt(eigenvalues[1:])
    ms_c = ms_c / np.sqrt(eigenvalues[1:])

    # Subset the points of the eigenfields that you are interested in.
    d_fields_1 = d_eigenfields[points_1, :, 1:] @ np.diag(ms_d)
    d_fields_2 = d_eigenfields[points_2, :, 1:]
    c_fields_1 = c_eigenfields[points_1, :, 1:] @ np.diag(ms_c)
    c_fields_2 = c_eigenfields[points_2, :, 1:]

    # Compute the K_d and K_c matrices
    # This does the outer product over the vector dimension and then sums over the eigen dimension
    K_d = np.einsum('ijm,klm->ikjl', d_fields_1, d_fields_2)
    K_c = np.einsum('ijm,klm->ikjl', c_fields_1, c_fields_2)

    K_d = K_d.transpose((0, 2, 1, 3)).reshape((len(points_1) * dim, len(points_2) * dim))
    K_c = K_c.transpose((0, 2, 1, 3)).reshape((len(points_1) * dim, len(points_2) * dim))

    # Compute scaling coefficients
    c_d = np.sum(ms_d[1:]) / vol
    c_c = np.sum(ms_c[1:]) / vol

    # Combine the matrices with the variance factors
    K = ((params["s_d"] ** 2) / c_d) * K_d + ((params["s_c"] ** 2) / c_c) * K_c

    if h_eigenfields is not None:
        h_fields_1 = h_eigenfields[points_1]
        h_fields_2 = h_eigenfields[points_2]

        K_h = np.einsum('ijm,klm->ikjl', h_fields_1, h_fields_2)
        K_h = K_h.transpose((0, 2, 1, 3)).reshape((len(points_1) * dim, len(points_2) * dim))

        K = K + (params["s_h"] ** 2) * K_h

    return K


def vector_mesh_kernel_ns(eigenvalues, d_eigenfields, c_eigenfields, points_1, points_2, dim, vol, kappa_d, kappa_c,
                          params, h_eigenfields=None):
    eigenvalues = jnp.array(eigenvalues)

    d_eigenfields = jnp.array(d_eigenfields)
    c_eigenfields = jnp.array(c_eigenfields)

    h_eigenfields = jnp.array(h_eigenfields)

    kappa_d = jnp.array(kappa_d)
    kappa_c = jnp.array(kappa_c)

    points_1 = jnp.array(points_1)
    points_2 = jnp.array(points_2)

    ms_d = matern_scaling_ns(eigenvalues, kappa_d, params["v"], dim)
    ms_c = matern_scaling_ns(eigenvalues, kappa_c, params["v"], dim)

    ms_d = ms_d / jnp.sqrt(eigenvalues[:, jnp.newaxis])
    ms_c = ms_c / jnp.sqrt(eigenvalues[:, jnp.newaxis])

    # Subset the points of the eigenfields that you are interested in.
    d_fields_1 = jnp.multiply(d_eigenfields[points_1], jnp.expand_dims(jnp.sqrt(ms_d.T[points_1]), 1))
    d_fields_2 = jnp.multiply(d_eigenfields[points_2], jnp.expand_dims(jnp.sqrt(ms_d.T[points_2]), 1))
    c_fields_1 = jnp.multiply(c_eigenfields[points_1], jnp.expand_dims(jnp.sqrt(ms_c.T[points_1]), 1))
    c_fields_2 = jnp.multiply(c_eigenfields[points_2], jnp.expand_dims(jnp.sqrt(ms_c.T[points_2]), 1))

    K_d = jnp.einsum('ijm,klm->ikjl', d_fields_1, d_fields_2)
    K_c = jnp.einsum('ijm,klm->ikjl', c_fields_1, c_fields_2)

    K_d = K_d.transpose((0, 2, 1, 3)).reshape((len(points_1) * dim, len(points_2) * dim))
    K_c = K_c.transpose((0, 2, 1, 3)).reshape((len(points_1) * dim, len(points_2) * dim))

    # Compute scaling coefficients
    c_d = jnp.sum(jnp.mean(ms_d, axis=1)) / vol
    c_c = jnp.sum(jnp.mean(ms_c, axis=1)) / vol

    # Final kernel with variance scaling
    K = ((params["s_d"] ** 2) / c_d) * K_d + ((params["s_c"] ** 2) / c_c) * K_c

    if h_eigenfields is not None:
        h_fields_1 = h_eigenfields[points_1]
        h_fields_2 = h_eigenfields[points_2]

        K_h = jnp.einsum('ijm,klm->ikjl', h_fields_1, h_fields_2)
        K_h = K_h.transpose((0, 2, 1, 3)).reshape((len(points_1) * dim, len(points_2) * dim))

        K = K + (params["s_h"] ** 2) * K_h

    return K


def sample_vector_field(eigenvalues, d_eigenfield, c_eigenfield, dim, vol, params, h_eigenfield=None):
    ms_d = matern_scaling(eigenvalues, params["k_d"], params["v"], dim)
    ms_c = matern_scaling(eigenvalues, params["k_c"], params["v"], dim)

    ms_d = ms_d / np.sqrt(eigenvalues)
    ms_c = ms_c / np.sqrt(eigenvalues)

    vs_d = np.zeros((len(d_eigenfield), len(d_eigenfield[1])))
    vs_c = np.zeros((len(c_eigenfield), len(c_eigenfield[1])))
    for i in range(1, len(ms_d)):
        vs_d = vs_d + np.array(d_eigenfield[:, :, i]) * np.random.normal(0, 1, 1) * np.sqrt(ms_d[i])
        vs_c = vs_c + np.array(c_eigenfield[:, :, i]) * np.random.normal(0, 1, 1) * np.sqrt(ms_c[i])

    c_d = sum(ms_d) / vol
    c_c = sum(ms_d) / vol

    vs_d = vs_d * params["s_d"] / np.sqrt(c_d)
    vs_c = vs_c * params["s_c"] / np.sqrt(c_c)

    field = vs_d + vs_c

    if h_eigenfield is not None:
        vs_h = np.zeros((len(d_eigenfield), len(d_eigenfield[1])))
        for i in range(h_eigenfield.shape[2]):
            vs_h = np.array(h_eigenfield[:, :, i]) * np.random.normal(0, 1, 1)

        field = field + vs_h * params["s_h"]

    return field


def sample_vector_field_ns(eigenvalues, d_eigenfields, c_eigenfields, dim, vol, kappa_d, kappa_c, params,
                           h_eigenfield=None):
    # kappa has a value at every grid point, shape: [N,]

    ms_d = matern_scaling_ns(eigenvalues, kappa_d, params["v"], dim)
    ms_c = matern_scaling_ns(eigenvalues, kappa_c, params["v"], dim)

    ms_d = ms_d / np.sqrt(eigenvalues[:, None])
    ms_c = ms_c / np.sqrt(eigenvalues[:, None])

    vs_d = np.zeros((len(d_eigenfields), len(d_eigenfields[1])))
    vs_c = np.zeros((len(c_eigenfields), len(c_eigenfields[1])))
    for i in range(1, len(ms_d)):
        vs_d = vs_d + np.diag(np.sqrt(ms_d[i])) @ np.array(d_eigenfields[:, :, i]) * np.random.normal(0, 1, 1)
        vs_c = vs_c + np.diag(np.sqrt(ms_c[i])) @ np.array(c_eigenfields[:, :, i]) * np.random.normal(0, 1, 1)

    c_d = np.sum(np.mean(ms_d, axis=1)) / vol
    c_c = np.sum(np.mean(ms_c, axis=1)) / vol

    vs_d = vs_d * params["s_d"] / np.sqrt(c_d)
    vs_c = vs_c * params["s_c"] / np.sqrt(c_c)

    field = vs_d + vs_c

    if h_eigenfield is not None:
        vs_h = np.zeros((len(d_eigenfields), len(d_eigenfields[1])))
        for i in range(h_eigenfield.shape[2]):
            vs_h = np.array(h_eigenfield[:, :, i]) * np.random.normal(0, 1, 1)

        field = field + vs_h * params["s_h"]

    return field


def predict_vectors(eigenvalues, d_eigenvectors, c_eigenvectors, obs_idx, pred_idx, vec_obs, dim, params,
                    vol, tau=1e-6, cov=False):
    K_vec_x_x = vector_mesh_kernel(eigenvalues, d_eigenvectors, c_eigenvectors, obs_idx, obs_idx, dim, params,
                                   vol)
    K_vec_x_x = K_vec_x_x + tau * np.eye(len(obs_idx) * dim)

    print(f"Condition number of K_vec_x_x: {np.linalg.cond(K_vec_x_x):.2e}")

    # obs to everything
    K_vec_x_x_star = vector_mesh_kernel(eigenvalues, d_eigenvectors, c_eigenvectors, obs_idx, pred_idx, dim,
                                        params, vol)

    post_cov = None

    if cov:
        print("Generating posterior covariance...")
        K_vec_x_star_x_star = vector_mesh_kernel(eigenvalues, d_eigenvectors, c_eigenvectors, pred_idx,
                                                 pred_idx, dim, params, vol)
        K_vec_x_star_x_star = K_vec_x_star_x_star + tau * np.eye(len(pred_idx) * dim)
        print(f"Condition number of K_vec_x_star_x_star: {np.linalg.cond(K_vec_x_star_x_star):.2e}")
        eigs = np.linalg.eigvalsh(K_vec_x_star_x_star)
        print("min eigval:", np.min(eigs))
        print("rank (eig > 1e-10):", np.sum(eigs > 1e-10), "/", len(eigs))
        post_cov = K_vec_x_star_x_star - K_vec_x_x_star.T @ np.linalg.solve(K_vec_x_x, K_vec_x_x_star)

    # flatten obs

    z_vec_obs = vec_obs.reshape(-1)

    # predict
    post_mean = K_vec_x_x_star.T @ scipy.linalg.solve(K_vec_x_x, z_vec_obs)

    return post_mean.reshape(len(pred_idx), dim), post_cov
