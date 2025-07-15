import numpy as np
import pydec
import scipy.linalg
import trimesh
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib


def generate_vector_basis(vertices=None, simplices=None, complex=None, boundary=None, flat=False, tolerance=1e-10,
                          depth=100):
    if complex is None:
        complex = pydec.SimplicialComplex((vertices, simplices))

    complex.construct_hodge()

    L = complex[0].d.T @ complex[1].star @ complex[0].d

    if depth is None:
        depth = len(complex.vertices) - 2

    # on flat domains we do not require the inclusion of the metric via the mass matrix
    if not flat:
        eigenvalues, d_eigenvectors = scipy.sparse.linalg.eigsh(A=L, M=complex[0].star,
                                                                k=depth,
                                                                sigma=0,
                                                                mode='normal',
                                                                tol=tolerance,
                                                                ncv=min(4 * depth, L.shape[0] - 1)
                                                                )
    else:
        eigenvalues, d_eigenvectors = scipy.sparse.linalg.eigsh(A=L,
                                                                k=depth,
                                                                sigma=0,
                                                                mode='normal',
                                                                tol=tolerance,
                                                                ncv=min(4 * depth, L.shape[0] - 1)
                                                                )

    eigenvalues[abs(eigenvalues) < tolerance] = 0

    c_eigenvectors = np.copy(d_eigenvectors)

    if boundary is not None:
        def unpack(b):
            for v1, v2 in b:
                yield v1
                yield v2

        boundary_vertices = np.array(sorted(set(unpack(boundary))))
        all_vertices = np.arange(L.shape[0])
        interior_vertices = np.setdiff1d(all_vertices, boundary_vertices)

        # Extract interior submatrix
        L_ii = L[interior_vertices[:, None], interior_vertices]

        # Recompute eigenvectors with Dirichlet zero boundary condition
        _, eigvecs_int = scipy.sparse.linalg.eigsh(L_ii, k=depth, which='SM')

        # Assemble full eigenvectors with zero at boundary
        c_eigenvectors = np.zeros((L.shape[0], depth))
        c_eigenvectors[interior_vertices, :] = eigvecs_int

    d_data = complex[0].d @ d_eigenvectors
    c_data = complex[0].d @ c_eigenvectors

    dim = complex.embedding_dimension()
    num_vertices = len(complex.vertices)
    num_simplices = len(complex[-1].simplices)

    d_vert_vecs = np.zeros((num_vertices, dim, depth))
    c_vert_vecs = np.zeros((num_vertices, dim, depth))

    edge_idx = complex[1].simplex_to_index

    # Precompute tri_normals
    if dim > 2:
        tri_mesh = trimesh.Trimesh(complex.vertices, complex.simplices)
        trimesh.repair.fix_normals(tri_mesh)

        tri_normals = tri_mesh.face_normals
    else:
        tri_normals = np.zeros((num_simplices, 3))
        tri_normals[:, 2] = 1  # [0,0,1] for all simplices

    simplices_indices = complex[-1].simplices
    verts_indices = simplices_indices

    # Precompute edge indices for each simplex
    simplices_edges = np.array([
        [edge_idx[pydec.Simplex((s[i], s[j]))] for i, j in [(0, 1), (0, 2), (1, 2)]
         ] for s in simplices_indices])

    # Precompute grad_diff for each simplex
    grad_diff = np.zeros((num_simplices, 3, dim))
    for s_idx, s in enumerate(simplices_indices):
        vert_pos = complex.vertices[s]
        d_lambda = pydec.barycentric_gradients(vert_pos)
        grad_diff[s_idx, 0] = d_lambda[1] - d_lambda[0]
        grad_diff[s_idx, 1] = d_lambda[2] - d_lambda[0]
        grad_diff[s_idx, 2] = d_lambda[2] - d_lambda[1]

    # Precompute angles and weights
    vert_pos = complex.vertices[np.array(simplices_indices)]

    v0 = vert_pos[:, 0, :]
    v1 = vert_pos[:, 1, :]
    v2 = vert_pos[:, 2, :]

    # Compute edge vectors for each angle:
    # angle at v0: edges from v0 to v1 and v0 to v2
    vec0 = np.stack([v1 - v0, v2 - v0], axis=1)
    # angle at v1: edges from v1 to v0 and v1 to v2
    vec1 = np.stack([v0 - v1, v2 - v1], axis=1)
    # angle at v2: edges from v2 to v0 and v2 to v1
    vec2 = np.stack([v0 - v2, v1 - v2], axis=1)

    # Combine the computed edge vectors to get (num_simplices, 3, 2, dim)
    vectors = np.stack([vec0, vec1, vec2], axis=1)

    # dot product between the two edge vectors at each angle
    dots = np.sum(vectors[:, :, 0, :] * vectors[:, :, 1, :], axis=-1)
    norms = np.linalg.norm(vectors, axis=3)
    cos_theta = dots / (norms[:, :, 0] * norms[:, :, 1])
    angles = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    weights = angles / (2 * np.pi)

    d_form_values = d_data[simplices_edges, :].transpose(2, 0, 1)
    c_form_values = c_data[simplices_edges, :].transpose(2, 0, 1)

    d_tri_vecs = np.einsum('skd,isk->isd', grad_diff, d_form_values) / 3
    d_tri_vecs = np.transpose(d_tri_vecs, (1, 0, 2))

    c_tri_vecs = np.einsum('skd,isk->isd', grad_diff, c_form_values) / 3
    c_tri_vecs = np.transpose(c_tri_vecs, (1, 0, 2))

    tri_normals_expanded = np.repeat(tri_normals[:, None, :], depth, axis=1)
    # if dim < 3:
    #     tri_normals_expanded = np.pad(tri_normals_expanded, ((0, 0), (0, 0), (0, 3 - dim)))

    rot_tri_vecs = np.cross(c_tri_vecs, tri_normals_expanded)[..., :dim]

    vertices_flat = verts_indices.astype(np.int64).reshape(-1)

    # Parallel processing of each eigenvector
    def process_eigenvector(i):
        d_vert_i = np.zeros((num_vertices, dim))
        c_vert_i = np.zeros((num_vertices, dim))

        tri_contrib = d_tri_vecs[:, i, :]
        rot_contrib = rot_tri_vecs[:, i, :]

        contrib_d = tri_contrib[:, None, :] * weights[:, :, None]
        contrib_c = rot_contrib[:, None, :] * weights[:, :, None]

        contrib_d_flat = contrib_d.reshape(-1, dim)
        contrib_c_flat = contrib_c.reshape(-1, dim)

        for d in range(dim):
            np.add.at(d_vert_i[:, d], vertices_flat, contrib_d_flat[:, d])
            np.add.at(c_vert_i[:, d], vertices_flat, contrib_c_flat[:, d])

        return d_vert_i, c_vert_i

    with tqdm_joblib(tqdm(desc="Processing eigenvectors", total=depth)):
        results = Parallel(n_jobs=-1)(delayed(process_eigenvector)(i) for i in range(depth))

    for i, (d_vert, c_vert) in enumerate(results):
        d_vert_vecs[:, :, i] = d_vert
        c_vert_vecs[:, :, i] = c_vert

    vol = np.sum(complex[-1].primal_volume)

    return eigenvalues, d_vert_vecs, c_vert_vecs, vol, complex


def generate_harmonic_basis(complex, dim=2):
    def edge2vec(complex, edge_vals, dim):
        vert_vecs = np.zeros((len(complex.vertices), dim))

        edge_idx = complex[1].simplex_to_index

        for n, s in enumerate(complex[-1].simplices):
            verts = s  # sorted(s)
            vert_pos = complex.vertices[verts]

            d_lambda = pydec.barycentric_gradients(complex.vertices[verts, :])
            edges = [pydec.Simplex(x) for x in pydec.combinations(s, 2)]
            indices = [edge_idx[x] for x in edges]
            values = [edge_vals[idx] for idx in indices]

            tri_vec = np.zeros(dim)

            for e, v in zip(pydec.combinations(range(len(verts)), 2), values):
                tri_vec += v * (d_lambda[e[1]] - d_lambda[e[0]])

            tri_vec /= (complex.complex_dimension() + 1)

            thetas = []
            for j, v in enumerate(vert_pos):
                other_vs = np.delete(vert_pos, j, axis=0)
                a, b = other_vs - v
                theta = np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))
                thetas.append(theta)

            weights = np.array(thetas) / (2 * np.pi)
            weights /= weights.sum()

            vert_vecs[verts, :] += weights[:, None] * tri_vec

        return vert_vecs

    hodge_laplacian = complex[1].d.T @ complex[2].star @ complex[1].d + complex[1].star @ complex[0].d @ complex[
        0].star_inv @ complex[0].d.T @ complex[1].star

    depth = 5
    tolerance = 1e-10

    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(A=hodge_laplacian, M=complex[1].star,
                                                          k=depth,
                                                          sigma=0,  # Target eigenvalues near 0
                                                          mode='normal',  # (L - σI)^-1
                                                          tol=tolerance,  # Match your later zeroing threshold
                                                          ncv=min(4 * depth, hodge_laplacian.shape[0] - 1),
                                                          # More Lanczos vectors
                                                          maxiter=5000  # Allow more iterations if needed
                                                          )

    harmonics_forms = eigenvectors[:, eigenvalues < tolerance]
    n_forms = harmonics_forms.shape[1]

    harmonic_fields = np.zeros((complex[0].num_simplices, dim, n_forms))

    for i in range(n_forms):
        harmonic_fields[:, :, i] = edge2vec(complex, harmonics_forms[:, i], dim)

    return harmonic_fields


def generate_scalar_basis(vertices, simplices, boundary=None, flat=False, tolerance=1e-10,
                          depth=None):
    complex = pydec.SimplicialComplex((vertices, simplices))

    complex.construct_hodge()

    L = complex[0].d.T @ complex[1].star @ complex[0].d

    if depth is None:
        depth = len(complex.vertices) - 2

    if not flat:
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(A=L, M=complex[0].star,
                                                              k=depth,
                                                              sigma=0,
                                                              mode='normal',
                                                              tol=tolerance,
                                                              ncv=min(4 * depth, L.shape[0] - 1)
                                                              )
    else:
        if boundary is None:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(A=L,
                                                                  k=depth,
                                                                  sigma=0,
                                                                  mode='normal',
                                                                  tol=tolerance,
                                                                  ncv=min(4 * depth, L.shape[0] - 1)
                                                                  )
        else:
            interior_vertices = np.full(L.shape[0], True)
            interior_vertices[boundary] = False

            # Extract interior submatrix
            L_ii = L[np.ix_(interior_vertices, interior_vertices)]

            # Recompute eigenvectors with Dirichlet zero boundary condition
            eigenvalues, eigvecs_int = scipy.sparse.linalg.eigsh(A=L_ii,
                                                                 k=depth,
                                                                 sigma=0,
                                                                 mode='normal',
                                                                 tol=tolerance,
                                                                 ncv=min(4 * depth, L.shape[0] - 1)
                                                                 )

            # Assemble full eigenvectors with zero at boundary
            eigenvectors = np.zeros((L.shape[0], depth))
            eigenvectors[interior_vertices, :] = eigvecs_int

    eigenvalues[abs(eigenvalues) < tolerance] = 0

    vol = np.sum(complex[-1].primal_volume)

    return eigenvalues, eigenvectors, vol, complex
