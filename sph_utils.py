import numpy as np


def spherical_tangent_basis(points, r=1):
    # points should be of shape (N, 3) and assumed on the unit sphere.
    x, y, z = points[:, 0] / r, points[:, 1] / r, points[:, 2] / r
    lat = np.arcsin(z)  # latitude in radians
    lon = np.arctan2(y, x)  # longitude in radians

    # East vector:
    east = np.column_stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)])

    # North vector:
    north = np.column_stack([
        -np.cos(lon) * np.sin(lat),
        -np.sin(lon) * np.sin(lat),
        np.cos(lat)
    ])

    # They should already be normalized.
    return north, east


def xyz_to_latlon(points, r=1):
    x, y, z = points[:, 0] / r, points[:, 1] / r, points[:, 2] / r
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))
    lon = (lon + 360) % 360
    return lat, lon


def project_to_latlon(points, tangents, r=1):
    lat, lon = xyz_to_latlon(points, r)

    b_north, b_east = spherical_tangent_basis(points, r)

    us = np.sum(b_east * tangents, axis=1)
    vs = np.sum(b_north * tangents, axis=1)

    return lat, lon, us, vs


def project_to_xyz(points, tangents_u, tangents_v, r=1):
    basis_northing, basis_easting = spherical_tangent_basis(points)
    projected_vecs = tangents_u[:, None] * basis_easting + tangents_v[:, None] * basis_northing
    return projected_vecs


def conv_lat_lon(lats, lons, r=1):
    lat = np.radians(lats)
    lon = np.radians(lons)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.vstack((x, y, z)).T
