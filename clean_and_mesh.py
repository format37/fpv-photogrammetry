"""Crop the fused dense cloud to the flown area, clean it, mesh it (screened Poisson),
trim low-density Poisson surface, report stats.

usage: clean_and_mesh.py FUSED.ply SPARSE_MODEL OUTDIR [--no-crop]
  --no-crop: the cloud is already cropped (crop_focus.py); only the ground plane, outlier filter and mesh
"""
import argparse, numpy as np, open3d as o3d, pycolmap

ap = argparse.ArgumentParser(); ap.add_argument("fused"); ap.add_argument("model"); ap.add_argument("outdir")
ap.add_argument("--no-crop", action="store_true"); args = ap.parse_args()
fused, model, outdir = args.fused, args.model, args.outdir
rec = pycolmap.Reconstruction(model)
C = np.array([im.projection_center() for im in rec.images.values() if im.has_pose])

pcd = o3d.io.read_point_cloud(fused)
print(f"fused: {len(pcd.points):,} points")

# ground plane from the dense cloud itself, normal pointing to the cameras
P = np.asarray(pcd.points)
d = np.linalg.norm(P - C.mean(0), axis=1); med = np.median(d)
near = pcd.select_by_index(np.flatnonzero(d < 2 * med))
(a, b, c, e), _ = near.segment_plane(0.02 * med, 3, 5000)
n = np.array([a, b, c]); s = np.linalg.norm(n); n, e = n / s, e / s
if C.mean(0) @ n + e < 0: n, e = -n, -e
height = np.median(C @ n + e)                                   # flying height above ground

# keep points whose ground footprint is within ~1.5 flying heights of the camera track footprint
foot_c = C - np.outer(C @ n + e, n)
foot_p = P - np.outer(P @ n + e, n)
from scipy.spatial import cKDTree
radius = 1.5 * height
dist, _ = cKDTree(foot_c).query(foot_p, k=1, distance_upper_bound=radius, workers=-1)
keep = np.isfinite(dist)
# and drop anything far above/below the ground (sky noise, mismatches)
hp = P @ n + e
keep &= (hp > -0.3 * height) & (hp < 0.6 * height)
if args.no_crop: keep[:] = True
roi = pcd.select_by_index(np.flatnonzero(keep))
print(f"ROI crop (footprint radius {radius:.3f}, flying height {height:.3f} model units): {len(roi.points):,} points")

roi, _ = roi.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"after statistical outlier removal: {len(roi.points):,}")
o3d.io.write_point_cloud(f"{outdir}/dense_roi.ply", roi)

# orient normals towards the cameras (fusion normals are already camera-facing; make sure)
if not roi.has_normals():
    roi.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
roi.orient_normals_to_align_with_direction(n)

mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(roi, depth=12, scale=1.05, linear_fit=False)
dens = np.asarray(dens)
mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.06))   # trim hallucinated surface
mesh = mesh.crop(roi.get_axis_aligned_bounding_box())
mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
tri_clusters, n_tri, _ = mesh.cluster_connected_triangles()
n_tri = np.asarray(n_tri); small = n_tri[np.asarray(tri_clusters)] < 0.01 * n_tri.max()
mesh.remove_triangles_by_mask(small); mesh.remove_unreferenced_vertices()
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh(f"{outdir}/mesh_poisson.ply", mesh)
print(f"poisson mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles -> {outdir}/mesh_poisson.ply")
np.savetxt(f"{outdir}/ground_plane.txt", np.r_[n, e, height], header="nx ny nz e height")

# light copy for weak PCs / Blender
lite = mesh.simplify_quadric_decimation(1_000_000); lite.remove_unreferenced_vertices(); lite.compute_vertex_normals()
o3d.io.write_triangle_mesh(f"{outdir}/mesh_poisson_1M.ply", lite)
print(f"light mesh: {len(lite.triangles):,} triangles -> {outdir}/mesh_poisson_1M.ply")
