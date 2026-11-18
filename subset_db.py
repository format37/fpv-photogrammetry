"""Copy a COLMAP database keeping only some images (with their keypoints, descriptors and matches), so that
image_registrator / point_triangulator see just those (e.g. re-register one pass without pulling in rejected clips).

Kept: images registered in --model, plus names starting with any --prefix, minus names in --exclude.
usage: subset_db.py database.db OUT.db --model sparse_fix/noc19 --prefix c19_ c15_t06 --exclude c19_t157.083.jpg
"""
import argparse, shutil, sqlite3, pycolmap

ap = argparse.ArgumentParser(); ap.add_argument("db"); ap.add_argument("out")
ap.add_argument("--model"); ap.add_argument("--prefix", nargs="*", default=[]); ap.add_argument("--exclude", nargs="*", default=[])
a = ap.parse_args()
keep = {im.name for im in pycolmap.Reconstruction(a.model).images.values()} if a.model else set()
shutil.copyfile(a.db, a.out); db = sqlite3.connect(a.out)
rows = db.execute("select image_id, name from images").fetchall()
want = {i for i, n in rows if (n in keep or n.startswith(tuple(a.prefix))) and n not in a.exclude} if a.prefix else \
       {i for i, n in rows if n in keep and n not in a.exclude}
drop = ",".join(str(i) for i, _ in rows if i not in want) or "-1"
for t in ("images", "keypoints", "descriptors"): db.execute(f"delete from {t} where image_id in ({drop})")
for t in ("matches", "two_view_geometries"):                       # pair_id = id1 * 2147483647 + id2
    db.execute(f"delete from {t} where (pair_id / 2147483647) in ({drop}) or (pair_id % 2147483647) in ({drop})")
db.commit(); db.execute("vacuum")
print(f"{a.out}: kept {len(want)} of {len(rows)} images; to register (not in the model): "
      f"{len([1 for i, n in rows if i in want and n not in keep])}")
