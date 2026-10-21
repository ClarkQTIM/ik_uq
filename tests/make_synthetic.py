"""
Build a small synthetic dataset with the same columns and coding as the study CSV,
for testing the pipeline without real data. Images are random noise.

    python tests/make_synthetic.py OUT_DIR
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

PACS = "/pacs/OSHU/Corneal_Ulcer_Image_Sets"
rng = np.random.default_rng(0)
out = sys.argv[1]
img_dir = os.path.join(out, "images")
os.makedirs(img_dir, exist_ok=True)

rows = []
pid = 0
spec = [  # study, site, participants, images each, etiology, has tests
    ("MADURAI photos", "Madurai", 40, 3, "mixed", True),
    ("MUTT1-Pictures", "Madurai", 10, 3, "fungal", True),
    ("MUTT1-Pictures", "Coimbatore", 15, 3, "fungal", False),
    ("MUTT2-Pictures", "Lumbini", 15, 2, "fungal", False),
    ("SCUT photos", "Madurai", 8, 3, "bacterial", True),
    ("SCUT photos", "Dartmouth", 15, 3, "bacterial", False),
    ("CLAIR I photos", "Madurai", 4, 1, "bacterial", True),
    ("CLAIR II photos", "Madurai", 6, 1, "fungal", True),
]
visit_codes = ["E", "3W", "3M"]
for study, site, n_p, n_img, etio, tests in spec:
    for _ in range(n_p):
        pid += 1
        if etio == "mixed":
            p_bac = float(rng.choice([0.00001, 0.9999, rng.uniform(0.1, 0.9)], p=[0.5, 0.35, 0.15]))
        else:
            p_bac = 0.9996 if etio == "bacterial" else 0.0016
        for k in range(n_img):
            name = (f"M{pid:04d} clinical_photo_{k + 1}.JPG" if study == "MADURAI photos"
                    else f"{pid:04d}{'R'} - {visit_codes[k % 3]} (1).JPG")
            fn = f"img_{len(rows):05d}.jpg"
            Image.fromarray(rng.integers(0, 255, (64, 80, 3), dtype=np.uint8)).save(os.path.join(img_dir, fn))
            bin_bac = int(p_bac > 0.5)
            r = dict(study=study, study_site=site, subject=f"S{pid}", pID=f"P{pid}",
                     file_path=os.path.join(PACS, "images", fn), image_name=name,
                     bac_smear=-1, fung_smear=-1, bac_cult_positive=-1, fung_cult_positive=-1,
                     MDS_fungus=-1, MDS_bacteria=-1,
                     fung_post_prob=-1.0, bac_post_prob=-1.0,
                     fung_LCA_bin=1 - bin_bac, bac_LCA_bin=bin_bac)
            if tests:
                r.update(bac_smear=bin_bac, fung_smear=1 - bin_bac,
                         bac_post_prob=p_bac, fung_post_prob=1 - p_bac)
            rows.append(r)
# rows with no label (both bins -1) and one row without a photo
for i in range(5):
    pid += 1
    rows.append(dict(rows[0], pID=f"P{pid}", subject=f"S{pid}", fung_LCA_bin=-1, bac_LCA_bin=-1,
                     fung_post_prob=-1.0, bac_post_prob=-1.0))
rows.append(dict(rows[1], file_path=np.nan))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(out, "synthetic.csv"), index=False)
print(f"{len(df)} rows, {df.pID.nunique()} participants -> {out}/synthetic.csv")
