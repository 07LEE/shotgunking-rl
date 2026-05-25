import os
import shutil

src_dir = "data/templates"
mapping = {
    "candidate_r7_c3_aw601_h45.png": "king_1_0.png",
    "candidate_r1_c2_aw315_h19.png": "pawn_2_0.png",
    "candidate_r1_c4_aw315_h19.png": "pawn_2_1.png",
    "candidate_r1_c6_aw315_h19.png": "pawn_2_2.png",
    "candidate_r1_c1_aw1408_h45.png": "pawn_2_3.png",
    "candidate_r1_c3_aw1532_h45.png": "pawn_2_4.png",
    "candidate_r0_c1_aw753_h34.png": "knight_3_0.png",
    "candidate_r0_c6_aw1552_h45.png": "knight_3_1.png",
    "candidate_r0_c2_aw1400_h45.png": "bishop_4_0.png",
    "candidate_r0_c5_aw330_h20.png": "bishop_4_1.png",
    "candidate_r0_c3_aw377_h24.png": "queen_6_0.png"
}

for src, dst in mapping.items():
    src_path = os.path.join(src_dir, src)
    dst_path = os.path.join(src_dir, dst)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"Copied: {src} -> {dst}")

# Clean up candidates
for filename in os.listdir(src_dir):
    if filename.startswith("candidate_"):
        os.remove(os.path.join(src_dir, filename))
        print(f"Removed: {filename}")
