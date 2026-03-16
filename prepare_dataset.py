import os
import shutil

# ====== EDIT THESE PATHS IF NEEDED ======

PROTOCOL_FILE = r"..\new_audio_model\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"

AUDIO_FOLDER = r"..\new_audio_model\LA\ASVspoof2019_LA_train\flac"


# output folders for keras training
REAL_OUT = r"dataset\real"
FAKE_OUT = r"dataset\fake"

# how many files you want (keep small for test)
LIMIT_REAL = 5000
LIMIT_FAKE = 5000

# ========================================

os.makedirs(REAL_OUT, exist_ok=True)
os.makedirs(FAKE_OUT, exist_ok=True)

real_count = 0
fake_count = 0

with open(PROTOCOL_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()

        file_id = parts[1]   # LA_T_xxxxxx
        label = parts[-1]    # bonafide or spoof

        src = os.path.join(AUDIO_FOLDER, file_id + ".flac")

        if label == "bonafide" and real_count < LIMIT_REAL:
            dst = os.path.join(REAL_OUT, file_id + ".flac")
            if os.path.exists(src):
                shutil.copy(src, dst)
                real_count += 1
                print("Copied REAL:", file_id)

        elif label == "spoof" and fake_count < LIMIT_FAKE:
            dst = os.path.join(FAKE_OUT, file_id + ".flac")
            if os.path.exists(src):
                shutil.copy(src, dst)
                fake_count += 1
                print("Copied FAKE:", file_id)

        if real_count >= LIMIT_REAL and fake_count >= LIMIT_FAKE:
            break

print("\nDone!")
print("Real files:", real_count)
print("Fake files:", fake_count)
