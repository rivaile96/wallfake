import cv2
import os

# Tentukan path sumber gambar dan tujuan dataset
SUMBER_DIR = 'sumber_fake'
DATA_DIR = 'dataset'
FAKE_DIR = os.path.join(DATA_DIR, 'fake')

# Buat folder tujuan jika belum ada
if not os.path.exists(FAKE_DIR):
    os.makedirs(FAKE_DIR)

# Muat classifier wajah
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 0
print(f"Memulai pemrosesan gambar dari folder '{SUMBER_DIR}'...")

# Loop setiap file di dalam folder sumber
for filename in os.listdir(SUMBER_DIR):
    # Gabungkan path folder dan nama file
    img_path = os.path.join(SUMBER_DIR, filename)

    # Baca gambar
    img = cv2.imread(img_path)
    if img is None:
        print(f"Gagal membaca file: {filename}")
        continue

    # Ubah ke grayscale untuk deteksi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Ambil wajah pertama yang terdeteksi (asumsi hanya ada satu wajah)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        wajah_terpotong = img[y:y+h, x:x+w]

        # Resize dan simpan
        try:
            wajah_terpotong_resized = cv2.resize(wajah_terpotong, (224, 224))
            save_path = os.path.join(FAKE_DIR, f'fake_{count}.jpg')
            cv2.imwrite(save_path, wajah_terpotong_resized)
            print(f"Wajah terdeteksi dan disimpan ke {save_path}")
            count += 1
        except Exception as e:
            print(f"Error saat memproses {filename}: {e}")
    else:
        print(f"Tidak ada wajah terdeteksi di {filename}")

print(f"\nSelesai! Sebanyak {count} gambar wajah palsu berhasil diproses.")