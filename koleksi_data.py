import cv2
import os

# --- BAGIAN BARU: PERSIAPAN FOLDER ---
# Tentukan path untuk menyimpan dataset
DATA_DIR = 'dataset'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Buat sub-folder untuk wajah asli
REAL_DIR = os.path.join(DATA_DIR, 'real')
if not os.path.exists(REAL_DIR):
    os.makedirs(REAL_DIR)
# ------------------------------------

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# --- BAGIAN BARU: HITUNGAN GAMBAR ---
count = 0
# ------------------------------------

print("Kamera menyala. Tekan SPASI untuk mengambil gambar, 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        wajah_terpotong = frame[y:y+h, x:x+w]

        try:
            wajah_terpotong_resized = cv2.resize(wajah_terpotong, (224, 224))
            cv2.imshow('Wajah Hasil Ekstraksi', wajah_terpotong_resized)
        except:
            continue # Lanjut ke frame berikutnya jika ada error

    cv2.imshow('Wallfake - Koleksi Data', frame)

    # --- BAGIAN BARU: LOGIKA MENYIMPAN GAMBAR ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32: # 32 adalah kode ASCII untuk tombol SPASI
        # Pastikan ada wajah terdeteksi sebelum menyimpan
        if 'wajah_terpotong_resized' in locals() and wajah_terpotong_resized.size != 0:
            file_path = os.path.join(REAL_DIR, f'real_{count}.jpg')
            cv2.imwrite(file_path, wajah_terpotong_resized)
            print(f"Gambar ke-{count} disimpan di {file_path}")
            count += 1
    # ------------------------------------------

print("Mematikan program...")
cap.release()
cv2.destroyAllWindows()