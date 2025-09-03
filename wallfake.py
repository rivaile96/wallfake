# Import library yang dibutuhkan
import cv2

# Muat file contekan (classifier) untuk deteksi wajah
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Nyalakan webcam (angka 0 berarti webcam utama)
cap = cv2.VideoCapture(0)

print("Kamera menyala. Tekan tombol 'q' untuk keluar.")

# Loop untuk membaca video frame per frame
while True:
    # Baca satu frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Gagal mengambil frame dari kamera.")
        break

    # Ubah gambar menjadi abu-abu (lebih mudah untuk deteksi)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Proses deteksi wajah
    # Angka 1.1 dan 4 adalah parameter, bisa diubah untuk sensitivitas
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Gambar kotak di setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:
        # Parameter: (gambar, titik_awal, titik_akhir, warna_kotak, ketebalan)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # Tampilkan hasilnya di jendela baru
    cv2.imshow('Deteksi Wajah Real-time', frame)

    # Cek jika pengguna menekan tombol 'q', lalu keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Matikan kamera dan tutup semua jendela
print("Mematikan kamera...")
cap.release()
cv2.destroyAllWindows()