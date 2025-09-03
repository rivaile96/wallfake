import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

print("Kamera menyala. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Tetap gambar kotak di video utama
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # --- INI BAGIAN BARUNYA ---
        # Potong wajah dari frame asli (berwarna)
        wajah_terpotong = frame[y:y+h, x:x+w]

        # Tampilkan wajah yang terpotong di jendela kedua
        # Pakai try-except untuk cegah error saat wajah di pinggir layar
        try:
            # Resize gambar wajah agar ukurannya konsisten
            wajah_terpotong_resized = cv2.resize(wajah_terpotong, (224, 224))
            cv2.imshow('Wajah Hasil Ekstraksi', wajah_terpotong_resized)
        except:
            pass
        # --- AKHIR BAGIAN BARU ---

    # Tampilkan video utama
    cv2.imshow('Wallfake - Video Utama', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Mematikan program...")
cap.release()
cv2.destroyAllWindows()