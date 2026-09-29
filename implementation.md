# Rencana implementasi kompetisi JOINTS x INSPIRE 2026

## Scan dan penyesuaian

Proyek berisi `main.ipynb` dan enam CSV sumber yang disebut prompt (termasuk sample submission). CSV memiliki kolom sesuai konteks: movies memakai `original_title`, `age_rating`, `genre`, `producer`, `director`, `writer`, `casts`; histori/train memuat tiket, show, okupansi, pasangan film-klaster dan tanggal. `.gitignore` mengabaikan `/dataset` dan file `.env`, sehingga sumber kompetisi dan kredensial tidak ikut commit. Notebook memiliki `compute_history_stats()` yang dipakai sebagai sumber logika statistik D1-D3.

**Koreksi ruang lingkup:** movies.csv tidak memiliki tanggal rilis, overview, popularity, runtime, atau original_language. Tahap 1 mengisi kolom catalog yang tersedia dan menyimpan kolom tambahan di file metadata eksternal. Split film berdasarkan tanggal rilis terbaru tidak dapat dibuat persis dari CSV karena tanggal rilis film train tidak tersedia; validasi memakai holdout tujuh observasi terakhir per pasangan sebagai pendekatan temporal.

## Jobdesk & progress

### Tahap 1 — Lengkapi data film yang hilang (prioritas tertinggi)
- [x] Ambil metadata TMDB untuk 20 judul test yang tidak ada di `movies.csv` (20/20 title format cocok).
- [x] Simpan metadata ber-schema kompatibel plus data tambahan di `external_data/missing_movies_metadata.csv`.
- [x] Dokumentasikan sumber, endpoint, tanggal, daftar title melalui audit di `external_data/README.md` dan `external_data/acquisition_audit.csv`.
- [x] Gabungkan katalog dan validasi: 0 title test tanpa genre / 0 tanpa age_rating setelah merge.
- [x] Siapkan alur akuisisi TMDB v3 memakai `TMDB_API_RAT` (bearer token) atau `TMDB_API_KEY`, dengan strict title matching dan audit hasil.

### Tahap 2 — Bangun tabel fitur terpadu
- [x] Siapkan skrip reproducible untuk statistik D1-D3, fitur klaster, kalender target, genre/rating, dan jarak bertanda ke hari libur.
- [x] Hasilkan tabel fitur `data/feature_table.parquet` berisi 72.611 baris, termasuk genre/rating eksternal. CSV fallback juga dibuat.

### Tahap 3 — Desain validasi time-based
- [x] Implementasikan MASE lokal dengan scale `max(1, mean D1-D3)`.
- [x] Siapkan validasi temporal holdout D4-D10 per pasangan, memakai tiga observasi sebelumnya untuk D1-D3.
- [x] Ukur skor validasi temporal: 35.728 baris; tidak ada pasangan berskala 1 pada subset validasi akhir. Ada 23 pasangan historis berskala 1, tetapi hanya 1 yang overlap dengan pasangan forecast test; label test tidak tersedia untuk menghitung MASE pada pasangan tersebut.

### Tahap 4 — Baseline model
- [x] Implementasikan baseline scale × multiplier kalender/libur × decay genre/hari.
- [x] Baseline validasi temporal: MASE 0,4283.

### Tahap 5 — Model utama
- [x] Siapkan implementasi gradient boosting L1 pada target `total_ticket/scale`, seed tetap.
- [x] Model LightGBM L1, seed 2026, validasi temporal setelah metadata: MASE 0,4528; baseline masih lebih baik (0,4283). Agregat klaster validasi dihitung dari data sebelum periode holdout untuk mencegah kebocoran.
- [x] Tambahkan genre/rating metadata ke fitur; validasi LightGBM membaik dari MASE 0,4692 menjadi 0,4528, namun baseline tetap lebih baik.

### Tahap 6 — Reprodusibilitas & dokumentasi
- [x] Sediakan skrip pipeline dan dokumentasi dependency/urutan eksekusi.
- [x] Rapikan notebook dengan alur data acquisition, preprocessing/QA, EDA, feature engineering, modeling, evaluation.
- [x] Restart Kernel & Run All dari kondisi bersih; dependency terpasang dari `requirements.txt`, pipeline notebook berhasil sampai akhir.

## Blockers

- **Metadata eksternal:** selesai. Kredensial TMDB berhasil dipakai. `.env` telah ditambahkan ke `.gitignore`; skrip tidak menampilkan atau menulis credential.
- **Validasi tanggal rilis:** sumber film train tidak menyediakan tanggal rilis, jadi split sesuai instruksi “rilis paling akhir” tidak dapat dibuat. Pipeline memakai tujuh observasi terbaru per pasangan sebagai validasi temporal. Dari 23 pasangan historis berskala 1, hanya satu masuk ke pasangan target test dan label test tidak tersedia untuk menghitung skornya.
- **Eksekusi notebook final:** selesai. Cell dependency memasang paket versi terpin; `data/feature_table.parquet` tersedia. Ada peringatan joblib/WMIC saat estimasi core, tetapi notebook dan pipeline selesai.

## Catatan asal skor MASE

Skor 0,4283 (baseline) dan 0,4528 (LightGBM L1) dihitung saat menjalankan `scripts/build_pipeline.py`, bukan dari `main.ipynb`. Skrip memakai holdout tujuh observasi terakhir per pasangan train sebagai D4-D10, dengan tiga observasi sebelumnya untuk skala D1-D3. Ini evaluasi terpisah dari notebook EDA dan belum merupakan hasil submission/Kaggle.
