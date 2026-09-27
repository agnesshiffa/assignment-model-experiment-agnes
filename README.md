# AI Model Experiment & Evaluation — Sentiment Analysis Ulasan Pelanggan

Eksperimen perbandingan dua pendekatan AI untuk klasifikasi sentimen ulasan pelanggan e-commerce (positif/negatif):

1. **Model klasik** — TF-IDF + Logistic Regression (Scikit-learn), dilatih sendiri.
2. **LLM API** — Gemini (`gemini-3.8-flash`) dengan prompting zero-shot & few-shot, tanpa training.

Kedua pendekatan dievaluasi pada **test set yang sama**.


---

## 1. Problem Statement

**Konteks.** Tim produk e-commerce ingin fitur otomatis untuk mengklasifikasikan sentimen ulasan pelanggan di halaman produk. Sebelum memilih pendekatan untuk produksi, dilakukan eksperimen awal untuk membandingkan model klasik dan LLM API.

**Objective.**
- Membangun classifier yang menerima teks ulasan (Bahasa Indonesia) dan mengeluarkan label sentimen.
- Membandingkan kedua pendekatan secara kuantitatif (Accuracy, Precision, Recall, F1-Score, confusion matrix) dan kualitatif (effort, kecepatan, biaya, keterbatasan).
- Memberikan rekomendasi technical approach berbasis data.

**Target / label.** Kolom `sentiment`, klasifikasi biner:
- `positif` — pelanggan puas (kualitas, pengiriman, pelayanan baik)
- `negatif` — pelanggan kecewa (barang rusak/tidak sesuai, pengiriman lambat, pelayanan buruk)

**Batasan & asumsi.**
- Hanya dua kelas, tanpa kelas netral; label di dataset dianggap *ground truth*.
- Fitur yang dipakai hanya `review_text`.
- Dataset asli tidak diubah; split 80/20 stratified, `random_state=42`.
- Metrik utama: **F1-macro** (kelas sedikit tidak seimbang, kesalahan dua arah sama-sama merugikan), dengan perhatian khusus pada **recall negatif** (keluhan yang lolos).
- Training Deep Learning dari nol di luar scope.

## 2. Dataset

`data/customer_reviews_sentiment.csv` — 200 ulasan, 4 kolom: `review_id`, `product_name`, `review_text`, `sentiment`.

| Karakteristik | Nilai |
|---|---|
| Jumlah baris | 200 |
| Distribusi label | 110 positif (55%), 90 negatif (45%) |
| Missing value | 0 |
| Panjang ulasan | 43–71 karakter (rata-rata 55) |
| **Teks ulasan unik** | **40** (160 baris adalah duplikat teks dengan `product_name` berbeda) |
| Teks sama dengan label berbeda | 0 |

> File dataset dibagikan dalam format Excel (`.xlsx`) dan disimpan ulang ke CSV tanpa perubahan isi (diverifikasi identik), sesuai struktur yang diminta.

**Temuan penting:** karena hanya ada 40 teks unik, split acak membuat **39 dari 40 ulasan test (98%) teksnya juga ada di train set**. Ini dianalisis sebagai limitation dan diuji dengan GroupKFold (lihat bawah).

## 3. Ringkasan Eksperimen

### Split data
- 80/20 stratified → **160 train, 40 test** (22 positif, 18 negatif), `random_state=42`.
- Test set disimpan di `results/test_set.csv` dan dipakai oleh kedua pendekatan.

### Pendekatan 1 — Model Klasik
- **Preprocessing:** `TfidfVectorizer(lowercase=True, ngram_range=(1,2), sublinear_tf=True)`, tanpa stopword removal (kata "tidak" membawa sinyal negasi). Bigram menangkap frasa seperti "tidak sesuai".
- **Model:** `LogisticRegression(C=1.0, max_iter=1000)`.
- **Uji tambahan:** GroupKFold 5 fold berdasarkan teks ulasan untuk mengukur performa pada teks yang benar-benar belum pernah dilihat.

### Pendekatan 2 — LLM API (Gemini)
- **Model:** `gemini-3.8-flash`. Awalnya memakai `gemini-3.6-flash` (model yang dipakai di kelas), tetapi kuota harian free tier-nya (20 request/hari/model) habis; `gemini-2.5-flash` sudah tidak tersedia untuk pengguna baru, sehingga dipakai `gemini-3.8-flash` sesuai rekomendasi pesan error API.
- **Prompt:** system instruction (peran, definisi label, aturan ulasan campuran, klasifikasi tiap ulasan secara independen) + dua varian:
  - *Zero-shot* — tanpa contoh.
  - *Few-shot* (prompt utama) — 4 contoh (2 positif, 2 negatif) yang **ditulis sendiri, bukan dari dataset**, agar tidak bocor ke test set; mencakup slang ("mantul", "zonk") dan negasi ("tidak mengecewakan").
- **Parameter:**

| Parameter | Nilai | Alasan |
|---|---|---|
| `temperature` | 0.0 | Klasifikasi butuh output deterministik & konsisten |
| `response_mime_type` | `application/json` | Output dipaksa berupa JSON array agar parsing andal |
| `BATCH_SIZE` | 20 | 20 ulasan per request untuk menghemat kuota & token |

- **Batch prompting:** percobaan awal dengan satu ulasan per request (80 request) terkena error `429 RESOURCE_EXHAUSTED` (kuota free tier habis). Solusinya, 20 ulasan dikirim sekaligus dalam satu request sebagai daftar bernomor, dan model mengembalikan JSON array label sesuai urutan. Seluruh eksperimen cukup 4 request.

- **Normalisasi output:** lowercase, buang tanda baca, petakan ke `positif`/`negatif`; selain itu → `invalid` (dihitung salah).
- Hasil di-cache di `results/llm_predictions_<mode>.csv`.

## 4. Hasil Evaluasi & Perbandingan

### Test set utama (40 ulasan, sama untuk semua pendekatan)

| Pendekatan | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Recall negatif | Salah prediksi |
|---|---|---|---|---|---|---|
| Model Klasik (TF-IDF + LogReg) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| LLM Gemini (zero-shot) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| LLM Gemini (few-shot) | 0.5000 | 1.0000 | 0.5051 | 0.6696 | 0.5556 | 20 |

**Confusion matrix model klasik** (baris = aktual, kolom = prediksi):

|  | pred negatif | pred positif |
|---|---|---|
| **negatif** | 18 | 0 |
| **positif** | 0 | 22 |

Confusion matrix Gemini dan grafik perbandingan: `documentation/model_comparison_summary.png` (dibuat otomatis oleh notebook).

### Metrik operasional

| Pendekatan | Butuh training | Latensi per ulasan | Biaya per 1.000 ulasan* |
|---|---|---|---|
| Model Klasik | Ya (±5 ms) | ±0,01 ms | ~USD 0 |
| LLM Gemini (zero-shot) | Tidak | ±234 ms (batch 20 ulasan/request) | ~USD 0.0194 |
| LLM Gemini (few-shot) | Tidak | ±679 ms (batch 20 ulasan/request) | ~USD 0.0125 |

\*Estimasi biaya memakai asumsi harga paid tier model Gemini Flash (input USD 0,30 / output USD 2,50 per 1 juta token) — sesuaikan dengan harga model yang dipakai di halaman pricing Google AI.

### Uji generalisasi model klasik (GroupKFold, teks belum pernah dilihat)

| Metrik | Nilai |
|---|---|
| Accuracy rata-rata | **0,605 ± 0,120** (per fold: 0,70 / 0,675 / 0,60 / 0,375 / 0,675) |
| F1-macro rata-rata | 0,575 ± 0,154 |
| Teks unik salah prediksi | 15 dari 40 (11 di antaranya negatif → positif) |

**Interpretasi:** skor sempurna model klasik di test set utama sebagian besar berasal dari hafalan kalimat identik. Pada teks baru, akurasinya turun ke ±0,60 — hanya sedikit di atas menebak kelas mayoritas (0,55) — dan kesalahannya didominasi keluhan yang lolos sebagai positif.

## 5. Analisis Trade-off dan Limitation

### Performa
- Test set utama: model klasik 1,00; Gemini zero-shot 1.00 dan few-shot 0.50. Skor model klasik *optimistic* karena duplikasi train/test.
- Teks baru: model klasik turun ke ±0,60, sedangkan LLM tidak terpengaruh karena tidak dilatih dengan data ini — skornya di test set sudah merupakan estimasi jujur untuk ulasan baru.
- Untuk ulasan yang mirip data training, keduanya setara; untuk kosakata/gaya bahasa baru, LLM lebih robust.

### Effort, kecepatan, biaya

| Aspek | Model Klasik | LLM API |
|---|---|---|
| Data berlabel | Wajib, kualitas & variasi menentukan hasil | Tidak perlu / beberapa contoh |
| Proses | Preprocessing, training, tuning, retraining | Prompt design, parsing, error handling |
| Kecepatan inferensi | Mikrodetik, lokal di CPU | Ratusan ms–detik per ulasan + rate limit |
| Biaya | ~0 per inferensi; biaya di labeling & maintenance | Per token, linear dengan volume; few-shot > zero-shot |
| Tambah kelas baru | Labeling & training ulang | Ubah prompt |

### Limitation
**Model klasik:** hanya kenal kata yang ada di training (OOV tak berbobot); negasi hanya tertangkap sebagian lewat bigram (contoh: "tidak mengecewakan" dan "tidak ada cacat" salah diprediksi negatif pada uji teks baru); dataset kecil & repetitif → risiko overfit ke template kalimat; perlu retraining berkala.

**LLM API:** saat eksperimen juga terkena error `503 UNAVAILABLE` (server model sedang penuh) sehingga sebagian request harus diulang — pada run final, 20 ulasan few-shot tetap gagal setelah 6 kali retry sehingga tercatat sebagai prediksi salah (invalid); skor few-shot karenanya lebih rendah dari kemampuan sebenarnya; bergantung pada vendor (downtime, rate limit, deprecation versi, perubahan harga) — di eksperimen ini pemanggilan per ulasan langsung terkena error 429 kuota free tier sehingga harus diganti batch prompting; output perlu validasi format; reproducibility jangka panjang tidak terjamin antar versi model; teks ulasan dikirim ke server eksternal (privasi data); latensi & biaya lebih tinggi.

**Eksperimen:** test set hanya 40 ulasan (26 teks unik) → 1 prediksi = 2,5 poin akurasi; dataset tidak memuat ulasan ambigu/sarkastik/campuran yang umum di produksi.

## 6. Rekomendasi Technical Approach

**Pendekatan hybrid: Gemini few-shot sebagai classifier awal & labeler, model klasik sebagai classifier utama saat skala besar.**

1. **Fase awal:** pakai Gemini few-shot. Data berlabel tim baru 40 kalimat unik — terlalu sedikit untuk model klasik yang andal (akurasi ±0,60 pada teks baru), sedangkan LLM sudah memberi performa tinggi tanpa training.
2. **Kumpulkan data:** simpan prediksi LLM pada ulasan riil sebagai *silver label*, review manusia pada sampel (terutama prediksi negatif & kasus ambigu).
3. **Fase skala:** latih ulang model klasik dengan data yang lebih besar & beragam; jika performanya pada data baru setara LLM, jadikan classifier utama (latensi jauh lebih rendah, biaya ~0).
4. **Routing:** ulasan dengan probabilitas model klasik rendah (mis. 0,4–0,6) diteruskan ke LLM, sehingga biaya API hanya untuk kasus sulit.

**Alasan:** model klasik saja berisiko di produksi karena skor tingginya bergantung pada duplikasi data; LLM saja akurat dan fleksibel tetapi biaya, latensi, rate limit, dan ketergantungan vendor jadi masalah pada volume besar & kebutuhan real-time. Hybrid mengambil kekuatan keduanya.

## 7. Cara Menjalankan

```bash
# 1. Clone repository
git clone https://github.com/<username>/assignment-model-experiment-<nama-peserta>.git
cd assignment-model-experiment-<nama-peserta>

# 2. Buat virtual environment & install dependency
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Set API key Gemini
cp .env.example .env             # Windows: copy .env.example .env
# lalu edit .env dan isi GEMINI_API_KEY (dapatkan di https://aistudio.google.com/apikey)

# 4. Jalankan notebook
jupyter notebook notebook/experiment_notebook.ipynb
# lalu Run All
```

Catatan:
- Pemanggilan Gemini hanya 4 request (batch 20 ulasan × 2 batch × 2 varian prompt), biasanya selesai < 1 menit. Kalau terkena error 429 (kuota habis), notebook otomatis menunggu lalu mencoba lagi.
- Hasil LLM di-cache di `results/`; run berikutnya tidak memanggil API lagi. Hapus file `results/llm_predictions_*.csv` untuk memanggil ulang.
- File `.env` sudah masuk `.gitignore` — **jangan commit API key**.

## 8. Struktur Project

```
model-experiment-assignment/
├── data/
│   └── customer_reviews_sentiment.csv     # dataset asli (tidak diubah)
├── notebook/
│   └── experiment_notebook.ipynb          # eksperimen lengkap
├── documentation/
│   └── model_comparison_summary.png       # confusion matrix + grafik metrik (dibuat notebook)
├── results/                               # dibuat notebook
│   ├── test_set.csv
│   ├── predictions_classic.csv
│   ├── llm_predictions_zero_shot.csv
│   ├── llm_predictions_few_shot.csv
│   ├── predictions_all.csv
│   └── model_comparison_summary.csv
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```
