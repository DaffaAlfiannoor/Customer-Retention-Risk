# Dataset Hasil Data Preprocessing

## Tujuan

Dataset ini adalah hasil dari Section 6 Data Preprocessing dan Feature
Engineering. Dataset disiapkan untuk digunakan oleh tim AI pada tahap
modeling berikutnya.

## Sumber

Dataset berasal dari data dummy atau sintetis untuk prototype Customer
Retention Risk Gymnasyum. Dataset bukan data customer nyata.

## Grain

Satu baris merepresentasikan satu customer pada prediction cutoff.

## Periode

- Observation window: 6 Januari 2025 sampai 31 Maret 2025
- Prediction cutoff: 31 Maret 2025
- Prediction horizon: 1 April 2025 sampai 27 April 2025
- Panjang horizon: 27 hari kalender

## Isi dataset

- Jumlah customer: 3,000
- Jumlah fitur: 20
- Jumlah kolom total: 22

## Target

Kolom target adalah `needs_follow_up`.

- `1`: future engagement tier adalah At Risk atau Need Attention
- `0`: future engagement tier adalah Active

Target ini menunjukkan kebutuhan follow-up berdasarkan future attendance.
Target bukan bukti customer pasti churn dan bukan pengukuran langsung
perubahan engagement historis.

## File

- `modeling_df.csv`: dataset customer-level hasil Section 6
- `feature_dictionary.csv`: kamus identifier, fitur, dan target
- `README.md`: dokumentasi dataset

## Status preprocessing

Dataset ini sudah melalui:

- agregasi ke level customer,
- pembentukan fitur historis,
- pembentukan fitur package,
- pembentukan fitur reschedule,
- pembentukan fitur interaksi admin,
- pembentukan target biner,
- pemeriksaan missing value,
- pemeriksaan data leakage.

Dataset ini belum melalui:

- train-test split,
- encoding,
- scaling,
- training model,
- evaluasi model.

## Data leakage

Fitur hanya menggunakan informasi yang tersedia sampai 31 Maret 2025.
Future outcome digunakan untuk membentuk target, tetapi tidak dimasukkan
sebagai fitur prediktor.

`customer_id` hanya identifier dan tidak boleh digunakan sebagai fitur model.

## Penggunaan oleh tim AI

Tim AI dapat membaca `modeling_df.csv`, kemudian:

1. memisahkan fitur dan target,
2. melakukan train-test split,
3. melakukan encoding dan scaling dalam pipeline,
4. melatih model,
5. mengevaluasi model pada data yang tidak digunakan saat training.

## Keterbatasan

Dataset bersifat sintetis. Target adalah proxy berbasis future attendance rate.
Horizon hanya mencakup 27 hari. Hasil model tidak dapat langsung
digeneralisasikan ke customer nyata tanpa validasi tambahan.
