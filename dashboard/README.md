# Dashboard Customer Retention Risk

Dasbor Streamlit untuk memantau indikator customer dan menyusun daftar prioritas follow-up.

## Menjalankan

```bash
streamlit run dashboard/app.py
```

## Sumber data

- Default: `data_ready/modeling_df.csv` (3.000 customer sintetis).
- Pembaruan data: unggah CSV melalui sidebar. Format kolom harus sama dengan file default.
- Kolom wajib: `customer_id`, 20 fitur prediktor, dan `needs_follow_up`.

## Isi dasbor

- Indikator: jumlah customer, proporsi perlu follow-up, rata-rata attendance rate, recent attendance, dan package usage.
- Filter: tipe package, rentang attendance rate, rentang recent attendance, maksimum days since last visit, minimum sinyal aturan, dan hanya customer perlu follow-up.
- Visual: komposisi kebutuhan follow-up, perbandingan attendance per kebutuhan follow-up, proporsi perlu follow-up per tipe package dan per package usage.
- Daftar prioritas: customer diurutkan berdasarkan jumlah sinyal aturan, dengan unduhan CSV.

## Catatan

- Skor prioritas bersifat aturan sederhana, bukan output model.
- Target `needs_follow_up` adalah label hasil future engagement pada horizon 1–27 April 2025.
- Data sintetis; bukan data customer nyata.
