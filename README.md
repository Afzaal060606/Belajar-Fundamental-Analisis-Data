# Proyek Analisis Data: E-Commerce Public Dataset (Olist)

Proyek ini merupakan analisis data menyeluruh dan pengembangan dashboard interaktif menggunakan **Streamlit** terhadap dataset transaksi e-commerce (periode **Januari 2017 – Agustus 2018**). Analisis difokuskan pada tiga aspek utama bisnis untuk mendukung pengambilan keputusan berbasis data:

1. **Performa Kategori Produk & Tren Pendapatan:** Mengidentifikasi kategori penyumbang pendapatan (*revenue*) tertinggi dan terendah serta memantau laju pertumbuhan bulanan (*Month-over-Month*).
2. **Ketepatan Waktu Pengiriman & Kepuasan Pelanggan:** Mengukur dampak keterlambatan pengiriman (*late delivery*) terhadap skor ulasan pelanggan (`review_score`) dan memetakan negara bagian (*state*) dengan tingkat keterlambatan tertinggi.
3. **Metode Pembayaran & Nilai Transaksi:** Menganalisis pangsa penggunaan tiap metode pembayaran, kontribusinya terhadap total nilai transaksi, serta rata-rata nilai pesanan (*Average Order Value* / AOV).

---

## Struktur Direktori

```text
├── dashboard/
│   ├── dashboard.py       # Aplikasi utama dashboard interaktif Streamlit
│   └── main_data.csv      # Dataset bersih hasil preprocessing dari notebook
├── data/                  # Dataset mentah awal (CSV)
├── notebook.ipynb         # Jupyter Notebook berisi proses wrangling, EDA, & visualisasi
├── README.md              # Dokumentasi proyek dan panduan penggunaan
├── requirements.txt       # Daftar pustaka Python yang dibutuhkan
└── url.txt                # Tautan aplikasi Streamlit Community Cloud
```

---

## Panduan Instalasi & Menjalankan Dashboard Lokal

### 1. Clone Repositori
```bash
git clone https://github.com/Afzaal060606/<nama-repo>.git
cd <nama-repo>
```

### 2. Setup Environment

**Opsi A — Menggunakan Conda (Anaconda/Miniconda):**
```bash
conda create --name main-ds python=3.10 -y
conda activate main-ds
pip install -r requirements.txt
```

**Opsi B — Menggunakan Terminal / Virtual Environment (`venv`):**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Jalankan Aplikasi Streamlit
Pastikan terminal berada di direktori *root* proyek, lalu jalankan perintah berikut:
```bash
streamlit run dashboard/dashboard.py
```
Dashboard akan otomatis terbuka di browser pada alamat **`http://localhost:8501`**.

---

## Dashboard Online (Streamlit Community Cloud)

Versi *online* dari dashboard ini dapat diakses melalui tautan yang tercantum pada file **`url.txt`** di direktori utama proyek.