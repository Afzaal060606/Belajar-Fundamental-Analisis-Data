import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

# ==============================================================================
# 1. KONFIGURASI HALAMAN & GAYA VISUAL
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Analisis E-Commerce Olist (2017–2018)",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid")
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8

PRIMARY_COLOR = "#1E40AF"      # Biru tua elegan
SECONDARY_COLOR = "#3B82F6"    # Biru medium
MUTED_COLOR = "#94A3B8"        # Abu-abu netral
ALERT_COLOR = "#DC2626"        # Merah untuk keterlambatan / penurunan
SUCCESS_COLOR = "#059669"      # Hijau untuk ketepatan waktu / pertumbuhan


# ==============================================================================
# 2. LOAD DATA DENGAN FALLBACK PATH & CACHING
# ==============================================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    possible_paths = [
        "dashboard/main_data.csv",
        "main_data.csv",
        os.path.join(os.path.dirname(__file__), "main_data.csv"),
    ]
    csv_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if csv_path is None:
        raise FileNotFoundError(
            "File 'main_data.csv' tidak ditemukan di folder 'dashboard/' maupun direktori root."
        )

    df = pd.read_csv(csv_path)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["product_category_name_english"] = df["product_category_name_english"].fillna("unknown")
    df["primary_payment_type"] = df["primary_payment_type"].fillna("unknown")
    return df


# ==============================================================================
# 3. FUNGSI HELPER AGREGASI DATA (PER PERTANYAAN BISNIS)
# ==============================================================================
def prepare_category_revenue(df: pd.DataFrame, top_n: int = 10):
    cat_rev = (
        df.groupby("product_category_name_english", as_index=False)["price"]
        .sum()
        .rename(columns={"product_category_name_english": "category", "price": "revenue"})
        .sort_values("revenue", ascending=False)
    )
    top_cat = cat_rev.head(top_n).copy()
    bottom_cat = cat_rev.tail(top_n).sort_values("revenue", ascending=True).copy()
    return top_cat, bottom_cat, cat_rev


def prepare_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["year_month"] = temp["order_purchase_timestamp"].dt.to_period("M").astype(str)
    monthly = (
        temp.groupby("year_month", as_index=False)["price"]
        .sum()
        .rename(columns={"price": "revenue"})
        .sort_values("year_month")
    )
    monthly["mom_growth"] = monthly["revenue"].pct_change() * 100
    return monthly


def prepare_delivery_satisfaction(orders_df: pd.DataFrame):
    delivered = orders_df.dropna(subset=["is_late"]).copy()
    delivered["status_pengiriman"] = np.where(
        delivered["is_late"] == 1.0, "Terlambat (Late)", "Tepat Waktu (On-Time)"
    )

    review_comp = (
        delivered.dropna(subset=["review_score"])
        .groupby("status_pengiriman", as_index=False)
        .agg(
            jumlah_pesanan=("order_id", "count"),
            rata_rata_skor=("review_score", "mean"),
        )
        .sort_values("rata_rata_skor", ascending=False)
    )

    state_late = (
        delivered.groupby("customer_state", as_index=False)
        .agg(
            total_pesanan=("order_id", "count"),
            pesanan_terlambat=("is_late", "sum"),
            persentase_terlambat=("is_late", "mean"),
        )
    )
    state_late["persentase_terlambat"] = state_late["persentase_terlambat"] * 100
    state_late = state_late.sort_values("persentase_terlambat", ascending=False)
    return review_comp, state_late


def prepare_payment_analysis(orders_df: pd.DataFrame) -> pd.DataFrame:
    pay_df = (
        orders_df.groupby("primary_payment_type", as_index=False)
        .agg(
            jumlah_transaksi=("order_id", "count"),
            total_nilai=("total_payment", "sum"),
            aov=("total_payment", "mean"),
        )
        .sort_values("total_nilai", ascending=False)
    )
    total_tx = pay_df["jumlah_transaksi"].sum()
    total_val = pay_df["total_nilai"].sum()
    pay_df["pangsa_transaksi_pct"] = (pay_df["jumlah_transaksi"] / total_tx * 100) if total_tx else 0
    pay_df["kontribusi_nilai_pct"] = (pay_df["total_nilai"] / total_val * 100) if total_val else 0

    label_map = {
        "credit_card": "Credit Card",
        "boleto": "Boleto",
        "voucher": "Voucher",
        "debit_card": "Debit Card",
    }
    pay_df["metode_label"] = pay_df["primary_payment_type"].map(label_map).fillna(pay_df["primary_payment_type"])
    return pay_df


# ==============================================================================
# 4. LOAD DATA & SIDEBAR FILTER INTERAKTIF
# ==============================================================================
df = load_data()

with st.sidebar:
    st.header("Filter Data")
    st.caption("Sesuaikan rentang waktu dan segmen untuk memperbarui seluruh metrik dan visualisasi.")

    min_date = df["order_purchase_timestamp"].min().date()
    max_date = df["order_purchase_timestamp"].max().date()

    date_range = st.date_input(
        label="Rentang Tanggal Pembelian",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    all_states = sorted(df["customer_state"].dropna().unique().tolist())
    selected_states = st.multiselect(
        label="Negara Bagian (State)",
        options=all_states,
        default=[],
        placeholder="Semua Negara Bagian (27 State)",
    )

    payment_labels = {
        "credit_card": "Credit Card",
        "boleto": "Boleto",
        "voucher": "Voucher",
        "debit_card": "Debit Card",
    }
    all_payments = sorted(df["primary_payment_type"].dropna().unique().tolist())
    selected_payments = st.multiselect(
        label="Metode Pembayaran",
        options=all_payments,
        default=[],
        format_func=lambda x: payment_labels.get(x, x),
        placeholder="Semua Metode Pembayaran",
    )

    st.divider()
    st.markdown(
        "**Cakupan Dataset:**\n"
        "- Periode: Jan 2017 – Agu 2018 (20 Bulan)\n"
        "- Mata Uang: Brazilian Real (BRL)"
    )

# Terapkan filter ke DataFrame utama
mask = (
    (df["order_purchase_timestamp"].dt.date >= start_date)
    & (df["order_purchase_timestamp"].dt.date <= end_date)
)
if selected_states:
    mask &= df["customer_state"].isin(selected_states)
if selected_payments:
    mask &= df["primary_payment_type"].isin(selected_payments)

main_df = df[mask].copy()

if main_df.empty:
    st.warning("Tidak ada data yang sesuai dengan kombinasi filter saat ini. Silakan ubah filter di sidebar.")
    st.stop()

# DataFrame tingkat pesanan (deduplikasi order_id agar metrik order/payment/review akurat)
orders_df = main_df.drop_duplicates(subset=["order_id"]).copy()


# ==============================================================================
# 5. HEADER & KPI RINGKASAN EKSEKUTIF
# ==============================================================================
st.title("Dashboard Analisis Performa E-Commerce")
st.markdown(
    f"Analisis komprehensif performa kategori produk, ketepatan waktu pengiriman, dan metode pembayaran "
    f"periode **{start_date.strftime('%d %b %Y')}** hingga **{end_date.strftime('%d %b %Y')}**."
)

total_orders = orders_df["order_id"].nunique()
total_revenue = main_df["price"].sum()
avg_review = orders_df["review_score"].mean()
late_rate = orders_df["is_late"].dropna().mean() * 100

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Total Pesanan (Orders)", f"{total_orders:,}")
with kpi2:
    st.metric("Total Pendapatan Produk", f"BRL {total_revenue:,.2f}")
with kpi3:
    st.metric("Rata-rata Skor Ulasan", f"{avg_review:.2f} / 5.00")
with kpi4:
    st.metric("Tingkat Keterlambatan", f"{late_rate:.2f}%")

st.divider()


# ==============================================================================
# 6. PERTANYAAN BISNIS 1: PERFORMA KATEGORI PRODUK & TREN PENDAPATAN
# ==============================================================================
st.subheader("1. Performa Kategori Produk & Tren Pertumbuhan Pendapatan (MoM)")
st.caption(
    "Identifikasi kategori penyumbang pendapatan tertinggi dan terendah serta dinamika pertumbuhan bulanan."
)

top_cat, bottom_cat, full_cat = prepare_category_revenue(main_df, top_n=10)
monthly_df = prepare_monthly_trend(main_df)

col_q1_left, col_q1_right = st.columns(2)

with col_q1_left:
    fig1a, ax1a = plt.subplots(figsize=(8, 4.8))
    colors_top = [PRIMARY_COLOR if i == 0 else SECONDARY_COLOR for i in range(len(top_cat))]
    sns.barplot(
        data=top_cat,
        x="revenue",
        y="category",
        palette=colors_top,
        hue="category",
        legend=False,
        ax=ax1a,
    )
    ax1a.set_title("Top 10 Kategori Produk Pendapatan Tertinggi", fontsize=11, fontweight="bold", pad=10)
    ax1a.set_xlabel("Total Pendapatan (BRL)", fontsize=9)
    ax1a.set_ylabel("")
    ax1a.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    ax1a.set_xlim(0, top_cat["revenue"].max() * 1.18)
    for bar in ax1a.patches:
        w = bar.get_width()
        ax1a.text(
            w + top_cat["revenue"].max() * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"BRL {w:,.0f}",
            va="center",
            fontsize=8,
            color="#1E293B",
        )
    sns.despine(ax=ax1a, top=True, right=True)
    plt.tight_layout()
    st.pyplot(fig1a)
    plt.close(fig1a)

with col_q1_right:
    fig1b, ax1b = plt.subplots(figsize=(8, 4.8))
    colors_bot = [ALERT_COLOR if i == 0 else MUTED_COLOR for i in range(len(bottom_cat))]
    sns.barplot(
        data=bottom_cat,
        x="revenue",
        y="category",
        palette=colors_bot,
        hue="category",
        legend=False,
        ax=ax1b,
    )
    ax1b.set_title("Bottom 10 Kategori Produk Pendapatan Terendah", fontsize=11, fontweight="bold", pad=10)
    ax1b.set_xlabel("Total Pendapatan (BRL)", fontsize=9)
    ax1b.set_ylabel("")
    ax1b.set_xlim(0, bottom_cat["revenue"].max() * 1.25)
    for bar in ax1b.patches:
        w = bar.get_width()
        ax1b.text(
            w + bottom_cat["revenue"].max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"BRL {w:,.2f}",
            va="center",
            fontsize=8,
            color="#1E293B",
        )
    sns.despine(ax=ax1b, top=True, right=True)
    plt.tight_layout()
    st.pyplot(fig1b)
    plt.close(fig1b)

# Grafik Tren Pendapatan Bulanan & Pertumbuhan MoM
fig1c, (ax_rev, ax_mom) = plt.subplots(
    2, 1, figsize=(13, 5.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
)

ax_rev.plot(
    monthly_df["year_month"],
    monthly_df["revenue"],
    marker="o",
    linewidth=2.2,
    color=PRIMARY_COLOR,
)
ax_rev.fill_between(monthly_df["year_month"], monthly_df["revenue"], color=SECONDARY_COLOR, alpha=0.12)
ax_rev.set_title("Tren Pendapatan Bulanan (BRL) & Laju Pertumbuhan Month-over-Month (MoM %)", fontsize=11, fontweight="bold", pad=10)
ax_rev.set_ylabel("Pendapatan (BRL)", fontsize=9)
ax_rev.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:,.0f}K"))
ax_rev.set_ylim(0, monthly_df["revenue"].max() * 1.18)
for idx, row in monthly_df.iterrows():
    ax_rev.annotate(
        f"{row['revenue']/1e3:.0f}K",
        (row["year_month"], row["revenue"]),
        textcoords="offset points",
        xytext=(0, 7),
        ha="center",
        fontsize=7.5,
        color="#1E293B",
    )
sns.despine(ax=ax_rev, top=True, right=True)

mom_vals = monthly_df["mom_growth"].fillna(0)
mom_colors = [SUCCESS_COLOR if v >= 0 else ALERT_COLOR for v in mom_vals]
ax_mom.bar(monthly_df["year_month"], mom_vals, color=mom_colors, alpha=0.85, width=0.6)
ax_mom.axhline(0, color="#64748B", linewidth=0.9)
ax_mom.set_ylabel("MoM (%)", fontsize=9)
ax_mom.set_xlabel("Bulan Transaksi", fontsize=9)
plt.xticks(rotation=45, ha="right", fontsize=8.5)
for i, v in enumerate(monthly_df["mom_growth"]):
    if pd.notna(v):
        offset = 3 if v >= 0 else -11
        ax_mom.annotate(
            f"{v:+.1f}%",
            (monthly_df["year_month"].iloc[i], v),
            textcoords="offset points",
            xytext=(0, offset),
            ha="center",
            fontsize=7,
            color="#1E293B",
        )
sns.despine(ax=ax_mom, top=True, right=True)
plt.tight_layout()
st.pyplot(fig1c)
plt.close(fig1c)

if not top_cat.empty and not bottom_cat.empty:
    best_cat_name = top_cat.iloc[0]["category"]
    best_cat_rev = top_cat.iloc[0]["revenue"]
    worst_cat_name = bottom_cat.iloc[0]["category"]
    worst_cat_rev = bottom_cat.iloc[0]["revenue"]
    peak_month = monthly_df.loc[monthly_df["revenue"].idxmax()]
    st.info(
        f"**Insight Utama:** Kategori **{best_cat_name}** memimpin pendapatan sebesar **BRL {best_cat_rev:,.2f}**, "
        f"sementara **{worst_cat_name}** berada di posisi terendah (**BRL {worst_cat_rev:,.2f}**). "
        f"Puncak pendapatan bulanan terjadi pada **{peak_month['year_month']}** sebesar **BRL {peak_month['revenue']:,.2f}**."
    )

st.divider()


# ==============================================================================
# 7. PERTANYAAN BISNIS 2: KETEPATAN WAKTU PENGIRIMAN & KEPUASAN PELANGGAN
# ==============================================================================
st.subheader("2. Dampak Ketepatan Waktu Pengiriman terhadap Kepuasan Pelanggan")
st.caption(
    "Perbandingan rata-rata review_score berdasarkan status ketepatan pengiriman serta pemetaan tingkat keterlambatan per negara bagian."
)

review_comp_df, state_late_df = prepare_delivery_satisfaction(orders_df)

col_q2_left, col_q2_right = st.columns([1, 1.35])

with col_q2_left:
    fig2a, ax2a = plt.subplots(figsize=(6.5, 4.5))
    palette_status = {
        "Tepat Waktu (On-Time)": SUCCESS_COLOR,
        "Terlambat (Late)": ALERT_COLOR,
    }
    sns.barplot(
        data=review_comp_df,
        x="status_pengiriman",
        y="rata_rata_skor",
        palette=palette_status,
        hue="status_pengiriman",
        legend=False,
        width=0.5,
        ax=ax2a,
    )
    ax2a.set_title("Rata-rata Skor Ulasan: Tepat Waktu vs. Terlambat", fontsize=11, fontweight="bold", pad=10)
    ax2a.set_xlabel("Status Pengiriman", fontsize=9)
    ax2a.set_ylabel("Rata-rata Review Score (1–5)", fontsize=9)
    ax2a.set_ylim(0, 5.3)
    for bar, (_, row) in zip(ax2a.patches, review_comp_df.iterrows()):
        h = bar.get_height()
        ax2a.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.12,
            f"{h:.2f}\n({row['jumlah_pesanan']:,} order)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#1E293B",
        )
    sns.despine(ax=ax2a, top=True, right=True)
    plt.tight_layout()
    st.pyplot(fig2a)
    plt.close(fig2a)

with col_q2_right:
    top_late_states = state_late_df.head(10).copy()
    fig2b, ax2b = plt.subplots(figsize=(8, 4.5))
    state_colors = [ALERT_COLOR if i < 3 else SECONDARY_COLOR for i in range(len(top_late_states))]
    sns.barplot(
        data=top_late_states,
        x="persentase_terlambat",
        y="customer_state",
        palette=state_colors,
        hue="customer_state",
        legend=False,
        ax=ax2b,
    )
    ax2b.set_title("Top 10 Negara Bagian dengan Tingkat Keterlambatan Tertinggi (%)", fontsize=11, fontweight="bold", pad=10)
    ax2b.set_xlabel("Persentase Pesanan Terlambat (%)", fontsize=9)
    ax2b.set_ylabel("Negara Bagian (State)", fontsize=9)
    ax2b.set_xlim(0, top_late_states["persentase_terlambat"].max() * 1.25)
    for bar, (_, row) in zip(ax2b.patches, top_late_states.iterrows()):
        w = bar.get_width()
        ax2b.text(
            w + top_late_states["persentase_terlambat"].max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.2f}% ({int(row['pesanan_terlambat'])}/{int(row['total_pesanan'])})",
            va="center",
            fontsize=8,
            color="#1E293B",
        )
    sns.despine(ax=ax2b, top=True, right=True)
    plt.tight_layout()
    st.pyplot(fig2b)
    plt.close(fig2b)

if len(review_comp_df) == 2 and not state_late_df.empty:
    ontime_score = review_comp_df.loc[
        review_comp_df["status_pengiriman"] == "Tepat Waktu (On-Time)", "rata_rata_skor"
    ].values[0]
    late_score = review_comp_df.loc[
        review_comp_df["status_pengiriman"] == "Terlambat (Late)", "rata_rata_skor"
    ].values[0]
    diff_score = ontime_score - late_score
    worst_state = state_late_df.iloc[0]
    st.info(
        f"**Insight Utama:** Pesanan terlambat mengalami penurunan kepuasan sebesar **{diff_score:.2f} poin** "
        f"(dari **{ontime_score:.2f}** menjadi **{late_score:.2f}**). "
        f"Negara bagian **{worst_state['customer_state']}** mencatat tingkat keterlambatan tertinggi sebesar "
        f"**{worst_state['persentase_terlambat']:.2f}%** ({int(worst_state['pesanan_terlambat'])} dari {int(worst_state['total_pesanan'])} pesanan)."
    )

st.divider()


# ==============================================================================
# 8. PERTANYAAN BISNIS 3: METODE PEMBAYARAN & NILAI TRANSAKSI
# ==============================================================================
st.subheader("3. Distribusi Metode Pembayaran, Kontribusi Nilai & Average Order Value (AOV)")
st.caption(
    "Evaluasi pangsa transaksi, kontribusi total pembayaran (BRL), dan rata-rata nilai keranjang belanja per metode pembayaran."
)

payment_df = prepare_payment_analysis(orders_df)

col_q3_left, col_q3_right = st.columns(2)

with col_q3_left:
    fig3a, ax3a = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(payment_df))
    width = 0.36

    rects1 = ax3a.bar(
        x - width / 2,
        payment_df["pangsa_transaksi_pct"],
        width,
        label="Pangsa Jumlah Transaksi (%)",
        color=SECONDARY_COLOR,
    )
    rects2 = ax3a.bar(
        x + width / 2,
        payment_df["kontribusi_nilai_pct"],
        width,
        label="Kontribusi Nilai Pembayaran (%)",
        color=PRIMARY_COLOR,
    )

    ax3a.set_title("Pangsa Transaksi (%) vs. Kontribusi Nilai Pembayaran (%)", fontsize=11, fontweight="bold", pad=10)
    ax3a.set_xticks(x)
    ax3a.set_xticklabels(payment_df["metode_label"], fontsize=9)
    ax3a.set_ylabel("Persentase (%)", fontsize=9)
    ax3a.set_ylim(0, max(payment_df["pangsa_transaksi_pct"].max(), payment_df["kontribusi_nilai_pct"].max()) * 1.18)
    ax3a.legend(frameon=True, fontsize=8.5)

    for rect in rects1:
        h = rect.get_height()
        ax3a.text(rect.get_x() + rect.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax3a.text(rect.get_x() + rect.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", fontsize=8, fontweight="bold")

    sns.despine(ax=ax3a, top=True, right=True)
    plt.tight_layout()
    st.pyplot(fig3a)
    plt.close(fig3a)

with col_q3_right:
    fig3b, ax3b = plt.subplots(figsize=(7.5, 4.5))
    aov_sorted = payment_df.sort_values("aov", ascending=False)
    aov_colors = [PRIMARY_COLOR if i == 0 else SECONDARY_COLOR for i in range(len(aov_sorted))]
    sns.barplot(
        data=aov_sorted,
        x="metode_label",
        y="aov",
        palette=aov_colors,
        hue="metode_label",
        legend=False,
        width=0.5,
        ax=ax3b,
    )
    ax3b.set_title("Rata-rata Nilai Pesanan / AOV (BRL) per Metode Pembayaran", fontsize=11, fontweight="bold", pad=10)
    ax3b.set_xlabel("Metode Pembayaran", fontsize=9)
    ax3b.set_ylabel("Average Order Value (BRL)", fontsize=9)
    ax3b.set_ylim(0, aov_sorted["aov"].max() * 1.2)
    for bar in ax3b.patches:
        h = bar.get_height()
        ax3b.text(
            bar.get_x() + bar.get_width() / 2,
            h + aov_sorted["aov"].max() * 0.025,
            f"BRL {h:,.2f}",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#1E293B",
        )
    sns.despine(ax=ax3b, top=True, right=True)
    plt.tight_layout()
    st.pyplot(fig3b)
    plt.close(fig3b)

# Tabel Ringkasan Metode Pembayaran
display_pay = payment_df[
    ["metode_label", "jumlah_transaksi", "pangsa_transaksi_pct", "total_nilai", "kontribusi_nilai_pct", "aov"]
].rename(
    columns={
        "metode_label": "Metode Pembayaran",
        "jumlah_transaksi": "Jumlah Transaksi",
        "pangsa_transaksi_pct": "Pangsa Transaksi (%)",
        "total_nilai": "Total Nilai Pembayaran (BRL)",
        "kontribusi_nilai_pct": "Kontribusi Nilai (%)",
        "aov": "Average Order Value / AOV (BRL)",
    }
)
st.dataframe(
    display_pay.style.format(
        {
            "Jumlah Transaksi": "{:,}",
            "Pangsa Transaksi (%)": "{:.2f}%",
            "Total Nilai Pembayaran (BRL)": "BRL {:,.2f}",
            "Kontribusi Nilai (%)": "{:.2f}%",
            "Average Order Value / AOV (BRL)": "BRL {:,.2f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

if not payment_df.empty:
    top_pay = payment_df.iloc[0]
    st.info(
        f"**Insight Utama:** **{top_pay['metode_label']}** mendominasi dengan **{top_pay['pangsa_transaksi_pct']:.2f}%** "
        f"dari total transaksi dan menyumbang **{top_pay['kontribusi_nilai_pct']:.2f}%** dari total nilai pembayaran "
        f"(**BRL {top_pay['total_nilai']:,.2f}**), sekaligus mencatat AOV tertinggi sebesar **BRL {top_pay['aov']:,.2f}**."
    )