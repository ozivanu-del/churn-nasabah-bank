"""Aplikasi Streamlit: Classification Models - Prediksi Churn Nasabah Bank.

Jalankan notebook Churn_Classification.ipynb lebih dulu, karena notebook itulah
yang membuat file model_churn.joblib yang dibaca aplikasi ini.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Identitas (silakan sesuaikan)
NAMA = "Henri Ilham Caniago"
NIM = "251321011"
KELAS = "46A"

FOLDER = Path(__file__).parent
FILE_DATA = FOLDER / "data" / "Churn_Modelling.csv"
if not FILE_DATA.exists():  # CSV boleh juga diletakkan satu folder dengan app.py
    FILE_DATA = FOLDER / "Churn_Modelling.csv"
FILE_MODEL = FOLDER / "model_churn.joblib"
BIRU, MERAH = "#2563EB", "#DC2626"

st.set_page_config(page_title="Prediksi Churn Nasabah Bank", page_icon="🏦", layout="wide")


@st.cache_data
def muat_data():
    return pd.read_csv(FILE_DATA)


@st.cache_resource
def muat_model(waktu_ubah):
    # waktu_ubah membuat cache diperbarui bila model dilatih ulang
    return joblib.load(FILE_MODEL)


def tampilkan(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def hitung_metrik(y, proba, ambang):
    pred = (proba >= ambang).astype(int)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1-Score": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, proba),
    }


for berkas in (FILE_DATA, FILE_MODEL):
    if not berkas.exists():
        st.error(f"File `{berkas.name}` belum ada. Jalankan dulu notebook Churn_Classification.ipynb.")
        st.stop()

df = muat_data()
paket = muat_model(FILE_MODEL.stat().st_mtime)
model_semua = paket["models"]
y_test = paket["y_test"]
nama_model = list(model_semua)

# ------------------------------------------------------------------ SIDEBAR
st.sidebar.title("🏦 Churn Nasabah Bank")
st.sidebar.info(f"**{NAMA}**\n\nNIM: {NIM}\n\nKelas: {KELAS}\n\nMata kuliah: Applied Machine Learning")
menu = st.sidebar.radio(
    "Menu",
    ["1. Beranda & Data", "2. Eksplorasi Data", "3. Perbandingan Model",
     "4. Evaluasi Model", "5. Prediksi Nasabah", "6. Konsep Klasifikasi"],
)
# Pilihan model hanya ditampilkan di menu yang hasilnya memang bergantung pada model.
# Baris session_state menjaga pilihan tetap tersimpan saat pindah menu.
st.session_state["pilih_model"] = st.session_state.get("pilih_model", paket["best_name"])
if menu.startswith(("4", "5")):
    st.sidebar.selectbox("Model yang dipakai", nama_model, key="pilih_model")
pilihan = st.session_state["pilih_model"]
st.sidebar.caption(f"Model terbaik menurut cross-validation: **{paket['best_name']}**")

# ------------------------------------------------------------------ 1. BERANDA
if menu.startswith("1"):
    st.title("Prediksi Churn Nasabah Bank")
    st.write(
        "Aplikasi **Classification Models** untuk memprediksi apakah seorang nasabah akan "
        "**keluar (churn)** atau **bertahan**. Ini klasifikasi **biner**: kelas 1 = keluar, kelas 0 = bertahan."
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Jumlah nasabah", f"{len(df):,}")
    k2.metric("Nasabah keluar", f"{df.Exited.sum():,}")
    k3.metric("Tingkat churn", f"{df.Exited.mean():.1%}")
    k4.metric("Jumlah fitur model", len(paket["features"]))

    st.subheader("Contoh data")
    jumlah = st.slider("Jumlah baris yang ditampilkan", 5, 100, 10)
    st.dataframe(df.head(jumlah), width="stretch", hide_index=True)

    st.subheader("Kamus data")
    st.table(pd.DataFrame(
        [
            ("RowNumber, CustomerId, Surname", "Identitas nasabah (tidak dipakai model)"),
            ("CreditScore", "Skor kredit"),
            ("Geography", "Negara: France, Germany, Spain"),
            ("Gender", "Jenis kelamin"),
            ("Age", "Usia (tahun)"),
            ("Tenure", "Lama menjadi nasabah (tahun)"),
            ("Balance", "Saldo rekening"),
            ("NumOfProducts", "Jumlah produk bank yang dipakai"),
            ("HasCrCard", "Punya kartu kredit (1 = ya)"),
            ("IsActiveMember", "Nasabah aktif (1 = ya)"),
            ("EstimatedSalary", "Perkiraan gaji"),
            ("Exited", "TARGET: 1 = keluar, 0 = bertahan"),
        ],
        columns=["Kolom", "Arti"],
    ))
    st.caption("Sumber data: Kaggle, shrutimechlearn/churn-modelling (Churn_Modelling.csv).")

# ------------------------------------------------------------------ 2. EDA
elif menu.startswith("2"):
    st.title("Eksplorasi Data")
    kiri, kanan = st.columns(2)
    with kiri:
        st.subheader("Distribusi target")
        hitung = df.Exited.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        batang = ax.bar(["Bertahan (0)", "Keluar (1)"], hitung.values, color=[BIRU, MERAH])
        ax.bar_label(batang)
        ax.set_ylabel("Jumlah nasabah")
        tampilkan(fig)
        st.caption("Data **tidak seimbang**: nasabah keluar jauh lebih sedikit, jadi accuracy saja bisa menyesatkan.")
    with kanan:
        st.subheader("Tingkat churn per kategori")
        label = {"Geography": "Negara", "Gender": "Jenis kelamin", "NumOfProducts": "Jumlah produk",
                 "IsActiveMember": "Nasabah aktif", "HasCrCard": "Punya kartu kredit", "Tenure": "Lama menjadi nasabah"}
        kolom = st.selectbox("Pilih kategori", list(label), format_func=label.get)
        ringkas = df.groupby(kolom)["Exited"].agg(["count", "mean"])
        fig, ax = plt.subplots(figsize=(6, 4))
        batang = ax.bar(ringkas.index.astype(str), ringkas["mean"] * 100, color=BIRU)
        ax.bar_label(batang, fmt="%.1f%%")
        ax.set_ylabel("Churn (%)")
        ax.set_xlabel(label[kolom])
        tampilkan(fig)
        st.dataframe(ringkas.rename(columns={"count": "Jumlah nasabah", "mean": "Proporsi churn"}), width="stretch")

    st.subheader("Sebaran fitur angka menurut kelas")
    angka = st.selectbox("Pilih fitur", ["Age", "CreditScore", "Balance", "EstimatedSalary"])
    fig, ax = plt.subplots(figsize=(11, 3.8))
    tepi = np.histogram_bin_edges(df[angka], bins=35)
    for kelas, warna, nama in [(0, BIRU, "Bertahan"), (1, MERAH, "Keluar")]:
        ax.hist(df.loc[df.Exited == kelas, angka], bins=tepi, alpha=0.55, color=warna, label=nama, density=True)
    ax.set_xlabel(angka)
    ax.set_ylabel("Kepadatan")
    ax.legend()
    tampilkan(fig)

    st.subheader("Korelasi antar fitur angka")
    korelasi = df.drop(columns=["RowNumber", "CustomerId"]).corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    gambar = ax.imshow(korelasi, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(korelasi)), korelasi.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(korelasi)), korelasi.columns)
    for i in range(len(korelasi)):
        for j in range(len(korelasi)):
            nilai = korelasi.iat[i, j]
            ax.text(j, i, f"{nilai:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(nilai) > 0.6 else "black")
    fig.colorbar(gambar)
    tampilkan(fig)

# ------------------------------------------------------------------ 3. PERBANDINGAN
elif menu.startswith("3"):
    st.title("Perbandingan Model Klasifikasi")
    st.subheader("Cross-validation 5-fold pada data latih (dasar pemilihan model)")
    st.dataframe(paket["cv"].style.format("{:.3f}").highlight_max(color="#BBF7D0"), width="stretch")
    st.success(f"Model dengan ROC-AUC cross-validation tertinggi: **{paket['best_name']}**")

    st.subheader(f"Hasil pada data uji ({len(y_test):,} nasabah, ambang 0,50)")
    uji = pd.DataFrame({n: hitung_metrik(y_test, paket["test_proba"][n], 0.5) for n in nama_model}).T
    st.dataframe(uji.style.format("{:.3f}").highlight_max(color="#BBF7D0"), width="stretch")

    metrik = st.selectbox("Grafik metrik data uji", list(uji.columns), index=list(uji.columns).index("ROC-AUC"))
    urut = uji[metrik].sort_values()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    batang = ax.barh(urut.index, urut.values, color=BIRU)
    ax.bar_label(batang, fmt="%.3f", padding=3)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(metrik)
    tampilkan(fig)

    st.subheader("Kurva ROC")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for n in nama_model:
        fpr, tpr, _ = roc_curve(y_test, paket["test_proba"][n])
        ax.plot(fpr, tpr, label=f"{n} ({uji.loc[n, 'ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Tebakan acak (0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(fontsize=8)
    tampilkan(fig)

# ------------------------------------------------------------------ 4. EVALUASI
elif menu.startswith("4"):
    st.title(f"Evaluasi: {pilihan}")
    proba = paket["test_proba"][pilihan]
    saran = paket["best_threshold"][pilihan]
    st.info(f"Ambang dengan F1 terbaik menurut data latih: **{saran:.2f}** (ambang bawaan 0,50).")
    ambang = st.slider("Ambang keputusan (probabilitas minimal untuk dianggap KELUAR)", 0.05, 0.95, 0.50, 0.01)

    m = hitung_metrik(y_test, proba, ambang)
    for kotak, (nama, nilai) in zip(st.columns(5), m.items()):
        kotak.metric(nama, f"{nilai:.3f}")
    tn, fp, fn, tp = confusion_matrix(y_test, (proba >= ambang).astype(int)).ravel()

    kiri, kanan = st.columns(2)
    with kiri:
        st.subheader("Confusion matrix")
        matriks = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.imshow(matriks, cmap="Blues")
        for (i, j), nilai in np.ndenumerate(matriks):
            kode = [["TN", "FP"], ["FN", "TP"]][i][j]
            ax.text(j, i, f"{kode}\n{nilai}", ha="center", va="center", fontsize=14,
                    color="white" if nilai > matriks.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Bertahan", "Keluar"])
        ax.set_yticks([0, 1], ["Bertahan", "Keluar"])
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Kenyataan")
        tampilkan(fig)
    with kanan:
        st.subheader("Kurva Precision-Recall")
        presisi, daya, _ = precision_recall_curve(y_test, proba)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(daya, presisi, color=BIRU)
        ax.scatter([m["Recall"]], [m["Precision"]], color=MERAH, s=80, zorder=3, label=f"Ambang {ambang:.2f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        tampilkan(fig)

    st.markdown(
        f"Dari **{tp + fn}** nasabah yang benar-benar keluar, model menangkap **{tp}** dan meloloskan **{fn}**. "
        f"Dari **{tp + fp}** nasabah yang ditandai akan keluar, **{fp}** ternyata bertahan."
    )

    st.subheader("Fitur paling berpengaruh (permutation importance)")
    penting = paket["importance"][pilihan].sort_values()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.barh(penting.index, penting.values, color=BIRU)
    ax.set_xlabel("Penurunan ROC-AUC bila nilai fitur diacak")
    tampilkan(fig)

# ------------------------------------------------------------------ 5. PREDIKSI
elif menu.startswith("5"):
    st.title("Prediksi Nasabah")
    st.write(f"Model yang dipakai: **{pilihan}**")
    model = model_semua[pilihan]
    ambang = st.slider("Ambang keputusan", 0.05, 0.95, 0.50, 0.01)
    satu, banyak = st.tabs(["Satu nasabah", "Banyak nasabah (CSV)"])

    with satu:
        with st.form("form_nasabah"):
            a, b, c = st.columns(3)
            masukan = {
                "CreditScore": a.number_input("Skor kredit", 300, 900, 650),
                "Geography": a.selectbox("Negara", sorted(df.Geography.unique())),
                "Gender": a.selectbox("Jenis kelamin", sorted(df.Gender.unique())),
                "Age": a.number_input("Usia", 18, 100, 40),
                "Tenure": b.slider("Lama menjadi nasabah (tahun)", 0, 10, 5),
                "Balance": b.number_input("Saldo", 0.0, 300000.0, 75000.0, 1000.0),
                "NumOfProducts": b.selectbox("Jumlah produk", [1, 2, 3, 4]),
                "HasCrCard": int(c.checkbox("Punya kartu kredit", True)),
                "IsActiveMember": int(c.checkbox("Nasabah aktif", True)),
                "EstimatedSalary": c.number_input("Perkiraan gaji", 0.0, 250000.0, 100000.0, 1000.0),
            }
            kirim = st.form_submit_button("Prediksi", type="primary")
        if kirim:
            p = model.predict_proba(pd.DataFrame([masukan])[paket["features"]])[0, 1]
            st.metric("Probabilitas keluar (churn)", f"{p:.1%}")
            st.progress(float(p))
            if p >= ambang:
                st.error("Prediksi: **KELUAR**. Nasabah ini layak diprioritaskan untuk program retensi.")
            else:
                st.success("Prediksi: **BERTAHAN**.")
            if masukan["NumOfProducts"] >= 3:
                jumlah = int((df.NumOfProducts >= 3).sum())
                st.warning(f"Hanya {jumlah} nasabah di data yang punya 3 produk atau lebih, hati-hati menafsirkan hasilnya.")

    with banyak:
        st.write("Unggah CSV yang memiliki kolom: " + ", ".join(f"`{k}`" for k in paket["features"]))
        contoh = df[paket["features"]].head(20)
        st.download_button("Unduh contoh CSV", contoh.to_csv(index=False), "contoh_nasabah.csv", "text/csv")
        unggah = st.file_uploader("File CSV", type="csv")
        if unggah is not None:
            try:
                baru = pd.read_csv(unggah)
                kurang = [k for k in paket["features"] if k not in baru.columns]
                if kurang:
                    st.error("Kolom yang belum ada: " + ", ".join(kurang))
                else:
                    p = model.predict_proba(baru[paket["features"]])[:, 1]
                    hasil = baru.assign(
                        Probabilitas_Churn=p.round(4),
                        Prediksi=np.where(p >= ambang, "Keluar", "Bertahan"),
                    ).sort_values("Probabilitas_Churn", ascending=False)
                    st.dataframe(hasil, width="stretch", hide_index=True)
                    st.download_button("Unduh hasil prediksi", hasil.to_csv(index=False), "hasil_prediksi.csv", "text/csv")
            except ValueError as galat:
                st.error(f"CSV tidak bisa diproses (periksa nilai kosong atau kategori yang tidak dikenal): {galat}")

# ------------------------------------------------------------------ 6. KONSEP
else:
    st.title("Konsep Classification Models")
    st.markdown(
        """
**Klasifikasi** adalah supervised learning yang memprediksi **kategori**. Bandingkan dengan regresi yang memprediksi angka.

| Jenis | Contoh |
|---|---|
| Biner (2 kelas) | Nasabah keluar atau bertahan (**aplikasi ini**) |
| Multikelas | Mengenali angka tulisan tangan 0-9 |
| Multilabel | Satu artikel diberi beberapa topik sekaligus |

### Logistic Regression
Walaupun namanya "regression", ini algoritma klasifikasi. Fungsi **sigmoid** mengubah skor menjadi probabilitas 0 sampai 1:
"""
    )
    st.latex(r"P(y=1\mid x)=\frac{1}{1+e^{-(b_0+b_1x_1+\dots+b_nx_n)}}")
    z = np.linspace(-8, 8, 200)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(z, 1 / (1 + np.exp(-z)), color=BIRU)
    ax.axhline(0.5, color=MERAH, linestyle="--", label="Ambang 0,5")
    ax.set_xlabel("Skor linear")
    ax.set_ylabel("Probabilitas")
    ax.legend()
    tampilkan(fig)

    st.markdown(
        """
### Algoritma lain yang dibandingkan
| Algoritma | Ide dasar |
|---|---|
| k-Nearest Neighbors | Suara terbanyak dari k nasabah yang paling mirip |
| Gaussian Naive Bayes | Teorema Bayes dengan anggapan fitur saling bebas |
| Support Vector Machine | Mencari batas pemisah dengan margin terlebar |
| Decision Tree | Rangkaian pertanyaan ya/tidak |
| Random Forest | Gabungan banyak pohon yang dibuat dari sampel acak |
| Gradient Boosting | Pohon dibuat berurutan untuk memperbaiki kesalahan sebelumnya |

### Mengukur kinerja
- **TP**: keluar dan diprediksi keluar. **TN**: bertahan dan diprediksi bertahan.
- **FP**: bertahan tetapi diprediksi keluar. **FN**: keluar tetapi diprediksi bertahan.
"""
    )
    st.latex(r"Accuracy=\frac{TP+TN}{TP+TN+FP+FN}\qquad Precision=\frac{TP}{TP+FP}")
    st.latex(r"Recall=\frac{TP}{TP+FN}\qquad F1=2\times\frac{Precision\times Recall}{Precision+Recall}")
    st.markdown(
        f"""
- **ROC-AUC**: kemampuan model memberi skor lebih tinggi pada nasabah yang keluar (0,5 = acak, 1 = sempurna).

### Data tidak seimbang
Hanya {df.Exited.mean():.1%} nasabah yang keluar. Model yang selalu menjawab "bertahan" sudah mendapat accuracy
{1 - df.Exited.mean():.1%}, padahal tidak menemukan satu pun nasabah yang akan keluar.
Cara menanganinya: **class weight**, **oversampling/undersampling**, atau **menggeser ambang keputusan**.
"""
    )
