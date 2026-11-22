import pandas as pd

df = pd.read_csv("heart_attack_prediction_indonesia.csv", sep=";")
print(df.shape)
print(df.head())

# Cek kesehatan data (kosong, duplikat, tipe data)
print("\nMissing Values:")
print(df.isna().sum())

df.info()

# Cek duplikat
print("\nJumlah data duplikat:", df.duplicated().sum())


import pandas as pd

# Membaca dataset
df = pd.read_csv("heart_attack_prediction_indonesia.csv")

# Menampilkan ukuran data awal
print("Ukuran Data Awal:")
print(df.shape)

# Cek konsistensi penulisan kolom kategorik
print("\nJenis Kelamin:")
print(df["gender"].value_counts())

print("\nWilayah:")
print(df["region"].value_counts())

print("\nTingkat Pendapatan:")
print(df["income_level"].value_counts())

print("\nStatus Merokok:")
print(df["smoking_status"].value_counts())

print("\nKonsumsi Alkohol:")
print(df["alcohol_consumption"].value_counts(dropna=False))

print("\nAktivitas Fisik:")
print(df["physical_activity"].value_counts())

print("\nPola Makan:")
print(df["dietary_habits"].value_counts())

print("\nPaparan Polusi Udara:")
print(df["air_pollution_exposure"].value_counts())

print("\nTingkat Stres:")
print(df["stress_level"].value_counts())

print("\nHasil EKG:")
print(df["EKG_results"].value_counts())

# Cek kolom numerik
print("\nStatistik Deskriptif:")

kolom_numerik = [
    "age",
    "cholesterol_level",
    "waist_circumference",
    "sleep_hours",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "fasting_blood_sugar",
    "cholesterol_hdl",
    "cholesterol_ldl",
    "triglycerides"
]

print(df[kolom_numerik].describe())

# Cek spasi berlebih pada kolom teks
print("\nJumlah Data Unik Setelah Strip:")

kolom_teks = [
    "gender",
    "region",
    "income_level",
    "smoking_status",
    "alcohol_consumption",
    "physical_activity",
    "dietary_habits",
    "air_pollution_exposure",
    "stress_level",
    "EKG_results"
]

for kolom in kolom_teks:
    print(kolom, df[kolom].str.strip().nunique())

# Hapus spasi berlebih pada kolom teks
for kolom in df.select_dtypes(include="object").columns:
    df[kolom] = df[kolom].str.strip()

# Cek nilai minimum, median, dan maksimum
print("\nNilai Minimum, Median, Maksimum:")
print(df.describe().T[["min", "50%", "max"]])

# Cek jumlah data kosong setiap kolom
print("\nJumlah Data Kosong:")
print(df.isnull().sum())

# Cek persentase data kosong
print("\nPersentase Data Kosong:")
print((df.isnull().mean() * 100).round(2))

# Cek nilai unik pada kolom kategorik
print("\nNilai Kategorik yang Tersedia:")

for kolom in kolom_teks:
    print(f"\n{kolom}:")
    print(df[kolom].unique())

# Cek nilai pada kolom biner
kolom_biner = [
    "hypertension",
    "diabetes",
    "obesity",
    "family_history",
    "previous_heart_disease",
    "medication_usage",
    "participated_in_free_screening",
    "heart_attack"
]

print("\nNilai pada Kolom Biner:")

for kolom in kolom_biner:
    print(kolom, df[kolom].unique())

# Cek nilai yang tidak sesuai pada kolom biner
print("\nPengecekan Nilai Kolom Biner:")

for kolom in kolom_biner:
    print(kolom, 
          df[~df[kolom].isin([0, 1])].shape[0])

# Cek data duplikat
print("\nJumlah Data Duplikat:")
print(df.duplicated().sum())

# Menampilkan data yang duplikat
print("\nData yang Mengalami Duplikasi:")
print(df[df.duplicated(keep=False)])

# Cek jumlah data berdasarkan target
print("\nJumlah Data Serangan Jantung:")
print(df["heart_attack"].value_counts())

# Cek jumlah data berdasarkan jenis kelamin
print("\nJumlah Data Berdasarkan Jenis Kelamin:")
print(df["gender"].value_counts())

# Cek jumlah data berdasarkan wilayah
print("\nJumlah Data Berdasarkan Wilayah:")
print(df["region"].value_counts())

# Cek nilai negatif pada kolom numerik
print("\nJumlah Nilai Negatif:")

for kolom in kolom_numerik:
    print(kolom, (df[kolom] < 0).sum())

# Cek data usia
print("\nRentang Usia:")
print("Usia Minimum:", df["age"].min())
print("Usia Maksimum:", df["age"].max())

# Cek data tekanan darah
print("\nRentang Tekanan Darah:")
print(df[[
    "blood_pressure_systolic",
    "blood_pressure_diastolic"
]].describe())

# Hapus data duplikat
df = df.drop_duplicates()

print("\nUkuran Data Setelah Cleansing:")
print(df.shape)

# Menampilkan 100 data pertama
df.head(100).to_csv(
    "preview_heart_attack_prediction.csv",
    index=False
)

print("\nData Preview Berhasil Disimpan!")