# DermaScan Backend

Backend Python untuk **DermaScan** — satu FastAPI service yang punya dua sisi:

1. **AI Studio** (web dashboard) — kelola dataset, jalankan training, evaluasi, deploy model.
2. **Prediction API** — endpoint yang dipanggil oleh **Flutter app (DermaScan+)** untuk prediksi.

Sesuai blueprint: retraining bersifat **manual** (di-trigger dari AI Studio), dan prediksi user **tidak pernah** otomatis masuk ke dataset training.

---

## 1. Instalasi

```bash
cd dermascan_backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> `tensorflow` cukup besar (~500MB+). Pastikan koneksi internet stabil saat install
> pertama kali, dan saat training pertama kali (download bobot ImageNet).

## 2. Menjalankan server

```bash
python run.py
```

- AI Studio (web dashboard): **http://127.0.0.1:8000/**
- Dokumentasi API interaktif (Swagger): **http://127.0.0.1:8000/docs**
- Prediction endpoint (dipakai Flutter): **http://127.0.0.1:8000/api/predict**

Kalau testing dari HP/emulator Flutter, ganti `127.0.0.1` dengan IP komputer kamu
di jaringan yang sama (mis. `http://192.168.1.10:8000`), atau IP `10.0.2.2` khusus
untuk Android Emulator.

---

## 3. Alur kerja AI Studio

```
Dataset Manager  →  Training  →  Models & Deployment
  (upload per         (start          (pilih model mana
   class, min.          training,       yang "live" untuk
   5 gambar/kelas)       manual)         Prediction API)
```

1. **Dataset Manager** — 4 cara mengisi dataset, pilih sesuai situasi:

   | Cara | Cocok untuk | Bagaimana |
   |---|---|---|
   | **Upload form** | Puluhan gambar | Pilih banyak file sekaligus (multi-select), submit satu kali |
   | **Import dari CSV/Excel (label sheet)** | Foto berantakan, belum disortir sama sekali | Satu folder isi semua foto campur (boleh nyebar di subfolder) + satu file CSV/Excel 2 kolom (nama file, kelas) → `python scripts/import_from_csv.py --labels label.csv --images ./foto_saya`. **Nggak perlu misahin folder manual sama sekali.** |
   | **Sync From Folder** | Sudah disortir per folder kelas | Taruh gambar langsung ke `data/dataset/<class_label>/`, lalu klik **Sync Dataset** di dashboard (atau `POST /api/dataset/sync`) |
   | **Script HAM10000** | Dataset resmi HAM10000 (~10.000 gambar) | `python scripts/import_ham10000.py --metadata HAM10000_metadata.csv --images HAM10000_images_part_1 HAM10000_images_part_2` — otomatis baca CSV bawaan HAM10000 dan sortir sendiri |

   Dashboard menampilkan jumlah gambar per kelas dan status "ready to train".
2. **Training** — atur epochs / batch size / learning rate, klik **Start Training**.
   Training jalan di background thread (server tetap responsif), progress bisa
   dicek lewat tabel **Training History**.
3. **Models & Deployment** — setiap training yang selesai menghasilkan satu model
   version. Klik **Deploy** untuk menjadikannya model aktif — model inilah yang
   dipakai Prediction API.

Kalau belum ada model yang di-deploy, Prediction API tetap jalan dan mengembalikan
**dummy prediction** (ditandai jelas lewat `is_dummy_prediction: true`), supaya
integrasi Flutter bisa dites dari awal tanpa harus nunggu model asli selesai
dilatih.

---

## 4. Kontrak API untuk Flutter

### `POST /api/predict`

**Request:** `multipart/form-data`, field `image` (file gambar).

```dart
var request = http.MultipartRequest('POST', Uri.parse('$baseUrl/api/predict'));
request.files.add(await http.MultipartFile.fromPath('image', imageFile.path));
var response = await request.send();
```

**Response (200):**

```json
{
  "predicted_class": "mel",
  "short_label": "MEL",
  "full_label": "Melanoma",
  "malignant_potential": "malignant",
  "confidence": 0.8421,
  "all_probabilities": [
    { "class_label": "mel", "short_label": "MEL", "full_label": "Melanoma", "probability": 0.8421 },
    { "class_label": "nv", "short_label": "NV", "full_label": "Melanocytic Nevi", "probability": 0.0912 }
  ],
  "is_dummy_prediction": false,
  "model_version_id": 3,
  "disclaimer": "DermaScan is intended for educational purposes and preliminary screening only. It is not a medical diagnostic tool.",
  "prediction_id": 42,
  "created_at": "2026-07-05T09:15:06.122685"
}
```

Field yang paling relevan buat UI hasil prediksi di Flutter:
- `full_label` + `confidence` → headline hasil scan.
- `malignant_potential` → dipakai buat warna/badge (benign = hijau, pre-malignant =
  kuning, malignant = merah), silakan disesuaikan dengan skema navy/teal app-nya.
- `is_dummy_prediction` → kalau `true`, sebaiknya tampilkan indikator kecil
  "model demo belum dilatih" supaya jelas ke user/juri lomba.
- `disclaimer` → selalu tampilkan di layar hasil, sesuai disclaimer di blueprint.

Field lain yang berguna untuk dataset/riwayat (kalau perlu ditampilkan di
"About Skin Cancer" misalnya): endpoint `GET /api/dataset/classes` mengembalikan
7 kelas HAM10000 beserta urutannya.

---

## 5. Struktur folder

```
dermascan_backend/
├── app/
│   ├── main.py              # entry point FastAPI, daftar semua router
│   ├── config.py            # semua path, konstanta kelas, hyperparameter default
│   ├── database.py          # setup SQLAlchemy (SQLite)
│   ├── models_db.py         # tabel: DatasetImage, TrainingRun, ModelVersion, PredictionLog
│   ├── schemas.py           # kontrak request/response (Pydantic)
│   ├── routers/
│   │   ├── dataset.py       # /api/dataset/*      (AI Studio)
│   │   ├── training.py      # /api/training/*     (AI Studio)
│   │   ├── models.py        # /api/models/*       (AI Studio)
│   │   ├── prediction.py    # /api/predict        (Flutter)
│   │   └── dashboard.py     # /api/dashboard/*    (AI Studio)
│   ├── services/            # logika bisnis dipisah dari router
│   └── ml/
│       ├── preprocessing.py # resize + normalisasi gambar
│       ├── model_builder.py # arsitektur EfficientNetB0
│       ├── train.py         # pipeline training penuh (augmentasi, split, evaluasi, export)
│       └── infer.py         # load model aktif + dummy fallback
├── data/
│   ├── dataset/<class_label>/   # gambar training, per kelas
│   ├── uploads/                 # gambar yang dikirim untuk prediksi (audit trail)
│   ├── models/                  # file .keras hasil training
│   └── dermascan.db             # SQLite (dibuat otomatis saat pertama run)
├── static/                  # AI Studio web dashboard (HTML/CSS/JS, navy/teal theme)
├── scripts/
│   ├── import_from_csv.py   # bulk-import unsorted images using a filename→label sheet (csv/xlsx)
│   └── import_ham10000.py   # bulk-import script for the full HAM10000 dataset
├── requirements.txt
└── run.py
```

---

## 6. Endpoint lengkap

| Endpoint | Method | Dipakai oleh | Keterangan |
|---|---|---|---|
| `/api/health` | GET | keduanya | health check |
| `/api/dataset/classes` | GET | AI Studio + Flutter (opsional) | daftar 7 kelas |
| `/api/dataset/upload` | POST | AI Studio | upload gambar per kelas |
| `/api/dataset/sync` | POST | AI Studio | scan `data/dataset/` dan daftarkan gambar yang ditaruh manual |
| `/api/dataset/stats` | GET | AI Studio | jumlah gambar per kelas, status ready-to-train |
| `/api/dataset/images` | GET | AI Studio | daftar gambar (bisa difilter per kelas) |
| `/api/dataset/images/{id}` | DELETE | AI Studio | hapus gambar |
| `/api/training/start` | POST | AI Studio | mulai training (jalan di background) |
| `/api/training/history` | GET | AI Studio | riwayat semua training run |
| `/api/training/{run_id}` | GET | AI Studio | detail run: classification report, confusion matrix, history kurva |
| `/api/models` | GET | AI Studio | daftar model version hasil training |
| `/api/models/{id}/deploy` | POST | AI Studio | jadikan model ini aktif |
| `/api/models/{id}/download` | GET | AI Studio | download file .keras |
| `/api/predict` | POST | **Flutter** | prediksi satu gambar |
| `/api/dashboard/summary` | GET | AI Studio | ringkasan untuk halaman Dashboard |

---

## 7. Deploy ke Render.com

Backend ini sudah disesuaikan supaya bisa langsung di-deploy ke Render:

- `run.py` otomatis baca port dari env var `$PORT` (wajib di Render, gak boleh hardcode)
- `app/config.py` bisa dikonfigurasi lewat env var `DERMASCAN_DATA_DIR` supaya data
  (dataset, model, database) bisa diarahkan ke **persistent disk**, bukan filesystem
  bawaan yang hilang tiap restart/redeploy
- `render.yaml` sudah disiapkan sebagai Blueprint — Render otomatis baca konfigurasi
  ini kalau kamu deploy lewat "New +" → "Blueprint"

### Yang WAJIB kamu tahu sebelum deploy

| Hal | Penjelasan |
|---|---|
| **Free plan TIDAK bisa pakai persistent disk** | Artinya `data/dataset`, `data/models`, dan `dermascan.db` akan **hilang** tiap kali service restart/redeploy/sleep. Kalau cuma mau demo Prediction API dengan model yang sudah dilatih (dan model file-nya kamu commit ke repo git), free plan masih bisa dipakai. Kalau mau AI Studio-nya (dataset & training) beneran kepakai jangka panjang, wajib pakai **paid plan + disk** (lihat `render.yaml`, sudah dikonfigurasi `plan: starter` + disk 5GB). |
| **RAM 512MB di free plan** | TensorFlow + EfficientNetB0 kemungkinan besar OOM di free tier, terutama pas training. Kalau mau training beneran jalan di Render, minimal pilih instance dengan RAM lebih besar dari starter. |
| **Free plan sleep setelah 15 menit idle** | Kalau training jalan di background thread pas instance-nya sleep karena gak ada HTTP request masuk, training bisa keputus. Kalau serius mau training di cloud, paid plan yang gak sleep. |
| **Build time** | Install `tensorflow` cukup memakan waktu build (~beberapa menit), pastikan gak kehabisan quota build minutes di free plan. |

### Apa yang boleh/jangan di-commit ke Git

`.gitignore` sudah disiapkan supaya:

- ❌ **`data/dataset/`** (foto training) — **JANGAN** ikut di-commit. Bisa ribuan
  file, gak dibutuhin sama sekali buat serving prediksi, cuma bikin repo bengkak
  dan build di Render jadi lama.
- ❌ **`data/uploads/`** dan **`data/*.db`** — data personal/ephemeral, gak perlu ikut.
- ✅ **`data/models/*.keras`** — **BOLEH & disarankan** di-commit. Cuma satu (atau
  beberapa) file, biasanya puluhan MB, dan ini **satu-satunya yang benar-benar
  dibutuhin** Prediction API buat jalan.

Kenapa ini penting khususnya buat Render free tier: karena gak ada persistent disk,
folder `data/` yang ditulis saat runtime (upload, training baru) bakal hilang tiap
restart. Tapi file yang **sudah ada di git repo** ikut ke-deploy ulang setiap kali,
jadi kalau model `.keras`-nya kamu commit, dia otomatis selalu ada.

Backend juga sudah punya startup hook (`app/main.py` → `sync_committed_models`)
yang otomatis scan `data/models/` tiap kali server nyala: kalau ada file `.keras`
yang belum terdaftar di database (kondisi normal di free tier karena DB-nya fresh
tiap restart), langsung didaftarkan dan **otomatis di-deploy** kalau belum ada
model aktif lain. Jadi alurnya:

1. Latih model di lokal seperti biasa lewat AI Studio.
2. Setelah model yang kamu mau muncul di `data/models/`, commit file itu ke git
   (pastikan tidak ke-ignore).
3. Push ke GitHub, Render redeploy otomatis.
4. Begitu server nyala, model itu langsung aktif — gak perlu buka AI Studio dan
   klik Deploy manual lagi tiap kali server restart.

### Langkah deploy

1. Push folder `dermascan_backend/` ini ke repo GitHub.
2. Di dashboard Render: **New +** → **Blueprint** → pilih repo tadi. Render otomatis
   baca `render.yaml`.
3. Kalau mau ubah dari `starter` ke plan lain, atau nonaktifkan disk (kalau memang
   cuma mau demo Prediction API tanpa AI Studio jangka panjang), edit `render.yaml`
   sebelum push.
4. Setelah deploy sukses, AI Studio bisa diakses di URL yang Render kasih (mis.
   `https://dermascan-backend.onrender.com/`), dan Flutter app tinggal ganti
   `baseUrl` ke URL itu.

### Alternatif lebih ringan: pisahkan training dari serving

Kalau khawatir soal RAM/timeout training di Render, cara yang lebih aman: **latih
model di komputer lokal kamu** (pakai AI Studio secara lokal seperti biasa), lalu
cuma commit file `.keras` hasilnya + set sebagai model deployed, dan deploy backend
ke Render hanya untuk **melayani prediksi** (endpoint `/api/predict`) ke Flutter.
Ini jauh lebih ringan buat free/starter plan karena gak perlu nge-train apa-apa di
server.

## 8. Catatan penting

- **Training di background thread**: request `POST /api/training/start` langsung
  return begitu run dibuat (status `pending`/`running`), tidak nunggu training
  selesai. AI Studio dashboard perlu polling `/api/training/history` atau
  `/api/training/{id}` untuk lihat progress/hasil akhir.
- **Satu model aktif**: hanya satu `ModelVersion` yang `is_deployed=True` di satu
  waktu. Setiap kali deploy model baru, model lama otomatis nonaktif dan cache
  model di memori langsung di-refresh.
- **Dummy prediction**: dipakai otomatis kalau belum ada model yang di-deploy.
  Selalu ditandai `is_dummy_prediction: true` — jangan pernah disamakan dengan
  hasil model asli di UI.
- **SQLite → PostgreSQL/MySQL**: tinggal ganti `DATABASE_URL` di `app/config.py`,
  seluruh kode router/service tidak perlu diubah karena semua akses lewat
  SQLAlchemy session.
- Sudah dites end-to-end di sandbox: upload dataset ✅, dummy prediction ✅, training
  pipeline (tf.data + augmentasi + fit + save/load) ✅. Download bobot ImageNet
  butuh akses internet normal saat run pertama kali di komputer kamu.
