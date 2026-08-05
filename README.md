# Workflow CI – Muhammad Fathurrohman

## Deskripsi

Repository ini berisi MLflow Project dan GitHub Actions Workflow untuk Kriteria 3 – CI/CD Pipeline klasifikasi gestur tangan BISINDO (Bahasa Isyarat Indonesia).

## Struktur Repository

```
Workflow-CI_Muhammad-Fathurrohman/
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions workflow
├── MLProject/
│   ├── MLProject           # MLflow Project definition
│   ├── conda.yaml          # Environment dependencies
│   ├── modelling.py        # Training script
│   └── train_processed.csv # Dataset
└── README.md
```

## GitHub Actions Workflow

Workflow CI otomatis berjalan saat:
- Push ke branch `main`
- Pull request ke branch `main`
- Manual trigger (workflow_dispatch)

### Tahapan Workflow

1. **Checkout** – Clone repository
2. **Setup Python 3.10** – Konfigurasi Python environment
3. **Install dependencies** – Install semua library yang dibutuhkan
4. **Run MLflow Project** – Jalankan training model Random Forest
5. **Upload artifacts** – Simpan model dan hasil ke GitHub Actions artifacts

## Model

- **Algoritma**: Random Forest Classifier
- **Feature Extraction**: HOG (Histogram of Oriented Gradients)
- **Dataset**: BISINDO – 40 kelas kata bahasa isyarat Indonesia
- **Tracking**: MLflow (lokal)

## Cara Menjalankan Lokal

```bash
cd MLProject
python modelling.py \
  --n_estimators 200 \
  --max_depth None \
  --min_samples_split 5 \
  --max_features sqrt \
  --test_size 0.2 \
  --random_state 42
```

## Cara Menjalankan via MLflow

```bash
mlflow run MLProject/ -P n_estimators=200 -P max_depth=None
```

## Author

Muhammad Fathurrohman
