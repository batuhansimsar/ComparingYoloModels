# 🚗 YOLO License Plate Detection — Model Comparison

Bu repo, aynı veri seti (`dataset_500` — 500 görsel) ve aynı eğitim parametreleri (30 epoch, batch 8, imgsz 640) kullanılarak eğitilen üç farklı YOLO11 modelinin plaka tespiti sonuçlarını karşılaştırmaktadır.

---

## 📦 Veri Seti

**Kaynak:** [Roboflow Universe — License Plate Recognition](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e) (Version 4)

Orijinal veri seti binlerce görsel içermektedir. Bu çalışmada **adil ve hızlı bir karşılaştırma** yapabilmek amacıyla veri setinden **rastgele 500 görsel** seçilerek `dataset_500` alt kümesi oluşturulmuştur. Her üç model de tamamen aynı bu 500 görsellik alt kümeyle eğitilmiştir.

### Veri Setini İndirme

```python
from roboflow import Roboflow

API_KEY = "yVAoRyBc3LTryAZFrECk"
rf = Roboflow(api_key=API_KEY)
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")

# Tam veri setini indir (YOLOv8 formatında)
dataset = project.version(4).download("yolov8", location="./dataset")
```

> ⚠️ **Not:** Yukarıdaki kod tam veri setini indirir. Bu çalışmada kullanılan `dataset_500`, indirilen bu tam veri setinden **rastgele 500 görsel seçilerek** oluşturulmuştur:
>
> ```python
> import os, random, shutil
>
> random.seed(42)  # Tekrar üretilebilirlik için
> all_images = os.listdir("./dataset/train/images")
> selected = random.sample(all_images, 500)
>
> os.makedirs("./dataset_500/train/images", exist_ok=True)
> os.makedirs("./dataset_500/train/labels", exist_ok=True)
>
> for img in selected:
>     shutil.copy(f"./dataset/train/images/{img}", f"./dataset_500/train/images/{img}")
>     label = img.replace(".jpg", ".txt")
>     shutil.copy(f"./dataset/train/labels/{label}", f"./dataset_500/train/labels/{label}")
> ```

| Özellik | Değer |
|---|---|
| Kaynak | Roboflow Universe |
| Proje | license-plate-recognition-rxg4e |
| Versiyon | 4 |
| Format | YOLOv8 |
| Toplam Görsel (Subset) | **500** (rastgele seçim) |
| Görev | Object Detection (Plaka Tespiti) |

---

## 📊 Sonuç Özeti

| Model | Epoch (Tamamlanan) | Precision | Recall | mAP@50 | mAP@50-95 | Parametre Sayısı |
|---|---|---|---|---|---|---|
| 🥇 **YOLO11n** | 22 *(early stop)* | 0.920 | **0.909** | **0.976** | 0.525 | ~2.6M |
| 🥈 YOLO11m Run 2 | 30 | 0.926 | 0.843 | 0.905 | **0.542** | ~20.1M |
| 🥉 YOLO11m Run 1 | 29 | **0.947** | 0.806 | 0.877 | 0.509 | ~20.1M |

> **🏆 Kazanan: YOLO11n** — Hem en yüksek mAP@50 (%97.6) hem de en yüksek Recall değerine sahip. 22 epoch'ta early stopping ile durmasına rağmen diğer modelleri geçti. Hem daha hızlı eğitildi, hem de daha hafif bir mimari sundu.

---

## 🏆 En İyi Model: YOLO11n

### Neden YOLO11n kazandı?

**mAP@50 = 0.976** → Bu, modelin plakaları %97.6 oranında doğru tespit ettiği anlamına gelir.

- ✅ **Recall 0.909** — Görseldeki plakaların %90.9'unu kaçırmadan buldu (YOLO11m'e göre +10 puan)
- ✅ **mAP50 0.976** — Tüm modeller arasında en yüksek değer (+7 puan fark)
- ✅ **22 epoch'ta converge** — Daha az eğitimle daha iyi sonuç → verimlilik
- ✅ **~2.6M parametre** — YOLO11m'in (%20M) çok daha altında → gerçek zamanlı uygulamalar için idealdir
- ✅ **Küçük model boyutu** — Edge cihazlar, Raspberry Pi, mobil uygulamalar için uygun

### YOLO11n Eğitim Grafikleri

![YOLO11n Eğitim Sonuçları](images/yolo11n/results.png)

| F1 Curve | PR Curve |
|---|---|
| ![YOLO11n F1](images/yolo11n/BoxF1_curve.png) | ![YOLO11n PR](images/yolo11n/BoxPR_curve.png) |

| Confusion Matrix | Confusion Matrix (Normalized) |
|---|---|
| ![YOLO11n CM](images/yolo11n/confusion_matrix.png) | ![YOLO11n CM Norm](images/yolo11n/confusion_matrix_normalized.png) |

**Validation tahminleri:**

![YOLO11n Val Predictions](images/yolo11n/val_batch0_pred.jpg)

---

## 🔵 YOLO11m — Run 1

> Aynı `dataset_500` ile 29 epoch. Recall düşük kalmış (%80.6), validation sırasında birkaç epoch'ta `nan` loss yaşandı (muhtemelen büyük batch instability).

### Eğitim Grafikleri

![YOLO11m Run1 Eğitim Sonuçları](images/yolo11m_run1/results.png)

| F1 Curve | PR Curve |
|---|---|
| ![Run1 F1](images/yolo11m_run1/BoxF1_curve.png) | ![Run1 PR](images/yolo11m_run1/BoxPR_curve.png) |

| Confusion Matrix | Confusion Matrix (Normalized) |
|---|---|
| ![Run1 CM](images/yolo11m_run1/confusion_matrix.png) | ![Run1 CM Norm](images/yolo11m_run1/confusion_matrix_normalized.png) |

**Validation tahminleri:**

![YOLO11m Run1 Val](images/yolo11m_run1/val_batch0_pred.jpg)

---

## 🟣 YOLO11m — Run 2

> Aynı konfigürasyonla tekrar çalıştırılan YOLO11m eğitimi. Recall ve mAP değerleri Run 1'e göre belirgin şekilde iyileşti. Parametre optimizasyonu açısından Run 1'den daha kararlı bir eğitim süreci geçirdi.

### Eğitim Grafikleri

![YOLO11m Run2 Eğitim Sonuçları](images/yolo11m_run2/results.png)

| F1 Curve | PR Curve |
|---|---|
| ![Run2 F1](images/yolo11m_run2/BoxF1_curve.png) | ![Run2 PR](images/yolo11m_run2/BoxPR_curve.png) |

| Confusion Matrix | Confusion Matrix (Normalized) |
|---|---|
| ![Run2 CM](images/yolo11m_run2/confusion_matrix.png) | ![Run2 CM Norm](images/yolo11m_run2/confusion_matrix_normalized.png) |

**Validation tahminleri:**

![YOLO11m Run2 Val](images/yolo11m_run2/val_batch0_pred.jpg)

---

## ⚙️ Eğitim Konfigürasyonu

| Parametre | YOLO11n | YOLO11m Run 1 | YOLO11m Run 2 |
|---|---|---|---|
| Base Model | `yolo11n.pt` | `yolo11m.pt` | `yolo11m.pt` |
| Dataset | `dataset_500` | `dataset_500` | `dataset_500` |
| Max Epochs | 30 | 30 | 30 |
| Tamamlanan Epoch | 22 | 29 | 30 |
| Batch Size | 8 | 8 | 8 |
| Image Size | 640 | 640 | 640 |
| Optimizer | Auto | Auto | Auto |
| Early Stopping Patience | 10 | 10 | 10 |
| Cache | RAM | RAM | RAM |
| Device | GPU (Colab) | GPU (Colab) | GPU (Colab) |

---

## 💡 Sonuç ve Öneriler

### Kullanım Senaryosu Bazlı Öneri

| Senaryo | Önerilen Model |
|---|---|
| 🚀 Gerçek zamanlı / Edge cihaz | **YOLO11n** |
| 💻 Yüksek doğruluk / sunucu taraflı | **YOLO11m Run 2** |
| 🔬 Araştırma & Daha fazla epoch | YOLO11m (100+ epoch dene) |

### Neden YOLO11m tam potansiyeline ulaşamadı?
1. **500 görsel az**: YOLO11m gibi büyük bir model, daha fazla veri gerektirir (önerilir: 5000+)
2. **30 epoch yetersiz**: Büyük modeller daha fazla epoch'ta stabilize olur (önerilir: 100+)
3. **nan loss sorunu**: Run 1'de bazı epoch'larda validation loss `nan` döndü → eğitim instability işareti

> **Sonuç:** Mevcut konfigürasyonla **YOLO11n açık ara en iyi modeldir**. YOLO11m'in gerçek potansiyelini görmek için tam veri seti + 100 epoch denenmesi önerilir.
