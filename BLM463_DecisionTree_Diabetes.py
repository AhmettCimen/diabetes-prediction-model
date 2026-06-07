
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, ConfusionMatrixDisplay
)

import warnings
warnings.filterwarnings('ignore')


# 1. VERİ YÜKLEME

df = pd.read_csv('Diabetes_type.csv')

print("=" * 55)
print("VERİ SETİ GENEL BİLGİLERİ")
print("=" * 55)
print(f"Satır  x  Sütun : {df.shape}")
print(f"Sütunlar        : {list(df.columns)}")
print(f"\nHedef Dağılımı:\n{df['Type-2 Diabetic'].value_counts()}")
print(f"\nEksik Değerler:\n{df.isnull().sum()}")
print(f"\nİstatistiksel Özet:\n{df.describe().round(2)}")

# 2. ÖN İŞLEME



X = df.drop(columns=['Type-2 Diabetic'])
y = df['Type-2 Diabetic']

feature_names = list(X.columns)
class_names   = ['Non-Diabetic', 'Diabetic']

# 2b. Eksik değerleri medyan ile doldur
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=feature_names)

# 2c. Eğitim / Test bölmesi  (%80 / %20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nEğitim seti : {X_train.shape[0]} örnek")
print(f"Test seti   : {X_test.shape[0]} örnek")

# ─────────────────────────────────────────────────────────────
# 3. MODELİ KURMA VE EĞİTME
# ─────────────────────────────────────────────────────────────

model = DecisionTreeClassifier(
    criterion   = 'gini',      # Gini impurity
    max_depth   = 5,           # Aşırı öğrenmeyi önlemek için derinlik sınırı
    min_samples_split = 10,    # Bölünmek için gereken minimum örnek sayısı
    min_samples_leaf  = 5,     # Yaprakta minimum örnek
    random_state = 42
)

model.fit(X_train, y_train)
print("\nModel eğitimi tamamlandı ✓")

# ─────────────────────────────────────────────────────────────
# 4. TAHMİN
# ─────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]   # pozitif sınıf olasılığı

# ─────────────────────────────────────────────────────────────
# 5. METRİKLER
# ─────────────────────────────────────────────────────────────

accuracy    = accuracy_score(y_test, y_pred)
precision   = precision_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)                    # = recall
specificity = recall_score(y_test, y_pred, pos_label=0)
f1          = f1_score(y_test, y_pred)
auc         = roc_auc_score(y_test, y_prob)

# 10-Katlı Çapraz Doğrulama
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print("\n" + "=" * 55)
print("DECISION TREE — PERFORMANS METRİKLERİ")
print("=" * 55)
print(f"  Accuracy             : {accuracy * 100:.2f}%")
print(f"  Precision            : {precision * 100:.2f}%")
print(f"  Sensitivity (Recall) : {sensitivity * 100:.2f}%")
print(f"  Specificity          : {specificity * 100:.2f}%")
print(f"  F1-Score             : {f1 * 100:.2f}%")
print(f"  AUC-ROC              : {auc * 100:.2f}%")
print(f"  10-Fold CV Accuracy  : {cv_scores.mean() * 100:.2f}% (±{cv_scores.std() * 100:.2f}%)")
print("=" * 55)

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ─────────────────────────────────────────────────────────────
# 6. AĞAÇ KURALLARI (metin çıktısı)
# ─────────────────────────────────────────────────────────────

print("\nKarar Ağacı Kuralları (ilk 3 seviye):")
print(export_text(model, feature_names=feature_names, max_depth=3))

# ─────────────────────────────────────────────────────────────
# 7. GÖRSELLEŞTİRMELER
# ─────────────────────────────────────────────────────────────
os.makedirs('img', exist_ok=True)

# ── Şekil 1: Karar Ağacı Yapısı ─────────────────────────────
plt.figure(figsize=(24, 10))
plot_tree(
    model,
    feature_names = feature_names,
    class_names   = class_names,
    filled        = True,
    rounded       = True,
    fontsize      = 8,
    impurity      = True,
    precision     = 2
)
plt.title('Karar Ağacı Yapısı (max_depth=5, Gini)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('img/dt_agac_yapisi.png', dpi=150, bbox_inches='tight')
plt.show()
print("Şekil 1 kaydedildi → img/dt_agac_yapisi.png")

# ── Şekil 2: Karışıklık Matrisi ──────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sol: sayısal
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Karışıklık Matrisi (Sayısal)', fontweight='bold')

# Sağ: yüzdesel
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1], cbar=False)
axes[1].set_title('Karışıklık Matrisi (%)', fontweight='bold')
axes[1].set_xlabel('Tahmin Edilen')
axes[1].set_ylabel('Gerçek')

plt.tight_layout()
plt.savefig('img/dt_karisiklik_matrisi.png', dpi=150, bbox_inches='tight')
plt.show()
print("Şekil 2 kaydedildi → img/dt_karisiklik_matrisi.png")

# ── Şekil 3: ROC Eğrisi ─────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='royalblue', lw=2,
         label=f'Decision Tree (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Rastgele Sınıflandırıcı')
plt.fill_between(fpr, tpr, alpha=0.1, color='royalblue')
plt.xlabel('Yanlış Pozitif Oranı (FPR)', fontsize=11)
plt.ylabel('Doğru Pozitif Oranı (TPR)', fontsize=11)
plt.title('ROC Eğrisi — Decision Tree', fontsize=13, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('img/dt_roc_egrisi.png', dpi=150, bbox_inches='tight')
plt.show()
print("Şekil 3 kaydedildi → img/dt_roc_egrisi.png")

# ── Şekil 4: Özellik Önem Dereceleri ────────────────────────
importances = model.feature_importances_
indices     = np.argsort(importances)

plt.figure(figsize=(9, 6))
plt.barh(
    [feature_names[i] for i in indices],
    importances[indices],
    color=plt.cm.Blues(np.linspace(0.4, 0.9, len(indices)))
)
plt.xlabel('Önem Derecesi (Gini)', fontsize=11)
plt.title('Özellik Önem Dereceleri — Decision Tree', fontsize=13, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('img/dt_ozellik_onem.png', dpi=150, bbox_inches='tight')
plt.show()
print("Şekil 4 kaydedildi → img/dt_ozellik_onem.png")

# ── Şekil 5: 10-Katlı Çapraz Doğrulama Sonuçları ────────────
plt.figure(figsize=(9, 5))
fold_labels = [f'Fold {i+1}' for i in range(len(cv_scores))]
bars = plt.bar(fold_labels, cv_scores * 100, color='steelblue', edgecolor='white')
plt.axhline(y=cv_scores.mean() * 100, color='red', linestyle='--',
            label=f'Ortalama: {cv_scores.mean()*100:.2f}%')
for bar, val in zip(bars, cv_scores):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8)
plt.ylim(80, 105)
plt.ylabel('Accuracy (%)', fontsize=11)
plt.title('10-Katlı Çapraz Doğrulama Sonuçları', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('img/dt_cross_validation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Şekil 5 kaydedildi → img/dt_cross_validation.png")

# ── Şekil 6: Metrik Özet Çubuğu ─────────────────────────────
metric_names  = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
metric_values = [accuracy, precision, sensitivity, specificity, f1, auc]
colors_bar = ['#2563EB','#7C3AED','#059669','#D97706','#DC2626','#0891B2']

plt.figure(figsize=(10, 5))
bars = plt.bar(metric_names, [v*100 for v in metric_values],
               color=colors_bar, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, metric_values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.4,
             f'{val*100:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.ylim(60, 110)
plt.ylabel('Skor (%)', fontsize=11)
plt.title('Decision Tree — Tüm Metrik Sonuçları', fontsize=13, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('img/dt_metrik_ozet.png', dpi=150, bbox_inches='tight')
plt.show()
print("Şekil 6 kaydedildi → img/dt_metrik_ozet.png")

print("\n✅ Tüm görseller kaydedildi.")
print("\n📊 ÖZET SONUÇLAR:")
print(f"   Accuracy      : %{accuracy*100:.2f}")
print(f"   Precision     : %{precision*100:.2f}")
print(f"   Sensitivity   : %{sensitivity*100:.2f}")
print(f"   Specificity   : %{specificity*100:.2f}")
print(f"   F1-Score      : %{f1*100:.2f}")
print(f"   AUC-ROC       : %{auc*100:.2f}")
print(f"   10-Fold CV    : %{cv_scores.mean()*100:.2f}")
