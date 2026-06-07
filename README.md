# Tip-2 Diyabet Tahmini - Decision Tree Sınıflandırması

## 1. Projenin Amacı ve Veri Seti Seçimi
Bu kapsamda, "Data in Brief" dergisinde yayınlanan Bangladeş menşeli Tip-2 Diyabet Dataset'i seçilmiş ve **Decision Tree** yöntemi ile hastalık tahmini gerçekleştirilmiştir.

## 2. Veri Seti İncelemesi ve Ön İşleme
Kullanılan Dataset, 1065 hastaya (840 diyabetik, 225 diyabetik olmayan) ait klinik, demografik ve biyokimyasal Feature'lar içermektedir. Veri setindeki Feature'ların anlamları kısaca şu şekildedir:

* **No. of Pregnancy:** Hamilelik Sayısı
* **Age:** Yaş
* **BMI:** Vücut Kitle İndeksi
* **BP(Systolic):** Sistolik Kan Basıncı
* **BP(Diastolic):** Diyastolik Kan Basıncı
* **Diabetes Pedigree Function:** Diyabet Soyağacı Fonksiyonu (Genetik Risk)
* **Insulin:** İnsülin Seviyesi
* **Skin Thickness(mm):** Cilt Kalınlığı
* **Glucose:** Glikoz (Kan Şekeri) Seviyesi

Dataset üzerinde modele hazırlık aşamasında aşağıdaki Preprocessing adımları sırasıyla uygulanmıştır:

1. **Keşifsel Veri Analizi (EDA):** İlk adım olarak veri setinin genel yapısı (1065 satır, 9 sütun), eksik (null) değer sayıları ve istatistiksel özetleri (describe) incelenmiştir. Hedef değişken (Type-2 Diabetic) dağılımı kontrol edilerek Class dengesizliği tespit edilmiştir.
2. **Girdi ve Hedef Ayrımı:** Veri seti, bağımsız değişkenler (X) ve hedef değişken (y) olarak ikiye ayrılmıştır.
3. **Missing Value Imputation (Eksik Veri Doldurma):** Veri setindeki eksik değerler (örneğin diyastolik kan basıncı ve cilt kalınlığındaki boşluklar), veri setindeki olası Outlier'lardan (aykırı değerler) etkilenmemesi amacıyla Scikit-learn kütüphanesindeki `SimpleImputer` sınıfı ile Median (medyan) yöntemi kullanılarak doldurulmuştur.
4. **Train/Test Split:** Modelin eğitilmesi ve bağımsız bir veri üzerinde test edilmesi için Dataset %80 Train ve %20 Test olacak şekilde ikiye ayrılmıştır. Veri setindeki Class'ların dengesiz olması nedeniyle Scikit-learn içerisindeki `train_test_split` fonksiyonunda `stratify=y` parametresi kullanılmış, bu sayede diyabetik ve sağlıklı hasta oranlarının Train ve Test setlerine eşit dağılması güvence altına alınmıştır. Ayrıca deneylerin tekrarlanabilir olması için `random_state=42` parametresi atanmıştır.

## 3. Sınıflandırma Yöntemi: Decision Tree
Proje kapsamında kural tabanlı yapısı sayesinde yorumlanabilirliği yüksek olan Decision Tree yöntemi benimsenmiştir. Overfitting'i engellemek ve modelin genelleme yeteneğini artırmak için model Hyperparameter'ları aşağıdaki gibi ayarlanmıştır:

* **Criterion:** Gini (Bölünmelerde saflık derecesini hesaplamak için)
* **Max Depth (`max_depth`):** 5
* **Min Samples Split (`min_samples_split`):** 10
* **Min Samples Leaf (`min_samples_leaf`):** 5

Aşağıdaki görselde, Train verisiyle kurulan Decision Tree'nin mantıksal yapısı ve Root Node'dan Leaf Node'lara doğru dallanma süreci görülmektedir:

![Karar Ağacı Yapısı](img/dt_agac_yapisi.png)

## 4. Modelin Eğitimi, Performans Metrikleri ve Görselleştirmeler
Model Train verisi ile eğitildikten sonra, Test verisi üzerinde Prediction yapılmış ve projenin isterleri doğrultusunda çeşitli performans ölçütleriyle değerlendirilmiştir.

Hesaplanan temel metrikler (Accuracy, Precision, Sensitivity, Specificity, F1-Score ve AUC-ROC) grafikte özetlenmiştir:

![Tüm Metrik Sonuçları](img/dt_metrik_ozet.png)

### Confusion Matrix
Modelin hangi Class'ları doğru (True Positive, True Negative), hangilerini yanlış tahmin ettiğini (False Positive, False Negative) detaylı olarak görebilmek için Confusion Matrix oluşturulmuştur:

![Karışıklık Matrisi](img/dt_karisiklik_matrisi.png)

### ROC Curve ve AUC
Modelin sınıflandırma kabiliyetini ve farklı Threshold değerlerindeki performansını gösteren ROC Curve çizdirilmiştir. AUC değerinin yüksekliği modelin başarılı bir ayırma kapasitesine sahip olduğunu göstermektedir:

![ROC Eğrisi](img/dt_roc_egrisi.png)

### Feature Importance
Modelin diyabet teşhisi koyarken Dataset'teki hangi değişkenleri daha fazla dikkate aldığı incelenmiştir. Decision Tree algoritmasına göre, tahminlemede Glikoz (Glucose) seviyesinin en önemli Feature olduğu tespit edilmiştir:

![Özellik Önem Dereceleri](img/dt_ozellik_onem.png)

### 10-Fold Cross Validation
Modelin tek bir Train/Test bölmesine bağımlı olmadığını kanıtlamak için 10-Fold Cross Validation kullanılmıştır. Model 10 farklı iterasyonda değerlendirilerek genel başarısı ve kararlılığı kanıtlanmıştır:

![10-Katlı Çapraz Doğrulama Sonuçları](img/dt_cross_validation.png)

## 5. Önceki Çalışmalarla Karşılaştırma
Geliştirdiğimiz "Decision Tree" modeli, aynı Dataset üzerinde literatürde çok yakın zamanda yapılan "A Healthcare-Oriented Machine Learning Framework for Early Detection of Type 2 Diabetes" (Alam vd., 2026) isimli akademik çalışmanın metodolojisi ve sonuçları ile kıyaslanmıştır.

### Projemiz ile Referans Çalışma Arasındaki Temel Farklılıklar

**Model Tercihi ve Çeşitliliği**
* **Bizim Çalışmamız:** Yalnızca Decision Tree modeli kullanılmış olup, modelin nasıl kararlar aldığını (kuralları) görselleştirmek ve yorumlanabilirlik düzeyini yüksek tutmak hedeflenmiştir.
* **Referans Makale:** Decision Tree de dahil olmak üzere toplam 8 farklı Machine Learning modeli (Random Forest, XGBoost, SVM vb.) test edilmiş ve karşılaştırılmıştır. En yüksek başarı, bir Ensemble (topluluk) modeli olan Random Forest ile elde edilmiştir.

**Veri Dengesizliği (Data Imbalance) Çözümleri**
* **Bizim Çalışmamız:** Class dağılımındaki dengesizliğe (Diabetic: 840, Non-Diabetic: 225) algoritmik olarak müdahale edilmemiş; sadece Train ve Test verisi ayrılırken `stratify` parametresi kullanılarak mevcut oranların birebir korunması sağlanmıştır.
* **Referans Makale:** Azınlık Class'ını çoğaltmak ve dengelemek adına SMOTE, ADASYN, Random Oversampling ve Undersampling gibi 5 farklı sentetik veri üretme ve Data Balancing (veri dengeleme) tekniği kullanılmış, bu sayede modelin başarısı yapay olarak artırılmıştır.

**Hiperparametre Optimizasyonu (Hyperparameter Tuning)**
* **Bizim Çalışmamız:** Decision Tree modelinin parametreleri (`max_depth=5`, `criterion='gini'`) olarak belirlenmiş ve Overfitting (aşırı öğrenme) riskini engelleyecek sade, şeffaf bir yapı kurulmuştur.
* **Referans Makale:** En iyi model performansına ulaşmak amacıyla GridSearchCV algoritması kullanılarak kapsamlı ve otomatik bir Hyperparameter Tuning süreci yürütülmüştür.

### Performans (Metrik) Karşılaştırması

Her iki çalışmada da ortak olarak kullanılan temel Classification Metric'leri aşağıda yan yana sunulmuştur:

| Metrik | Referans Model (Random Forest - Tuned) | Bizim Modelimiz (Decision Tree) |
| :--- | :--- | :--- |
| **Accuracy** | %98.21 | %96.24 |
| **Precision** | %98.47 | %95.98 |
| **F1-Score** | %98.43 | %97.66 |
| **AUC-ROC** | %98.00 (0.98) | %93.51 |

**Performans Karşılaştırması Devamı:**
Makalede, sentetik veri üretiminin (Random Oversampling) kullanıldığı optimize senaryoda Random Forest ile %99.50 oranında bir maksimum başarı bildirilmiştir. Makaledeki diğer modellere (KNN: %78-87, SVM: %94) kıyasla bizim projemizde uygulanan Decision Tree modeli; oversampling uygulanmamasına ve çok daha sade bir yapıda olmasına rağmen %96.24 Accuracy ve %97.66 F1-Score gibi oldukça güçlü ve rekabetçi seviyelere ulaşmıştır.

### Genel Değerlendirme ve Çıkarım
Akademik çalışmada elde edilen yüksek skorların (%98-99) temelinde Dataset'in aşırı derecede daraltılması (485 örneğe düşürülmesi) ve yoğun sentetik veri üretimi (oversampling) yatmaktadır. Gerçek dünya verilerindeki değişkenlikler göz önüne alındığında, bu durum modelin Overfitting yatkınlığını artırıp genellenebilirliğini (Generalization) sınırlandırabilir. Diğer yandan bizim geliştirdiğimiz model, orijinal ve daha büyük Dataset (1065 örnek) üzerinde eğitilerek kural tabanlı, yorumlanılabilirlik özelliği yüksek bir yapıda kurulmuştur. Elde ettiğimiz %96'lık Accuracy oranı, klinik kullanıma uygunluk ve gerçek dünyaya uygulanılabilirlik açısından son derece pratik ve akademik sonuçlarla doğrudan rekabet edebilecek bir güvenilirlik sunmaktadır.

## 6. Sonuç ve Değerlendirme
Projede hedeflendiği üzere, dergiden tıbbi bir Dataset edinilmiş ve "Decision Tree" yöntemi uygulanarak bir Classification algoritması uçtan uca tasarlanmıştır. Gerekli Missing Value atamaları yapılmış, model Overfitting'e karşı optimize edilmiş ve proje dokümanında istenen Accuracy, Sensitivity, Specificity, F1-Score gibi tüm metrikler çeşitli grafiklerle desteklenerek zenginleştirilmiştir.

Proje kapsamında izlenen mantıksal akış; veri yükleme -> Preprocessing -> Hyperparameter ayarlı model eğitimi -> metriklerin hesaplanması -> veri görselleştirme adımlarından oluşmuş olup, sonuçlar literatürdeki diğer akademik çalışmalar ile de kıyaslanmıştır.
