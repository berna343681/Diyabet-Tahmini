

# Diyabet Tahmin Projesi

Bu proje, makine öğrenimi ve özellik mühendisliği kullanarak diyabet hastalığını tahmin etmeyi amaçlamaktadır. **Random Forest Classifier** ile model oluşturulmuş ve çeşitli sağlık ölçümleri kullanılarak diyabet tahmini yapılmıştır.

## Veri Seti

- **Kaynak**: [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
- **Sütunlar**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome

## Özellik Mühendisliği

Veri setine yeni özellikler eklenmiş ve eksik değerler işlenmiştir. Yaş, BMI ve glukoz seviyelerine göre kategoriler oluşturulmuştur.

## Modelleme

- **Model**: RandomForestClassifier
- **Başarım Metrikleri**: Accuracy, Recall, Precision, F1-Score, AUC

## Kurulum ve Çalıştırma

1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/kullanıcı-adı/diyabet-tahmini.git
   cd diyabet-tahmini
   ```
2. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
3. Ana dosyayı çalıştırın:
   ```bash
   python main.py
   ```