from data_preprocessing import load_data, missing_values_table, handle_missing_values, fill_missing_values_with_median
from feature_engineering import create_new_features, encode_features, scale_features
from model import train_model, evaluate_model
from utils import check_df, grab_col_names
from sklearn.model_selection import train_test_split

# Veri yükleme
df = load_data("diabetes.csv")
check_df(df)

# Eksik değer analizi ve doldurma
zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
df = handle_missing_values(df, zero_columns)
na_columns = missing_values_table(df, na_name=True)
df = fill_missing_values_with_median(df, zero_columns)

# Özellik çıkarımı
df = create_new_features(df)

# Kategorik ve sayısal değişkenlerin ayrıştırılması
cat_cols, num_cols, _ = grab_col_names(df)
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Outcome"]]

# Encoding ve standardization
df = encode_features(df, binary_cols, cat_cols)
df = scale_features(df, num_cols)

# Model kurma ve değerlendirme
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)
