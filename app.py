import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import openpyxl  # Required for pd.read_excel

# -------------------------------------------------------------------
# Page Setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Automated ML Model Builder",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Automated ML Model Builder")
st.write("Upload your data, select what you want to predict, and this app will automatically build a model for you.")

# -------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------
@st.cache_data
def load_data(file):
    """Loads data from CSV or Excel file."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            st.error("Error: Invalid file format. Please upload a .csv or .xlsx file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def build_model(df, target_column, problem_type, ignore_cols):
    """Builds and evaluates a model based on user inputs."""
    
    # 1. Define Features (X) and Target (y)
    st.write("### 1. Preparing Data")
    try:
        y = df[target_column]
        X = df.drop(columns=[target_column] + ignore_cols)
        st.write(f"**Target variable (y):** `{target_column}`")
        st.write(f"**Ignored columns:** `{ignore_cols}`")
        st.write(f"**Feature columns (X):** {X.columns.tolist()}")
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return

    # 2. Automatic Feature Detection
    st.write("### 2. Detecting Feature Types")
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    st.info(f"**Found {len(numeric_features)} numerical features:**\n{numeric_features}")
    st.info(f"**Found {len(categorical_features)} categorical features:**\n{categorical_features}")

    # 3. Create Preprocessing Pipelines
    # (We add SimpleImputer to handle any missing data automatically)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 4. Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 5. Select Model Based on Problem Type
    if problem_type == "Classification":
        model = RandomForestClassifier(random_state=42)
    else:  # Regression
        model = RandomForestRegressor(random_state=42)

    # 6. Create the full ML Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # 7. Split data and train model
    st.write("### 3. Training Model")
    with st.spinner("Splitting data and training model... This may take a moment."):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Handle potential data type issues in target
        if problem_type == "Classification":
            y_train = y_train.astype(str)
            y_test = y_test.astype(str)
            
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)
    st.success("Model training complete! ✅")

    # 8. Evaluate Model and Show Results
    st.write("### 4. Model Evaluation")
    if problem_type == "Classification":
        accuracy = accuracy_score(y_test, y_pred)
        st.metric(label="**Accuracy**", value=f"{accuracy * 100:.2f}%")
        st.text("Classification Report:")
        st.code(classification_report(y_test, y_pred))
    
    else: # Regression
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        st.metric(label="**R-squared (R2)**", value=f"{r2:.4f}")
        st.metric(label="**Root Mean Squared Error (RMSE)**", value=f"{rmse:.4f}")

    st.write("### 5. Sample Predictions")
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    st.dataframe(results_df.head(10))

# -------------------------------------------------------------------
# Main App Interface
# -------------------------------------------------------------------

# --- 1. File Uploader ---
uploaded_file = st.file_uploader("Upload your data (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.success(f"File '{uploaded_file.name}' loaded successfully. Found {df.shape[0]} rows and {df.shape[1]} columns.")
        st.dataframe(df.head())
        
        # --- 2. User Configuration ---
        st.header("Model Configuration")
        
        all_columns = df.columns.tolist()
        
        target_column = st.selectbox(
            "Which column do you want to predict? (Target Variable)",
            options=all_columns,
            index=None,
            help="Select the column the model should learn to predict."
        )
        
        problem_type = st.selectbox(
            "What type of problem is this?",
            options=["Classification", "Regression"],
            help="**Classification:** Predicts a category (e.g., 'Yes'/'No', 'Red'/'Green').\n\n**Regression:** Predicts a number (e.g., price, temperature)."
        )

        ignore_cols = st.multiselect(
            "Select any columns to ignore (e.g., ID, Name, Phone Number)",
            options=[col for col in all_columns if col != target_column],
            help="Select columns that are not useful for prediction."
        )

        # --- 3. Run Button ---
        if st.button("**Build Model**", type="primary"):
            if target_column is None:
                st.error("Please select a target variable to predict.")
            else:
                build_model(df, target_column, problem_type, ignore_cols)