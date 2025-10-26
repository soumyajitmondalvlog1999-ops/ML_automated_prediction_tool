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

def build_model(df, target_column, problem_type, ignore_cols):
    """Builds and evaluates a model based on user inputs."""
    
    st.write("### 1. Preparing Data")
    
    # --- 1. Define Features (X) and Target (y) ---
    try:
        # --- Data Cleaning (Automatic) ---
        for col in df.columns:
            if col not in ignore_cols and col != target_column:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass # Keep it as an object
        
        y = df[target_column]
        X = df.drop(columns=[target_column] + ignore_cols)
        
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return

    # --- 2. Sanity Check for Classification ---
    if problem_type == "Classification":
        y = y.astype(str).str.strip().str.replace('.', '', regex=False)
        unique_classes = y.nunique()
        if unique_classes > 50:
            st.error(f"""
            **Configuration Error!**
            
            You selected **Classification**, but your target column ('{target_column}')
            has **{unique_classes}** unique values.
            
            * Classification is for predicting a few categories (e.g., 'True'/'False').
            * Your data looks like a **Regression** problem (predicting a number).
            
            **Please select a different target column or change the "Problem Type" to "Regression".**
            """)
            return
        if unique_classes < 2:
            st.error(f"""
            **Configuration Error!**
            
            Your target column ('{target_column}') only has **{unique_classes}** unique value.
            The model needs at least two different classes to learn from (e.g., 'True' and 'False').
            """)
            return

    # --- 3. Automatic Feature Detection ---
    st.write("### 2. Detecting Feature Types")
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    st.info(f"**Found {len(numeric_features)} numerical features:**\n{numeric_features}")
    st.info(f"**Found {len(categorical_features)} categorical features:**\n{categorical_features}")

    # --- 4. NEW FIX: Force Data Types ---
    # Force all categorical features to be strings
    for col in categorical_features:
        X[col] = X[col].astype(str)

    # Force target 'y' to be encoded integers (0, 1, 2...) for classification
    if problem_type == "Classification":
        y = y.astype('category').cat.codes
    # --- END FIX ---

    # --- 5. Create Preprocessing Pipelines ---
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # SimpleImputer will now use 'most_frequent' (a string) on all-string columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # --- 6. Combine transformers ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough')

    # --- 7. Select Model Based on Problem Type ---
    if problem_type == "Classification":
        model = RandomForestClassifier(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)

    # --- 8. Create the full ML Pipeline ---
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # --- 9. Split data and train model ---
    st.write("### 3. Training Model")
    with st.spinner("Splitting data and training model... This may take a moment."):
        
        try:
            stratify_option = (y if problem_type == 'Classification' else None)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_option
            )
        except ValueError as e:
            st.warning(f"""
            **Stratification Warning:** Could not stratify data (Error: {e}). 
            This usually happens if one class has very few samples.
            Proceeding without stratification.
            """)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)
        
    st.success("Model training complete! ✅")

    # --- 10. Evaluate Model and Show Results ---
    st.write("### 4. Model Evaluation")
    if problem_type == "Classification":
        accuracy = accuracy_score(y_test, y_pred)
        st.metric(label="**Accuracy**", value=f"{accuracy * 100:.2f}%")
        st.text("Classification Report:")
        try:
            # We must map the integer labels (0, 1) back to their names for the report
            target_names = y.astype('category').cat.categories.astype(str)
            st.code(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        except Exception as e:
            st.warning(f"Could not generate classification report. (Error: {e})")
            # Fallback report without names
            st.code(classification_report(y_test, y_pred, zero_division=0))
            
    else: # Regression
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        st.metric(label="**R-squared (R2)**", value=f"{r2:.4f}")
        st.metric(label="**Root Mean Squared Error (RMSE)**", value=f"{rmse:.4f}")

    st.write("### 5. Sample Predictions")
    
    # Map predictions back to original labels for display
    if problem_type == "Classification":
        original_labels = y.astype('category').cat.categories
        y_test_labels = y_test.map(lambda x: original_labels[x])
        y_pred_labels = [original_labels[p] for p in y_pred]
        results_df = pd.DataFrame({'Actual': y_test_labels, 'Predicted': y_pred_labels})
    else:
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        
    st.dataframe(results_df.head(10))
# -------------------------------------------------------------------
# Main App Interface
# -------------------------------------------------------------------

# --- 1. File Uploader ---
st.header("1. Upload Your Data")
uploaded_file = st.file_uploader(
    "Upload your data (CSV, TXT, Excel)", 
    type=["csv", "xlsx", "xls", "txt"]
)

df = None  # Initialize df as None

if uploaded_file is not None:
    
    # --- 2. Load Options (NEW) ---
    st.subheader("Load Options")
    
    # Check if it's an Excel file
    is_excel = uploaded_file.name.endswith(('.xls', '.xlsx'))
    
    if is_excel:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading Excel file: {e}")
    else:
        # It's a text file (CSV, TXT, etc.)
        st.info("Your file is a text file. Please specify the load options:")
        col1, col2 = st.columns(2)
        with col1:
            separator = st.text_input(
                "Delimiter (Separator)", 
                value=",", 
                help="What character separates your columns? (e.g., `,` for comma, `\\t` for tab, `;` for semicolon)"
            )
        with col2:
            header_row = st.number_input(
                "Header Row", 
                value=0, 
                min_value=0, 
                help="Which row number contains the column names? (0 is the first row)"
            )
        
        if separator == "\\t": # Handle tab character
            separator = "\t"

        try:
            df = pd.read_csv(uploaded_file, sep=separator, header=header_row)
        except Exception as e:
            st.error(f"Error loading text file: {e}. Check your delimiter and header row.")

    # --- 3. Continue if 'df' is loaded ---
    if df is not None:
        st.success(f"File '{uploaded_file.name}' loaded successfully. Found {df.shape[0]} rows and {df.shape[1]} columns.")
        st.dataframe(df.head())
        
        # --- 4. User Configuration ---
        st.header("2. Configure Your Model")
        
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
            help="**Classification:** Predicts a category (e.q., 'Churn'/'Stay', 'Red'/'Green').\n\n**Regression:** Predicts a number (e.g., price, temperature)."
        )

        ignore_cols = st.multiselect(
            "Select any columns to ignore (e.g., ID, Name, Phone Number)",
            options=[col for col in all_columns if col != target_column],
            help="Select columns that are not useful for prediction (like 'phnum' in your churn data)."
        )

        # --- 5. Run Button ---
        st.header("3. Build Your Model")
        if st.button("**Build Model**", type="primary"):
            if target_column is None:
                st.error("Please select a target variable to predict.")
            else:
                build_model(df, target_column, problem_type, ignore_cols)



