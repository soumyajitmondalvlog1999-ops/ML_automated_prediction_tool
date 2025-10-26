# 🤖 Automated ML Model Builder 🧠

This Streamlit web app lets you build and evaluate machine learning models on the fly. **No code required!**

Upload any clean, structured dataset (CSV, TXT, or Excel), configure your options, and let the app automatically train a model and show you the results.

[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B.svg?style=for-the-badge&logo=Streamlit)](https://streamlit.io)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E.svg?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)

---

## 📽️ Live Demo

**(Recommended: Record a short GIF of your app in action and replace the image below!)**

![App Demo](httpss://user-images.githubusercontent.com/ (link_to_your_demo.gif) )

*To create a GIF, use a free tool like [GIPHY Capture](https://giphy.com/apps/giphycapture) or [LICEcap](https://www.cockos.com/licecap/), record your screen, and drag the GIF file into this README editor.*

---

## ✨ Key Features

* **📤 Upload Any Data:** Supports `.csv`, `.txt`, and `.xlsx` (Excel) files.
* **⚙️ Flexible Load Options:**
    * Specify the **delimiter** (e.g., `,`, `\t`, `;`).
    * Set the **header row number** (e.g., `0` for the first row, `2` for the third).
    * Supports files with **no header** at all.
* **🔬 Smart Configuration:**
    * Visually select your **target variable** (what you want to predict).
    * Choose your **problem type** (Classification or Regression).
    * Select any columns to **ignore** (like IDs or names).
* **🤖 Automated ML:**
    * Automatically detects **numeric** and **categorical** features.
    * Applies data cleaning (imputation) and preprocessing (scaling, encoding).
    * Trains a `RandomForest` model to get a powerful baseline.
* **📈 Instant Results:**
    * **Classification:** Get an **Accuracy** score and a full **Classification Report**.
    * **Regression:** Get **R-squared (R2)** and **Root Mean Squared Error (RMSE)**.
    * See a table of **Actual vs. Predicted** values from the test set.

---

## 🚀 How to Use

1.  **Upload Your Data:** Click "Browse files" and select your `.csv`, `.txt`, or `.xlsx` file.
2.  **Set Load Options:**
    * **No Header?** Check the "My file does NOT have a header row" box.
    * **Header on a different row?** Enter the row number (e.g., `2` for the 3rd row).
    * **Text File?** Specify the delimiter (like `,` or `\t`).
3.  **Configure Your Model:**
    * Select the **column you want to predict** from the dropdown.
    * Tell the app if it's a **Classification** (predicting a category) or **Regression** (predicting a number) problem.
    * Select any columns you want the model to **ignore**.
4.  **Build Your Model:** Click the "Build Your Model" button and see the results!

---

## 💻 Tech Stack

* **Streamlit** (Web App Framework)
* **Pandas** (Data Manipulation)
* **Scikit-learn** (Machine Learning Pipeline)
* **Numpy** (Numerical Operations)
* **Openpyxl** (For reading Excel files)

---

## ✍️ Author

**Created by SOUMYAJIT MONDAL** 👋
