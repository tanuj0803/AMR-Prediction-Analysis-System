import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AMR Prediction System",
    page_icon="🧬",
    layout="wide"
)


# --------------------------------------------------
# Load trained model and preprocessing pipeline
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load("rf_final.pkl")
    preprocessor = joblib.load("preprocessor_final.pkl")
    threshold = joblib.load("rf_threshold.pkl")

    return model, preprocessor, threshold


rf_model, preprocessor, threshold = load_model()


# --------------------------------------------------
# Load saved analysis results
# --------------------------------------------------

@st.cache_data
def load_analysis_data():

    model_results = pd.read_csv(
        "model_comparison.csv"
    )

    species_results = pd.read_csv(
        "species_model_results.csv"
    )

    species_resistance = pd.read_csv(
        "species_resistance_rates.csv"
    )

    eskape_results = pd.read_csv(
        "eskape_comparison.csv"
    )

    return (
        model_results,
        species_results,
        species_resistance,
        eskape_results
    )


(
    model_results,
    species_results,
    species_resistance,
    eskape_results
) = load_analysis_data()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🧬 AMR Prediction & Analysis System")

st.caption(
    "Machine Learning Prototype for Antimicrobial Resistance Analysis"
)

st.warning(
    "Academic decision-support prototype only. "
    "This system is not a clinical diagnostic tool."
)


# --------------------------------------------------
# Navigation
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Amikacin Prediction",
        "Model Comparison",
        "Species Analysis",
        "ESKAPE Analysis",
        "Explainability"
    ]
)


# ==================================================
# OVERVIEW
# ==================================================

if page == "Overview":

    st.header("Project Overview")

    st.write(
        """
        This project uses machine learning to analyze
        antimicrobial resistance patterns across bacterial
        species.

        The current prediction task focuses on predicting
        Amikacin resistance using bacterial information,
        sample information, and antimicrobial susceptibility
        results.
        """
    )

    st.subheader("Project Workflow")

    st.markdown(
        """
        **AMR Dataset**
        ↓
        
        **Data Preprocessing**
        ↓
        
        **Feature Engineering**
        ↓
        
        **Machine Learning Models**
        ↓
        
        **Model Evaluation**
        ↓
        
        **Species & ESKAPE Analysis**
        ↓
        
        **Explainability**
        ↓
        
        **Streamlit Decision-Support Prototype**
        """
    )

    st.subheader("Prediction Target")

    st.info(
        "Amikacin Resistance / Susceptibility"
    )

    st.subheader("Models Evaluated")

    st.write(
        "Random Forest, XGBoost, SVM and LightGBM"
    )


# ==================================================
# AMIKACIN PREDICTION
# ==================================================

elif page == "Amikacin Prediction":

    st.header("Amikacin Resistance Prediction")

    st.write(
        "Enter the available sample and antimicrobial "
        "susceptibility information."
    )

    col1, col2 = st.columns(2)

    with col1:

        species = st.selectbox(
            "Species",
            [
                "Acinetobacter baumannii",
                "Klebsiella pneumoniae",
                "Proteus mirabilis",
                "Pseudomonas aeruginosa",
                "Serratia marcescens",
                "Escherichia coli",
                "Enterobacter hormaechi",
                "Enterobacter cloacae",
                "Citrobacter freundii",
                "Morganella morganii",
                "Klebsiella aerogenes",
                "Acinetobacter pitii",
                "Proteus vulgaris",
                "Klebsiella oxytoca",
                "Citrobacter koseri"
            ]
        )

        country = st.text_input(
            "Country",
            value="India"
        )

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Unknown"
            ]
        )

        age_group = st.text_input(
            "Age Group",
            value="Adult"
        )

    with col2:

        speciality = st.text_input(
            "Speciality",
            value="General"
        )

        source = st.text_input(
            "Source",
            value="Unknown"
        )

        year = st.number_input(
            "Year",
            min_value=2000,
            max_value=2030,
            value=2025,
            step=1
        )


    st.subheader("Antibiotic Susceptibility")

    st.write(
        "Enter the observed result for each antibiotic."
    )

    antibiotics = [
        "Amoxycillin clavulanate_I",
        "Ampicillin_I",
        "Aztreonam_I",
        "Cefepime_I",
        "Ceftazidime avibactam_I",
        "Ceftazidime_I",
        "Colistin_I",
        "Imipenem_I",
        "Levofloxacin_I",
        "Meropenem_I",
        "Piperacillin tazobactam_I",
        "Tigecycline_I"
    ]

    antibiotic_values = {}

    columns = st.columns(3)

    for index, antibiotic in enumerate(antibiotics):

        with columns[index % 3]:

            antibiotic_values[antibiotic] = st.selectbox(
                antibiotic.replace("_I", ""),
                ["S", "R"],
                key=antibiotic
            )


    st.divider()

    if st.button(
        "Predict Amikacin Resistance",
        type="primary"
    ):

        input_data = {
            "Species": species,
            "Country": country,
            "Gender": gender,
            "Age Group": age_group,
            "Speciality": speciality,
            "Source": source,
            "Year": year
        }

        input_data.update(
            antibiotic_values
        )

        input_df = pd.DataFrame(
            [input_data]
        )

        try:

            encoded_input = preprocessor.transform(
                input_df
            )

            probability = rf_model.predict_proba(
                encoded_input
            )[0, 1]

            prediction = int(
                probability >= threshold
            )

            st.subheader("Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                if prediction == 1:

                    st.error(
                        "Predicted Result: Resistant"
                    )

                else:

                    st.success(
                        "Predicted Result: Susceptible"
                    )

            with result_col2:

                st.metric(
                    "Resistance Probability",
                    f"{probability * 100:.2f}%"
                )

            st.caption(
                f"Random Forest decision threshold: "
                f"{threshold:.2f}"
            )

        except Exception as error:

            st.error(
                f"Prediction error: {error}"
            )


# ==================================================
# MODEL COMPARISON
# ==================================================

elif page == "Model Comparison":

    st.header("Model Performance Comparison")

    st.dataframe(
        model_results.round(4),
        use_container_width=True
    )

    metric = st.selectbox(
        "Select Metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        model_results["Model"],
        model_results[metric]
    )

    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Model Comparison - {metric}"
    )

    plt.xticks(rotation=20)
    plt.tight_layout()

    st.pyplot(fig)


# ==================================================
# SPECIES ANALYSIS
# ==================================================

elif page == "Species Analysis":

    st.header(
        "Species-wise Amikacin Resistance Analysis"
    )

    st.dataframe(
        species_resistance.round(2),
        use_container_width=True
    )

    plot_data = species_resistance.head(15)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        plot_data["Species"],
        plot_data["Resistance_Rate"]
    )

    ax.set_xlabel(
        "Amikacin Resistance Rate (%)"
    )

    ax.set_ylabel(
        "Bacterial Species"
    )

    ax.set_title(
        "Amikacin Resistance Across Bacterial Species"
    )

    ax.invert_yaxis()

    plt.tight_layout()

    st.pyplot(fig)

    st.subheader("Species-wise Model Performance")

    st.dataframe(
        species_results.round(4),
        use_container_width=True
    )


# ==================================================
# ESKAPE ANALYSIS
# ==================================================

elif page == "ESKAPE Analysis":

    st.header(
        "ESKAPE-related Species Comparison"
    )

    st.dataframe(
        eskape_results.round(2),
        use_container_width=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        eskape_results["Species"],
        eskape_results["Resistance Rate (%)"]
    )

    ax.set_ylabel(
        "Amikacin Resistance Rate (%)"
    )

    ax.set_xlabel(
        "Species"
    )

    ax.set_title(
        "Amikacin Resistance Across ESKAPE-related Species"
    )

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig)


# ==================================================
# EXPLAINABILITY
# ==================================================

# ==================================================
# EXPLAINABILITY
# ==================================================

elif page == "Explainability":

    st.header("Model Explainability")

    st.write(
        """
        Explainability helps us understand which input features
        influence the Random Forest model's predictions.

        SHAP (SHapley Additive exPlanations) was used during
        the project analysis to study the contribution of
        different features.
        """
    )

    st.subheader("What influences the prediction?")

    st.info(
        """
        The Random Forest model uses bacterial species,
        sample information, year, and susceptibility results
        for other antibiotics to predict Amikacin resistance.
        """
    )

    st.subheader("Important Feature Groups")

    feature_data = pd.DataFrame({
        "Feature Group": [
            "Bacterial Species",
            "Antibiotic Susceptibility Results",
            "Patient / Sample Information",
            "Year"
        ],
        "Role in Prediction": [
            "Captures differences in resistance patterns between bacterial species.",
            "Provides susceptibility patterns for other antibiotics.",
            "Provides demographic and sample-related information.",
            "Captures changes in resistance patterns across time."
        ]
    })

    st.dataframe(
        feature_data,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("How to Interpret the Prediction")

    st.markdown(
        """
        - **Higher resistance probability** indicates that the model
          estimates a greater likelihood of Amikacin resistance.

        - **Lower resistance probability** indicates that the model
          estimates a greater likelihood of Amikacin susceptibility.

        - Changes in the S/R values of other antibiotics can change
          the predicted probability because these values are used
          as model input features.

        - SHAP analysis can be used to identify which individual
          features contribute more strongly to a particular prediction.
        """
    )

    st.subheader("Project Limitation")

    st.warning(
        """
        This explainability module is intended for academic analysis.
        The model should not be used as a standalone clinical decision
        system or as a replacement for laboratory antimicrobial
        susceptibility testing.
        """
    )
