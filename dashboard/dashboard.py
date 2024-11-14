import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from mealpy import BinaryVar, GA, Problem
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time


class FeatureSelectionProblem(Problem):
    def __init__(self, bounds=None, minmax="max", X_train=None, y_train=None, classifier=None,
                 classifier_name=None, progress_placeholder=None, chart_placeholder=None, **kwargs):
        self.X_train = X_train
        self.y_train = y_train
        self.classifier = classifier
        self.classifier_name = classifier_name
        self.fitness_history = []
        self.progress_placeholder = progress_placeholder
        self.chart_placeholder = chart_placeholder
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        selected_features = x.astype(bool)

        if not selected_features.any():
            return 0.0

        X_selected = self.X_train[:, selected_features]

        # if callable(self.classifier):
        #     model = self.classifier()
        # else:
        #     model = self.classifier.__class__(**self.classifier.get_params())

        # model.fit(X_selected, self.y_train)
        # y_val_pred = model.predict(X_selected)
        # fitness_score = accuracy_score(self.y_train, y_val_pred)

        # Cross-val:
        scores = cross_val_score(
            self.classifier(), X_selected, self.y_train, cv=5)
        fitness_score = np.mean(scores)

        self.fitness_history.append(fitness_score)

        self.update_progress()

        return fitness_score

    def update_progress(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=self.fitness_history,
            mode='lines+markers',
            name=self.classifier_name
        ))
        fig.update_layout(
            title=f'{self.classifier_name} Feature Selection Progress',
            xaxis_title='Evaluations',
            yaxis_title='Fitness Score',
            yaxis_range=[0, 1]
        )
        self.chart_placeholder.plotly_chart(fig, use_container_width=True)

        current_best = max(self.fitness_history)
        self.progress_placeholder.text(
            f"Current best fitness for {self.classifier_name}: {current_best:.4f}"
        )


def load_data():
    df = pd.read_csv('data/alzheimers_disease_data.csv')
    return df


def run_feature_selection(clf_name, clf, X_train_std, y_train, bounds, progress_placeholder, chart_placeholder, feature_names, X_test_std, y_test):
    problem = FeatureSelectionProblem(
        bounds=bounds,
        minmax="max",
        X_train=X_train_std,
        y_train=y_train,
        classifier=clf,
        classifier_name=clf_name,
        progress_placeholder=progress_placeholder,
        chart_placeholder=chart_placeholder
    )

    term_dict = {
        "max_epoch": 20,
        "max_fe": 1000,
        "max_time": 30,
        "max_early_stop": 5,
    }

    model = GA.BaseGA(epoch=20, pop_size=50, pc=0.9, pm=0.05)
    best_solution = model.solve(problem, termination=term_dict)

    selected_features = best_solution.solution.astype(bool)
    selected_feature_names = np.array(feature_names)[selected_features]
    selected_features_df = pd.DataFrame(
        selected_feature_names, columns=["Selected Features"])

    X_selected_train = X_train_std[:, selected_features]
    X_selected_test = X_test_std[:, selected_features]

    if callable(clf):
        model = clf()
    else:
        model = clf.__class__(**clf.get_params())

    model.fit(X_selected_train, y_train)
    y_pred = model.predict(X_selected_test)

    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    return {
        "Best feature selection": best_solution.solution,
        "Best fitness score": best_solution.target.fitness,
        "Fitness history": problem.fitness_history,
        "Selected Features DataFrame": selected_features_df,
        "Confusion Matrix": conf_matrix,
        "Classification Report": report_df,
        "y_pred": y_pred,
    }


def main():
    st.title("Machine Learning Feature Selection with Genetic Algorithm")

    st.header("Data Exploration")

    df = load_data()
    df = df.drop(['PatientID', 'DoctorInCharge'], axis=1)
    feature_names = df.drop(columns='Diagnosis').columns.tolist()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Statistics")
    st.dataframe(df.describe())

    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    st.header("Feature Selection")

    classifiers = {
        "Artificial Neural Network": lambda: MLPClassifier(max_iter=300, learning_rate_init=0.001, solver='adam'),
        "Support Vector Machine": SVC,
    }

    bounds = BinaryVar(n_vars=X.shape[1], name="feature_selection")

    if 'results' not in st.session_state:
        st.session_state.results = {}

    if st.button("Start Feature Selection"):
        classifier_columns = st.columns((3, 1))

        for name, clf in classifiers.items():
            st.subheader(name)
            progress_placeholder = st.empty()
            chart_placeholder = st.empty()

            result = run_feature_selection(
                name, clf, X_train_std, y_train, bounds,
                progress_placeholder, chart_placeholder, feature_names, X_test_std, y_test
            )

            st.session_state.results[name] = result

            st.subheader(f"Confusion Matrix for {name}")
            disp = ConfusionMatrixDisplay(
                confusion_matrix=result['Confusion Matrix'], display_labels=np.unique(y))
            disp.plot(cmap=plt.cm.Blues)
            st.pyplot(plt)

            st.subheader(f"Classification Report for {name}")
            st.dataframe(result['Classification Report'])

            st.subheader(f"Selected Features for {name}")
            st.dataframe(result["Selected Features DataFrame"])
        st.header("Final Comparison")
        fig = go.Figure()

        for name, result in st.session_state.results.items():
            fig.add_trace(go.Scatter(
                y=result['Fitness history'],
                mode='lines+markers',
                name=name
            ))

        fig.update_layout(
            title='Algorithm Comparison',
            xaxis_title='Evaluations',
            yaxis_title='Fitness Score',
            yaxis_range=[0, 1]
        )

        st.plotly_chart(fig, use_container_width=True)

        st.header("Final Results")
        results_df = pd.DataFrame({
            name: {
                'Best Fitness Score': result['Best fitness score'],
                'Number of Selected Features': sum(result['Best feature selection'])
            }
            for name, result in st.session_state.results.items()
        }).T

        st.dataframe(results_df)


if __name__ == "__main__":
    main()
