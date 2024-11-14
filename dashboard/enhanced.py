import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import mutual_info_classif
from mealpy import BinaryVar, GA, Problem
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class EnhancedFeatureSelectionProblem(Problem):
    def __init__(self, bounds=None, minmax="max", X_train=None, y_train=None, classifier=None,
                 classifier_name=None, progress_placeholder=None, chart_placeholder=None, **kwargs):
        self.X_train = X_train
        self.y_train = y_train
        self.classifier = classifier
        self.classifier_name = classifier_name
        self.fitness_history = []
        self.progress_placeholder = progress_placeholder
        self.chart_placeholder = chart_placeholder
        self.best_cv_score = 0
        self.n_features_history = []
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        selected_features = x.astype(bool)

        if not selected_features.any():
            return 0.0

        X_selected = self.X_train[:, selected_features]
        n_features = np.sum(selected_features)

        feature_penalty = 0.001 * n_features

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf = self.classifier()
        scores = cross_val_score(
            clf, X_selected, self.y_train, cv=cv, scoring='accuracy')

        fitness_score = np.mean(scores) - feature_penalty

        self.fitness_history.append(fitness_score)
        self.n_features_history.append(n_features)

        if fitness_score > self.best_cv_score:
            self.best_cv_score = fitness_score

        self.update_progress()

        return fitness_score

    def update_progress(self):
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=('Fitness Score Progress', 'Number of Selected Features'))

        fig.add_trace(
            go.Scatter(y=self.fitness_history, mode='lines+markers',
                       name='Fitness Score'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(y=self.n_features_history, mode='lines+markers',
                       name='Number of Features'),
            row=2, col=1
        )

        fig.update_layout(height=600, showlegend=True)
        fig.update_yaxes(title_text="Fitness Score", row=1, col=1)
        fig.update_yaxes(title_text="Feature Count", row=2, col=1)

        self.chart_placeholder.plotly_chart(fig, use_container_width=True)

        self.progress_placeholder.text(
            f"{self.classifier_name} Progress:\n"
            f"Current best CV score: {self.best_cv_score:.4f}\n"
            f"Current feature count: {self.n_features_history[-1]}"
        )


def run_feature_selection(clf_name, clf, X_train_std, y_train, bounds,
                          progress_placeholder, chart_placeholder,
                          feature_names, X_test_std, y_test):
    problem = EnhancedFeatureSelectionProblem(
        bounds=bounds,
        minmax="max",
        X_train=X_train_std,
        y_train=y_train,
        classifier=clf,
        classifier_name=clf_name,
        progress_placeholder=progress_placeholder,
        chart_placeholder=chart_placeholder
    )

    ga_params = {
        "epoch": 50,
        "pop_size": 100,
        "pc": 0.9,
        "pm": 0.1
    }

    term_dict = {
        "max_epoch": 50,
        "max_fe": 5000,
        "max_early_stop": 10
    }

    model = GA.BaseGA(**ga_params)
    best_solution = model.solve(problem, termination=term_dict)

    selected_features = best_solution.solution.astype(bool)
    selected_feature_names = np.array(feature_names)[selected_features]

    selected_features_df = pd.DataFrame({
        "Selected Features": selected_feature_names
    })

    X_selected_train = X_train_std[:, selected_features]
    X_selected_test = X_test_std[:, selected_features]

    if callable(clf):
        model = clf()
    else:
        model = clf.__class__(**clf.get_params())

    cv_scores = cross_val_score(model, X_selected_train, y_train, cv=5)

    model.fit(X_selected_train, y_train)
    y_pred = model.predict(X_selected_test)

    return {
        "Best feature selection": best_solution.solution,
        "Best fitness score": best_solution.target.fitness,
        "Fitness history": problem.fitness_history,
        "Feature count history": problem.n_features_history,
        "Selected Features DataFrame": selected_features_df,
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose(),
        "CV Scores": cv_scores,
        "y_pred": y_pred,
    }


def main():
    st.title("Enhanced Machine Learning Feature Selection with Genetic Algorithm")

    classifiers = {
        "Artificial Neural Network": lambda: MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            learning_rate_init=0.001,
            solver='adam',
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10,
            random_state=42
        ),
        "Support Vector Machine": lambda: SVC(
            kernel='rbf',
            C=1.0,
            probability=True,
            random_state=42
        )
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
