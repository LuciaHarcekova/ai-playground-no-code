# AI & Machine Learning Cheatsheet Summary

## What is Machine Learning?

Machine Learning (ML) is a subset of Artificial Intelligence (AI) that enables systems to learn from data and improve performance over time without being explicitly programmed. ML models identify patterns in data to make predictions or decisions.

## AI vs Machine Learning

* **Artificial Intelligence (AI):** A broad field aiming to create machines capable of mimicking human intelligence, including reasoning, learning, and problem-solving.
* **Machine Learning (ML):** A subset of AI focused on developing algorithms that allow systems to learn patterns from data.

## Machine Learning Workflow

**Problem → Data → Preprocessing → Split → Train → Evaluate → Tune → Deploy → Monitor**

* **Problem:** Define the objective and success criteria.
* **Data:** Collect relevant datasets.
* **Preprocessing:** Clean, encode, and scale features.
* **Split:** Divide data into training and testing sets.
* **Train:** Fit models on training data.
* **Evaluate:** Measure performance using metrics.
* **Tune:** Optimize hyperparameters and model selection.
* **Deploy:** Integrate the model into production.
* **Monitor:** Track performance and update as needed.

---

## Types of Machine Learning

| Type                   | What it Does                                                           | Data Requirement                                    | Example Use Cases                                          | Benefits                                                              | Limitations                                  |
| ---------------------- | ---------------------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------- |
| Supervised Learning    | Learns a mapping from input to output using labeled data               | Requires labeled data (features + target)           | Predicting house prices, spam detection, medical diagnosis | High accuracy with enough labeled data; easy to evaluate with metrics | Requires large labeled datasets; may overfit |
| Unsupervised Learning  | Finds hidden patterns or groupings without labeled outcomes            | Only input features, no target labels               | Customer segmentation, anomaly detection, topic modeling   | Useful for exploring unknown data; can reveal hidden structures       | Hard to interpret; no guaranteed accuracy    |
| Reinforcement Learning | Learns by interacting with an environment, receiving rewards/penalties | Requires an environment with feedback/reward system | Self-driving cars, game-playing AI, robotics control       | Can learn complex sequential tasks; adapts through experience         | Computationally expensive; slow convergence  |

---

## Supervised Learning

Supervised learning trains models on labeled data to learn the mapping from input (features) to output (labels/values).

### Classification Models (Categorical target)

| Model               | Explanation                                   | Example Usage                    |
| ------------------- | --------------------------------------------- | -------------------------------- |
| Logistic Regression | Models probability of binary outcomes         | Email spam detection             |
| Decision Trees      | Classifies using feature-based rules          | Loan approval                    |
| Random Forests      | Ensemble of decision trees                    | Diagnosing diseases              |
| SVM                 | Finds best separating hyperplane              | Handwritten digit classification |
| k-NN                | Assigns label based on nearest neighbors      | Product recommendation           |
| Naïve Bayes         | Probabilistic classifier using Bayes’ theorem | Sentiment analysis               |
| Gradient Boosting   | Sequential boosted trees for high accuracy    | Fraud detection                  |
| Neural Networks     | Layers of neurons for complex patterns        | Face recognition                 |

**Evaluation Metrics**

* **Accuracy:** Correct predictions / Total → Fraction of correct predictions; misleading for imbalanced data.
* **Precision:** TP / (TP+FP) → Of predicted positives, how many are truly positive.
* **Recall (Sensitivity):** TP / (TP+FN) → Of actual positives, how many were identified.
* **F1-Score:** Harmonic mean of precision & recall → Balances precision & recall.
* **ROC-AUC:** Tradeoff between TPR and FPR → AUC closer to 1 = better discrimination.
* **Confusion Matrix:** Table of TP, TN, FP, FN → Visual breakdown of results.

### Regression Models (Continuous target)

| Model                           | Explanation                                      | Example Usage                  |
| ------------------------------- | ------------------------------------------------ | ------------------------------ |
| Linear Regression               | Fits a straight line between features and output | House price prediction         |
| Polynomial Regression           | Captures curves with polynomial features         | Population growth modeling     |
| Ridge & Lasso Regression        | Regularized linear models to reduce overfitting  | Stock price prediction         |
| Decision Tree Regression        | Splits data into branches based on conditions    | Car value prediction           |
| Random Forest Regression        | Ensemble of trees for robust predictions         | Sales revenue forecasting      |
| Support Vector Regression (SVR) | Finds best-fit boundary with error tolerance     | Temperature trends prediction  |
| Gradient Boosting               | Sequential models correcting previous errors     | Electricity demand forecasting |

**Regression Metrics**

* **MAE:** Average absolute differences → How far predictions are from actual values.
* **MSE:** Average squared differences → Penalizes large errors; sensitive to outliers.
* **RMSE:** Square root of MSE → Same unit as target; interpretable.
* **R² Score:** Proportion of variance explained → 0 = no fit, 1 = perfect fit.
* **Adjusted R²:** R² adjusted for number of predictors → Useful for multiple regression.

---

## Regression vs Classification Summary

| Aspect             | Regression                                                                             | Classification                                                                    |
| ------------------ | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Target Variable    | Continuous                                                                             | Categorical                                                                       |
| Goal               | Predict exact numeric values                                                           | Assign input to predefined categories                                             |
| Example Problems   | House price, stock price, temperature                                                  | Spam detection, disease diagnosis, image recognition                              |
| Algorithms         | Linear, Polynomial, Ridge/Lasso, Decision Trees, Random Forest, SVR, Gradient Boosting | Logistic, Decision Trees, Random Forests, SVM, k-NN, Naïve Bayes, Neural Networks |
| Evaluation Metrics | MAE, MSE, RMSE, R²                                                                     | Accuracy, Precision, Recall, F1 Score, ROC-AUC                                    |
| Output             | Real number                                                                            | Class label or probability                                                        |
| Use Cases          | Finance, real estate, forecasting                                                      | Healthcare, fraud detection, image/speech recognition                             |
| Challenges         | Sensitive to outliers; assumes numeric relationships                                   | Class imbalance, overfitting, interpretability                                    |

---

## Unsupervised Learning

### Clustering

* **k-Means:** Groups data into k clusters → Customer segmentation.
* **Hierarchical Clustering:** Nested clusters → Document grouping.
* **DBSCAN:** Density-based clustering → Anomaly detection.
* **GMM:** Probabilistic assignment → Image segmentation.
* **Mean Shift:** Auto-discovers cluster count → Geographical hotspot detection.

### Dimensionality Reduction

* **PCA:** Reduce dimensions retaining variance → Image compression.
* **SVD:** Factorizes matrices → Recommendation systems.
* **t-SNE:** Visualizes high-dimensional data → Word embeddings visualization.
* **Autoencoders:** Compress & reconstruct data → Noise reduction in images.

---

## Reinforcement Learning Methods

* **Q-Learning:** Optimal actions using Q-table → Robot maze navigation.
* **Deep Q-Networks:** Uses deep learning → Playing Atari games.
* **SARSA:** Learns based on actual action → Safe driving policy.
* **Policy Gradient Methods:** Optimize policies directly → Robotic arm control.
* **Actor-Critic:** Combines value & policy-based learning → Continuous control tasks.
* **PPO & TRPO:** Stable training → Humanoid robot control.
* **Monte Carlo:** Learn from averaging episodes → Game simulations.

---

## Data Preprocessing

* Handle missing values: remove, fill, interpolate.
* Encode categorical data: LabelEncoder, OneHotEncoder, OrdinalEncoder.
* Feature scaling: StandardScaler, MinMaxScaler, RobustScaler.

## Model Training Essentials

* Train-test split prevents overfitting.
* Hyperparameter tuning: GridSearchCV, Bayesian Optimization, AutoML.
* Feature engineering & selection: PCA, feature importance, correlation matrix.

---

## Deep Learning Fundamentals

* **ANN:** General-purpose → Housing prices prediction.
* **CNN:** Convolution layers → Object detection in photos.
* **RNN:** Recurrent layers → Next word prediction.
* **Transformers (BERT, GPT):** Attention-based → Machine translation, chatbots.

---

## Bias-Variance Trade-off

* **Bias:** Error from overly simple models (underfitting).
* **Variance:** Error from overly complex models (overfitting).
* **Goal:** Balance bias and variance for optimal generalization.

## Common Pitfalls

* **Data Leakage:** Test data influences training.
* **Imbalanced Data:** Unequal class distribution.
* **Overfitting / Underfitting:** Model too complex/simple for data.
