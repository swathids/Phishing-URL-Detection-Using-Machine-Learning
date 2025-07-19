# Phishing-URL-Detection-Using-Machine-Learning
Phishing URL Detection Using Machine Learning" is a classification project using URL structure and HTML content-based features to detect phishing websites.



This project uses the PhiUSIIL Phishing URL Dataset, publicly available on Mendeley Data. The dataset contains over 10,000 labeled URL samples, each classified as either phishing or legitimate. It provides 54 features that describe various aspects of the URLs, including both structural properties (e.g., URLLength, TLD, ContainsIPAddress) and content-based characteristics extracted from the HTML source code (e.g., HasTitle, HasSubmitButton, LineOfCode). These features offer a comprehensive view of both the superficial and underlying traits of a webpage, making the dataset ideal for training phishing detection models.

The project is executed in two parts: initially with an intensive preporocessing and cleaning of the dataset proceeded by ML classification using various models and then comparing their performances.

## Requirements for the Project
The phishing URL detection project involves several stages—data preprocessing, feature engineering, model training, evaluation, and visualization—necessitating a robust Python-based data science environment. The following are the essential requirements:
 
Python Version: 3.8 or later

Core Libraries:

NumPy and Pandas – for data manipulation and numerical operations

Matplotlib and Seaborn – for visualizing distributions, boxplots, and KDE plots

Scikit-learn – for preprocessing (scaling, train-test splitting), model building (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN), and evaluation metrics

XGBoost – for implementing the eXtreme Gradient Boosting algorithm

Other Libraries:

scipy.stats – for statistical checks (e.g., skewness, Z-score)

warnings – to suppress unwanted warnings during data exploration

Environment Requirements
Jupyter Notebook or Google Colab (recommended for its GPU/TPU support and zero-install setup)

Minimum 4 GB RAM (8 GB or more recommended)

A stable internet connection if using Colab or downloading the dataset

Dataset
PhiUSIIL Phishing URL Dataset, containing 54 features per entry (URL-based and source-code-based), with over 10,000 entries, and a binary label indicating whether the URL is phishing or legitimate.



## PREPROCESSING

Preprocessing Steps for Phishing URL Detection Dataset
The phishing URL detection project involved meticulous preprocessing of the raw dataset to enhance data quality and improve model performance. The dataset, obtained from the PhiUSIIL repository on Kaggle, contained over 10,000 records with 54 features encompassing a wide range of indicators. These features were derived from URL-based heuristics (e.g., URLLength, HasIPAddress, SuspiciousTLD), HTML and JavaScript content attributes (e.g., LineOfCode, HasTitle, HasSubmitButton), and DNS/network-level properties. The main objective of preprocessing was to prepare the data in a form suitable for machine learning algorithms by addressing missing values, outliers, skewness, and feature redundancy.

The first stage of preprocessing involved checking for null values and confirming data consistency—no missing or NaN values were observed. Next, duplicate columns were scanned using correlation and identical value checks; none were identified. A key focus was then placed on distribution analysis, especially on skewed features. Using skew() from the pandas library, the top 10 most skewed features were identified to be heavily right-skewed, which could potentially bias the model. To manage this, IQR-based clipping was performed to reduce the impact of extreme outliers without removing the data points, preserving data balance. This was followed by log1p transformation on the most skewed features to bring their distributions closer to normal, stabilizing variance. Furthermore, Z-score normalization was applied to standardize features, making them comparable in scale for algorithms like Logistic Regression and SVM. All transformations were performed only after train-test splitting to avoid data leakage. The result of these steps was a clean, normalized, and balanced dataset where feature ranges and distributions were more consistent, aiding better model generalization and convergence.

 1) Dataset Cleaning & Exploration Steps
Dataset imported and explored (shape, structure, and feature types)

Verified label balance (phishing vs legitimate)

Checked for null/missing values – none found

Checked for duplicate rows – removed if any were present

Verified uniqueness of records

Explored class distribution and imbalance (target feature)

Performed basic EDA using describe(), info(), correlation matrix

Checked for constant and quasi-constant features

Checked for duplicated columns or features with high correlation (none dropped)

Visualized label distribution using countplots and pie charts

2) Preprocessing & Transformation Steps
Computed skewness of all numeric features using df.skew()

Identified top 10 most skewed features

Plotted boxplots to observe extreme outliers

Performed IQR-based clipping on skewed features to mitigate outliers

Applied np.log1p() transformation on skewed features to reduce asymmetry

Verified changes in distribution using kernel density plots (KDE)

Applied Z-score normalization on all numerical features to standardize scale

Visualized before and after normalization using KDE plots and boxplots

Final preprocessed dataset verified to be outlier-handled, scaled, and model-ready

After completing the full preprocessing pipeline, the dataset was significantly refined and optimized for model training. From the original 54 features, a total of 23 carefully selected features were retained for machine learning. These features were chosen based on their relevance, reduced skewness, and strong correlation with the target label, while also ensuring multicollinearity and redundancy were minimized. The features span across multiple categories, including URL structure, special character patterns, HTML tag presence, and behavioral indicators — all contributing meaningful signals for phishing detection.



# ML ALGORITHMS 

Machine Learning Algorithms Used
The project explored multiple supervised machine learning algorithms to classify URLs as phishing or legitimate. Each model was selected for its theoretical strengths and empirical suitability to structured tabular data:

Logistic Regression
A linear classification model that estimates the probability of a URL being phishing using a logistic (sigmoid) function. Serves as a simple yet effective baseline.

Support Vector Machine (SVM)
Constructs an optimal hyperplane in a high-dimensional feature space to separate classes. Particularly effective after normalization, especially in handling non-linearly separable data.

Decision Tree Classifier
A rule-based model that splits the data based on the most informative features. While fast and interpretable, it tends to overfit on complex data without regularization.

Random Forest Classifier
An ensemble of decision trees trained on random feature subsets and bootstrapped samples. Provides robust performance by reducing variance and handling noisy features well.

Gradient Boosting Classifier
Builds sequential decision trees where each new tree focuses on correcting the errors of the previous ones. Capable of capturing complex patterns with fine-grained control over learning.

XGBoost (Extreme Gradient Boosting)
An advanced and optimized implementation of gradient boosting that includes regularization, parallel processing, and tree pruning. XGBoost often delivers state-of-the-art results on structured data and significantly improved both accuracy and training efficiency in this project.

K-Nearest Neighbors (KNN)
An instance-based algorithm that classifies a data point based on the majority vote of its K nearest neighbors. It performs adequately with proper feature scaling but is less effective for high-dimensional datasets.

## Results Obtained

Among the various models evaluated for phishing URL detection, XGBoost achieved the highest performance with an accuracy of 99.93%, F1-score of 0.9994, and a near-perfect ROC-AUC of 0.99999, making it the most reliable model overall. Gradient Boosting also performed exceptionally well with an accuracy of 99.82%, precision and recall of 0.9984, F1-score of 0.9984, and an outstanding ROC-AUC of 0.99998. Similarly, the Random Forest classifier showed robust results with 99.89% accuracy, 0.9991 F1-score, and 0.99994 AUC. SVM with RBF kernel followed closely, delivering 99.82% accuracy, 0.9984 F1-score, and a ROC-AUC of 0.99996.

While slightly behind the top performers, K-Nearest Neighbors (KNN) still achieved a strong 99.81% accuracy and 0.9983 F1-score, with a ROC-AUC of 0.99943. Decision Tree classifier obtained 99.75% accuracy, 0.9978 F1-score, and an AUC of 0.99748. Lastly, Logistic Regression, though a simpler model, performed remarkably well with an accuracy of 99.46%, F1-score of 0.9953, and ROC-AUC of 0.99979. All models had high mean cross-validation AUC scores, with Gradient Boosting and Random Forest showing the lowest standard deviation, suggesting consistent and reliable performance across folds.


# Significance of the  Project
Phishing attacks remain one of the most prevalent and damaging forms of cybercrime, exploiting human vulnerabilities to steal sensitive data such as login credentials, credit card numbers, and personal information. With the rapid growth of the internet and online services, attackers are continually crafting more sophisticated phishing websites that closely mimic legitimate ones, making traditional rule-based detection methods increasingly ineffective.This project addresses the urgent need for intelligent, automated phishing detection systems using machine learning. By leveraging a diverse set of 54 URL and HTML-based features, the project builds predictive models that can generalize across a wide range of phishing strategies. The use of preprocessing techniques, feature selection, and multiple ML algorithms ensures that the final models are not only accurate but also robust and generalizable.

From a broader perspective, the project holds significant real-world relevance in the field of cybersecurity. It demonstrates how data-driven approaches can proactively detect threats before user interaction, thereby preventing potential harm. In cybersecurity contexts, such systems play a critical role in strengthening the first line of defense against phishing campaigns. The insights and methodology developed through this work can be integrated into browser security extensions, email filters, or organizational firewalls, making the internet safer for individuals, enterprises, and institutions alike.

# Future Enhancements
In the future, this phishing URL detection project can be significantly enhanced by incorporating advanced deep learning models such as LSTMs, CNNs, or Transformer-based architectures, which are capable of capturing sequential and contextual patterns in URLs more effectively than traditional machine learning models. Real-time phishing detection systems, like browser plug-ins or cloud-based APIs, can also be developed for practical deployment. Additionally, the inclusion of more sophisticated features such as WHOIS registration data, DNS records, SSL certificate details, and visual webpage cues (via screenshots or HTML rendering) can enrich the feature space and improve detection performance. Leveraging continual learning or periodic model retraining strategies will also be crucial in adapting to the dynamic nature of phishing tactics, making the solution robust and scalable in real-world cybersecurity application.

# Conclusion

The project presents a comprehensive machine learning-based approach to detect phishing URLs using a curated set of informative features derived from URL structure, HTML content, and domain characteristics. A deep understanding of data preprocessing played a pivotal role, as steps like handling duplicates, managing skewness, and applying normalization significantly improved the quality and learnability of the dataset. Careful feature selection and transformation ensured the models were trained on the most relevant and unbiased inputs. The project further highlights the strength of advanced machine learning algorithms—particularly XGBoost and Random Forest—which demonstrated high accuracy, precision, and robustness in detecting malicious URLs. This work not only emphasizes the critical role of preprocessing in the ML pipeline but also showcases the practical value of automated phishing detection systems as scalable tools in modern cybersecurity defense strategies. Special thanks to Ms. Krishna Priya for her valuable mentoring support and guidance throughout the course of this project.








