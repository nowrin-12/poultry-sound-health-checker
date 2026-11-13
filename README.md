Poultry Sound Health Checker 🐔

A machine learning demo that classifies short poultry audio clips as Healthy, Unhealthy, or Noise.

Live demo: https://poultry-sound-health-checker.streamlit.app

How it works
Audio is resampled to 16 kHz and cut into 1-second clips.
For each clip, 40 MFCCs are computed and summarized (mean + standard deviation = 80 features).
An RBF SVM predicts the class. For longer recordings, predictions from overlapping clips are averaged.
Results

Evaluated with 5-fold GroupKFold, grouped by recording file so clips from the same file never appear in both train and test.

**Model	Accuracy** 

MFCC + SVM (used in the app)	~94%  

MFCC + Random Forest	~94%  

Small CNN on mel spectrograms	~88%  

YAMNet embeddings + Logistic Regression	~77%  


A shuffled-label sanity check dropped accuracy to ~33%, as expected for 3 classes.

Dataset Source : https://data.mendeley.com/

Dataset Name        : [Poultry Vocalization Signal Dataset for Early Disease Detection Mendeley.]

346 recordings:  
139 Healthy, 
121 Unhealthy,   
86 Noise.  

**Files**  

chicken.ipynb - data processing, model comparison and training  

streamlit_app.py - the web app  

final_svm.joblib - trained model  

requirements.txt - dependencies  

**Limitations**  

Small, single-source dataset. Not tested on other farms, breeds, or microphones.  

Results may not generalize to real-world conditions.  

This is a learning/research demo, not a veterinary or diagnostic tool.
