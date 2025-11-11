# Traditional ML Pipeline Results

## Project Information
- **Dataset**: GoEmotions
- **Task**: Multi-label Emotion Classification (28 emotions)
- **Completion Time**: 2025-08-31 22:02:04

## Dataset Statistics
- **Total Samples**: 203,147
- **Training Samples**: 142,202
- **Validation Samples**: 30,472
- **Test Samples**: 30,473
- **Number of Features**: 17,056

## Best Model Performance
- **Model**: Multinomial NB
- **Test F1 Macro**: 0.3081
- **Test F1 Micro**: 0.3589
- **Test Hamming Loss**: 0.0660

## Model Comparison

| Model | F1 Macro | F1 Micro | Hamming Loss | Training Time (s) |
|-------|----------|----------|--------------|-------------------|
| Complement NB | 0.2561 | 0.3118 | 0.1107 | 0.7 |
| Linear SVM | 0.2707 | 0.3067 | 0.1266 | 83.7 |
| Logistic Regression | 0.2729 | 0.3092 | 0.1248 | 39.9 |
| Multinomial NB | 0.3044 | 0.3533 | 0.0667 | 0.7 |
| Random Forest | 0.2174 | 0.2198 | 0.0507 | 365.9 |
| SGD Classifier | 0.2574 | 0.2916 | 0.1387 | 14.1 |


## Key Findings
1. The Multinomial NB model achieved the best performance with F1 Macro score of 0.3081
2. Most challenging emotions: desire, disgust, caring, disapproval, annoyance, confusion, excitement, curiosity, approval, disappointment, embarrassment, realization, nervousness, relief, grief, pride
3. Feature extraction included TF-IDF, character n-grams, and linguistic features for a total of 17,056 features

## File Locations
- Models saved in: `C:\Users\isaac\OneDrive\Desktop\GoEmotions_NLP_Project\models\traditional_ml`
- Results saved in: `C:\Users\isaac\OneDrive\Desktop\GoEmotions_NLP_Project\results\traditional_ml_results`
- Visualizations: model_comparison.png, emotion_performance.png, error_analysis.png
