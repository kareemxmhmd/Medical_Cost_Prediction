import joblib
import pandas as pd

model_pipeline = joblib.load("model.pkl")

def predict_medical_cost(age, sex, bmi, children, smoker, region):
    
    data = {
        'age': [age],
        'sex': [sex.lower()],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker.lower()],
        'region': [region.lower()]
    }

    input_df = pd.DataFrame(data)
    
    prediction = model_pipeline.predict(input_df)
    return prediction[0]


print("Medical Cost Prediction")

cost = predict_medical_cost(
    int(input("Enter age (e.g., 19): ")),
    input("Enter sex (male/female): ").strip(),
    float(input("Enter BMI (e.g., 27.9): ")),
    int(input("Enter number of children: ")),
    input("Smoker? (yes/no): ").strip(),
    input("Region (southwest/southeast/northwest/northeast): ").strip(),
)

print(f"Estimated Medical Cost: ${cost:,.2f}")