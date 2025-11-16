#pickling the model 

import joblib 

#define filename 
model_filename = 'six_factor_model.joblib'

# Save (pickle) the fitted model using joblib
try:
    joblib.dump(model, model_filename)
    print(f"\nModel saved to {model_filename}")
except Exception as e:
    print(f"\nError saving model: {e}")
    