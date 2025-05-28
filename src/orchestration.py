import os

# Run data preprocessing service
os.system("python3 src/data_prep_service.py")

# Run model training service
os.system("python3 src/train_service.py")

# Run prediction service
os.system("python3 src/predict_service.py")