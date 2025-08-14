# %%
reqUrl = "https://572ceb441bd4.ngrok-free.app/predict"

# %%
import os
import json
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize


# Parameters
IMAGE_SIZE = (256, 256)  # You can change this
class_labels = ['Fake', 'Real']

import joblib
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications import efficientnet_v2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# ...existing code...
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
from sklearn.linear_model import LogisticRegression
import json # Import json for VLM output

# Define constants
DL_IMAGE_SIZE = [256, 256]
class_labels = ['Fake', 'Real']


# %%
import base64

def file_to_base64(filepath):
    """
    Converts a file to a Base64-encoded string.

    Args:
        filepath (str): The path to the file.

    Returns:
        str: The Base64-encoded string, or None if an error occurs.
    """
    try:
        with open(filepath, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode('utf-8')
            return encoded_string
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# # Example usage:
# # Assuming you have a file named 'example.jpg' in the same directory.
# # Replace 'path/to/your/file.png' with the actual path to your file.
# file_path = "e:/sem4/proj/FusionModel/Eval/SlovakID-Real-1.png"
# base64_string = file_to_base64(file_path)

from fastapi import UploadFile
from fastapi.responses import JSONResponse
import os
import shutil
import traceback
from io import BytesIO
from PIL import Image
import requests
import json



headersList = {
"Accept": "*/*",
"User-Agent": "Thunder Client (https://www.thunderclient.com)",
"ngrok-skip-browser-warning": "true",
"Content-Type": "application/json" 
}

def save_bytes_to_temp_image(image_bytes, suffix=".jpg"):
    img = Image.open(BytesIO(image_bytes))
    temp_path = f"temp_upload{suffix}"
    img.save(temp_path)
    return temp_path

class IDCard:
    scaler = None
    best_model = None
    label_encoder = None
    eff_datagen = None
    eff_model = None
    yolo_model = None
    yolo_class_names = None
    ID_model = None
    ID_processor = None
    if os.path.exists(r'E:\sem4\proj\FusionModel\IDModel\IDCard_meta_learner.joblib'):
        meta_learner = joblib.load(r'E:\sem4\proj\FusionModel\IDModel\IDCard_meta_learner.joblib')
    else:
        meta_learner = None

    def __init__(self):
        # NOTE: For this example, we will not load the actual models as they
        # are not available. We will simulate them instead.
        print("Initializing IDCard model components...")
        self.scaler = joblib.load(r'E:\sem4\proj\FusionModel\IDModel\IDCard_scaler.joblib')
        self.best_model = joblib.load(r'E:\sem4\proj\FusionModel\IDModel\IDCard_svm_real_fake_model.joblib')
        self.label_encoder = joblib.load(r'E:\sem4\proj\FusionModel\IDModel\IDCard_label_encoder.joblib')
        self.eff_datagen = ImageDataGenerator(
            preprocessing_function=efficientnet_v2.preprocess_input,
            shear_range=0.2,
            zoom_range=0.2,
            featurewise_center=True
        )
        self.eff_model = tf.keras.models.load_model(r'E:\sem4\proj\FusionModel\IDModel\IDCard_efficientnetv2_model.keras', custom_objects={'EfficientNetV2B0': EfficientNetV2B0})
        self.yolo_model = YOLO(r"E:\sem4\proj\FusionModel\IDModel\IDCard-yolo-best.pt")
        self.yolo_class_names = self.yolo_model.names

        # # Initialize VLM components
        # self.ID_model = AutoModelForImageTextToText.from_pretrained("AliceRolan/gemma-idcard-FT",
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="eager",
        #     quantization_config=bnb_config,
        # )
        # self.ID_processor = AutoProcessor.from_pretrained("AliceRolan/gemma-idcard-FT")

        # Initialize meta-learner (logistic regression is a good starting point)
        # if self.meta_learner is None:
        #     self.meta_learner = LogisticRegression(solver='liblinear')
        print("IDCard model initialization complete.")

    def train_meta_learner(self, X_train_predictions, y_train_labels):
        """
        Trains the meta-learner on the predictions of the base models.

        Args:
            X_train_predictions (np.array): A numpy array of shape (n_samples, 3)
                                            containing the predictions from the
                                            three base models (SVM, EfficientNet, YOLO).
            y_train_labels (np.array): A numpy array of shape (n_samples,)
                                       containing the true labels for the training data.
        """
        print("Training meta-learner...")
        self.meta_learner.fit(X_train_predictions, y_train_labels)
        print("Meta-learner training complete.")

    def predict(self, image_path,meta_learn=False):
        """
        Makes a prediction using the ensemble model.

        Args:
            X_eval (np.array): Features for the SVM model.
            image_path (str): Path to the image file for DL and VLM models.

        Returns:
            dict: A dictionary containing all model predictions and the final fused prediction.
        """
        print(f"\nMaking prediction for image: {image_path}")

        # 1. Get predictions from base models

        # a. SVM prediction
        # img_path = os.path.join(label_path, img_file)
        # time_taken_svm = time.time() - start_time
        start_time = time.time()
        
        img = imread(image_path, as_gray=True)  # Convert to grayscale
        img_resized = resize(img, IMAGE_SIZE, anti_aliasing=True)
        X_eval = np.array([img_resized.flatten()])  # Wrap in list to make 2D array
        X_eval_scaled = self.scaler.transform(X_eval)
        svm_prediction = self.best_model.predict(X_eval_scaled)[0]
        time_taken_svm = time.time() - start_time
        start_time = time.time()

        # b. EfficientNetV2 prediction
        # Temporarily skip EfficientNetV2 prediction for meta-learner training
        # if not meta_learn:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=DL_IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array_processed = self.eff_datagen.standardize(img_array)
        eff_predictions = self.eff_model.predict(img_array_processed)
        eff_prediction = np.argmax(eff_predictions[0])
        eff_confidence = eff_predictions[0][eff_prediction]
        time_taken_eff = time.time() - start_time
        start_time = time.time()

        # c. YOLO prediction
        yolo_results = self.yolo_model(image_path)
        yolo_results = yolo_results[0]
        yolo_top1_class = yolo_results.probs.top1
        yolo_confidence = yolo_results.probs.data[yolo_top1_class].item()
        time_taken_yolo = time.time() - start_time

        if meta_learn:
            # We'll use the numerical predictions/class indices as features
            # Include EfficientNetV2 prediction if not skipped
            return np.array([[svm_prediction, eff_prediction, yolo_top1_class]])

        # # d. VLM prediction
        # vlm_raw_output = execute_prompt(self.ID_model, self.ID_processor, Image.open(image_path), top_p=0.9, temperature=0.5)
        # vlm_parsed_output = json.loads(vlm_raw_output)
        # vlm_fraud_score = vlm_parsed_output.get("fraud_score", 0)
        # vlm_explanation = vlm_parsed_output.get("explanation", "No explanation provided.")

        # # 2. VLM Integration and Thresholding
        # vlm_is_reliable = False
        # if vlm_fraud_score >= 0.75: # Define a threshold for VLM reliability
        #     vlm_is_reliable = True

        # 3. Prepare data for the meta-learner
        # We'll use the numerical predictions/class indices as features
        meta_features = np.array([[svm_prediction, eff_prediction, yolo_top1_class]])
        base64_string = file_to_base64(image_path)
        payload = json.dumps({
            "category": "Real" if "real" in os.path.basename(image_path).lower() else "Fake" ,  # Extract category from the path
            "base64_image": base64_string,
            "docType": "currency" if  "currency" in image_path.lower() else "idcard"
        })

        ext_response = requests.request("POST", reqUrl, data=payload, headers=headersList)
        ext_response_json = ext_response.json() if ext_response.content else {}

        if ext_response_json.get("finetuned_vlm_prediction", "Unknown").get("Authenticity", "Unknown") == 'Fake':
            vlm_prediction = 0
        else:
            vlm_prediction = 1
        # 4. Fused prediction using the meta-learner
        if self.meta_learner:
            if hasattr(self.meta_learner, "predict_proba"):
                proba = self.meta_learner.predict_proba(meta_features)[0]
                max_proba_idx = np.argmax(proba)
                fused_label = self.label_encoder.inverse_transform([max_proba_idx])[0]
                fused_label_proba = proba[max_proba_idx]

            else:
                fused_prediction_encoded = self.meta_learner.predict(meta_features)[0]
                fused_label = self.label_encoder.inverse_transform([fused_prediction_encoded])[0]
                fused_label_proba = "N/A"  # Meta-learner does not support probability prediction
        else:
            # Fallback to majority voting if meta-learner isn't trained
            predictions = [svm_prediction, eff_prediction, yolo_top1_class]
            mode = max(set(predictions), key=predictions.count)
            fused_label = self.label_encoder.inverse_transform([mode])[0]
            fused_label_proba = "N/A"  # Meta-learner does not support probability prediction
        # Print meta-learner score (probability for the predicted class)
        
        # 5. Return all results, including the fused prediction and VLM output
        return {
            "fused_final_prediction": fused_label,
            "fused_final_prediction_proba": f"{fused_label_proba:.4f}" if isinstance(fused_label_proba, float) else fused_label_proba,
            "svm_prediction(baseline)": self.label_encoder.inverse_transform([svm_prediction])[0],
            "time_taken_svm": f"{time_taken_svm:.4f}",
            "efficientnet_prediction": class_labels[eff_prediction],
            "efficientnet_confidence": f"{eff_confidence:.4f}",
            "time_taken_efficientnet": f"{time_taken_eff:.4f}",
            "yolo_prediction": self.yolo_class_names[yolo_top1_class],
            "yolo_confidence": f"{yolo_confidence:.4f}",
            "time_taken_yolo": f"{time_taken_yolo:.4f}",
            "vlm_response": ext_response_json  # Include VLM response
            # "vlm_fraud_score": vlm_fraud_score,
            # "vlm_explanation": vlm_explanation,
            # "vlm_is_reliable": vlm_is_reliable,
            
        }

# %%

class Currency:
    scaler = None
    best_model = None
    label_encoder = None
    eff_datagen = None
    eff_model = None
    yolo_model = None
    yolo_class_names = None
    ID_model = None
    ID_processor = None
    if os.path.exists(r'E:\sem4\proj\FusionModel\CurrencyModel\Currency_meta_learner.joblib'):
        meta_learner = joblib.load(r'E:\sem4\proj\FusionModel\CurrencyModel\Currency_meta_learner.joblib')
    else:
        meta_learner = None

    def __init__(self):
        # NOTE: For this example, we will not load the actual models as they
        # are not available. We will simulate them instead.
        print("Initializing Currency model components...")
        self.scaler = joblib.load(r'E:\sem4\proj\FusionModel\CurrencyModel\currency_scaler.joblib')
        self.best_model = joblib.load(r'E:\sem4\proj\FusionModel\CurrencyModel\currency_svm_real_fake_model.joblib')
        self.label_encoder = joblib.load(r'E:\sem4\proj\FusionModel\CurrencyModel\currency_label_encoder.joblib')
        self.eff_datagen = ImageDataGenerator(
            preprocessing_function=efficientnet_v2.preprocess_input,
            shear_range=0.2,
            zoom_range=0.2,
            featurewise_center=True
        )
        self.eff_model = tf.keras.models.load_model(r'E:\sem4\proj\FusionModel\CurrencyModel\currency_efficientnetv2_model.keras', custom_objects={'EfficientNetV2B0': EfficientNetV2B0})
        self.yolo_model = YOLO(r"E:\sem4\proj\FusionModel\CurrencyModel\currency-yolo-best.pt")
        self.yolo_class_names = self.yolo_model.names

        # # Initialize VLM components
        # self.ID_model = AutoModelForImageTextToText.from_pretrained("AliceRolan/gemma-idcard-FT",
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="eager",
        #     quantization_config=bnb_config,
        # )
        # self.ID_processor = AutoProcessor.from_pretrained("AliceRolan/gemma-idcard-FT")

        # Initialize meta-learner (logistic regression is a good starting point)
        if self.meta_learner is None:
            self.meta_learner = LogisticRegression(solver='liblinear')
        print("Currency model initialization complete.")

    def train_meta_learner(self, X_train_predictions, y_train_labels):
        """
        Trains the meta-learner on the predictions of the base models.

        Args:
            X_train_predictions (np.array): A numpy array of shape (n_samples, 3)
                                            containing the predictions from the
                                            three base models (SVM, EfficientNet, YOLO).
            y_train_labels (np.array): A numpy array of shape (n_samples,)
                                       containing the true labels for the training data.
        """
        print("Training meta-learner...")
        self.meta_learner.fit(X_train_predictions, y_train_labels)
        print("Meta-learner training complete.")

    def predict(self, image_path,meta_learn=False):
        """
        Makes a prediction using the ensemble model.

        Args:
            X_eval (np.array): Features for the SVM model.
            image_path (str): Path to the image file for DL and VLM models.

        Returns:
            dict: A dictionary containing all model predictions and the final fused prediction.
        """
        print(f"\nMaking prediction for image: {image_path}")

        # 1. Get predictions from base models

        # a. SVM prediction
        # img_path = os.path.join(label_path, img_file)
        
        start_time = time.time()
        if not meta_learn:
            
            img = imread(image_path, as_gray=True)  # Convert to grayscale
            img_resized = resize(img, IMAGE_SIZE, anti_aliasing=True)
            X_eval = np.array([img_resized.flatten()])  # Wrap in list to make 2D array
            X_eval_scaled = self.scaler.transform(X_eval)
            svm_prediction = self.best_model.predict(X_eval_scaled)[0]
        time_taken_svm = time.time() - start_time
        start_time = time.time()
        # b. EfficientNetV2 prediction
        # Temporarily skip EfficientNetV2 prediction for meta-learner training
        # if not meta_learn:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=DL_IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array_processed = self.eff_datagen.standardize(img_array)
        eff_predictions = self.eff_model.predict(img_array_processed)
        eff_prediction = np.argmax(eff_predictions[0])
        eff_confidence = eff_predictions[0][eff_prediction]
        time_taken_eff = time.time() - start_time
        start_time = time.time()

        # c. YOLO prediction
        yolo_results = self.yolo_model(image_path)
        yolo_results = yolo_results[0]
        yolo_top1_class = yolo_results.probs.top1
        yolo_confidence = yolo_results.probs.data[yolo_top1_class].item()

        time_taken_yolo = time.time() - start_time

        base64_string = file_to_base64(image_path)
        # payload = json.dumps({
        # "category":image_path.split(os.sep)[-2],  # Extract category from the path
        # "base64_image": base64_string,
        # "docType": image_path.split(os.sep)[-1].split('_')[1]  # Add docType to payload
        # })
        payload = json.dumps({
            "category": "Real" if "real" in os.path.basename(image_path).lower() else "Fake" ,  # Extract category from the path
            "base64_image": base64_string,
            "docType": "currency" if  "currency" in image_path.lower() else "idcard"
        })

        ext_response = requests.request("POST", reqUrl, data=payload, headers=headersList)
        ext_response_json = ext_response.json() if ext_response.content else {}

        if ext_response_json.get("finetuned_vlm_prediction", "Unknown").get("Authenticity", "Unknown") == 'Fake':
            vlm_prediction = 0
        else:
            vlm_prediction = 1
        

        # Append external response to json_response
        # json_response["vlm_response"] = ext_response_json

        if meta_learn:
            # We'll use the numerical predictions/class indices as features
            # Include EfficientNetV2 prediction if not skipped
            return np.array([[eff_prediction, yolo_top1_class, vlm_prediction]])

        # # d. VLM prediction
        # vlm_raw_output = execute_prompt(self.ID_model, self.ID_processor, Image.open(image_path), top_p=0.9, temperature=0.5)
        # vlm_parsed_output = json.loads(vlm_raw_output)
        # vlm_fraud_score = vlm_parsed_output.get("fraud_score", 0)
        # vlm_explanation = vlm_parsed_output.get("explanation", "No explanation provided.")

        # # 2. VLM Integration and Thresholding
        # vlm_is_reliable = False
        # if vlm_fraud_score >= 0.75: # Define a threshold for VLM reliability
        #     vlm_is_reliable = True

        # 3. Prepare data for the meta-learner
        # We'll use the numerical predictions/class indices as features
        meta_features = np.array([[eff_prediction, yolo_top1_class, vlm_prediction]])

        # 4. Fused prediction using the meta-learner
        if self.meta_learner:
            if hasattr(self.meta_learner, "predict_proba"):
                proba = self.meta_learner.predict_proba(meta_features)[0]
                max_proba_idx = np.argmax(proba)
                fused_label = self.label_encoder.inverse_transform([max_proba_idx])[0]
                fused_label_proba = proba[max_proba_idx]

            else:
                fused_prediction_encoded = self.meta_learner.predict(meta_features)[0]
                fused_label = self.label_encoder.inverse_transform([fused_prediction_encoded])[0]
                fused_label_proba = "N/A"  # Meta-learner does not support probability prediction
        else:
            # Fallback to majority voting if meta-learner isn't trained
            predictions = [svm_prediction, eff_prediction, yolo_top1_class]
            mode = max(set(predictions), key=predictions.count)
            fused_label = self.label_encoder.inverse_transform([mode])[0]
            fused_label_proba = "N/A"  # Meta-learner does not support probability prediction


        # 5. Return all results, including the fused prediction and VLM output
        return {
            "fused_final_prediction": fused_label,
            "fused_final_prediction_proba": f"{fused_label_proba:.4f}" if isinstance(fused_label_proba, float) else fused_label_proba,
            "svm_prediction(baseline)": self.label_encoder.inverse_transform([svm_prediction])[0],
            "time_taken_svm": f"{time_taken_svm:.4f}",
            "efficientnet_prediction": class_labels[eff_prediction],
            "efficientnet_confidence": f"{eff_confidence:.4f}",
            "time_taken_efficientnet": f"{time_taken_eff:.4f}",
            "yolo_prediction": self.yolo_class_names[yolo_top1_class],
            "yolo_confidence": f"{yolo_confidence:.4f}",
            "time_taken_yolo": f"{time_taken_yolo:.4f}",
            "vlm_response": ext_response_json  # Include VLM response
            # "vlm_fraud_score": vlm_fraud_score,
            # "vlm_explanation": vlm_explanation,
            # "vlm_is_reliable": vlm_is_reliable,
            
        }


# %%
# %%
# del id_card_model
import traceback

# --- Step 2: Initialize the IDCard class (or mock version) ---
id_card_model = IDCard()
currency_model = Currency()


# %%



from fastapi import FastAPI, UploadFile
import nest_asyncio

# Apply nest_asyncio to allow nested loops in a notebook environment
nest_asyncio.apply()

# Initialize the FastAPI app
app = FastAPI(title="Fraud Detection API",
              description="API for detecting fraud using a fusion model.",
              version="1.0.0",redoc_url="/api/docs")

@app.get("/")
async def root():
    return {"message": "Hello from your FastAPI app!"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

from fastapi.responses import JSONResponse

def convert_numpy_types(obj):
    """
    Recursively convert numpy types in a dict to native Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj



@app.post("/predict/")
async def predict(file: UploadFile):
    try:
        if not file:
            return JSONResponse(content={"error": "No file provided."})
        content = await file.read()
        # Save bytes to temp image file for YOLO and other models
        image_path = save_bytes_to_temp_image(content)

        # Check if the image file exists
        if not os.path.exists(image_path):
            return JSONResponse(content={"error": "Image file not found."})
        # Choose model based on filename
        if "currency" in file.filename.lower():
            test_model = currency_model.predict(image_path, False)
        else:
            test_model = id_card_model.predict(image_path, False)

        test_model = convert_numpy_types(test_model)
        json_response = test_model
        # base64_string = file_to_base64(image_path)


        # payload = json.dumps({
        # "category":json_response.get("fused_final_prediction", "Unknown"),
        # "base64_image": base64_string,
        # "docType": file.filename.split('.')[-1]  # Add docType to payload
        # })

        # ext_response = requests.request("POST", reqUrl, data=payload, headers=headersList)
        # ext_response_json = ext_response.json() if ext_response.content else {}

        # # Append external response to json_response
        # json_response["vlm_response"] = ext_response_json

        # Cle
        
        # Clean up temp file
        os.remove(image_path)
        return JSONResponse(content=json_response)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)})
    
def predict_with_cotent(content,filename):
    try:
        image_path = save_bytes_to_temp_image(content,filename)
        print(f"Image saved to {image_path}")
        # Check if the image file exists
        if not os.path.exists(image_path):
            return JSONResponse(content={"error": "Image file not found."})
        # Choose model based on filename
        if "currency" in filename.lower():
            test_model = currency_model.predict(image_path, False)
        else:
            test_model = id_card_model.predict(image_path, False)

        test_model = convert_numpy_types(test_model)
        json_response = test_model
        # base64_string = file_to_base64(image_path)


        # payload = json.dumps({
        # "category":json_response.get("fused_final_prediction", "Unknown"),
        # "base64_image": base64_string,
        # "docType": "currency" if  "currency" in filename.lower() else "idcard"
        # })

        # ext_response = requests.request("POST", reqUrl, data=payload, headers=headersList)
        # ext_response_json = ext_response.json() if ext_response.content else {}

        # # Append external response to json_response
        # json_response["vlm_response"] = ext_response_json
        # Clean up temp file
        os.remove(image_path)
        return json_response
    except Exception as e:
        traceback.print_exc()
        return ({"error": str(e)})


# ...existing code...


# Run the Uvicorn server on port 8000
# uvicorn.run(app, host="localhost", port=8000)





import streamlit as st
import streamlit as st
import requests
import time
from PIL import Image
import json
import asyncio

# Streamlit UI
# Set Streamlit page config for wide layout
st.set_page_config(layout="wide")
# Inject CSS to make Streamlit app use full height
st.markdown("""
    <style>
        .main {
            min-height: 100vh;
            height: 100vh;
        }
        [data-testid="stAppViewContainer"] {
            min-height: 100vh;
            height: 100vh;
        }
        [data-testid="stVerticalBlock"] {
            min-height: 100vh;
            height: 100vh;
        }
        .block-container {
            min-height: 100vh;
            height: 100vh;
        }
    </style>
""", unsafe_allow_html=True)
st.title("Fraud Detection App")
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()  # Read bytes ONCE
    col1, col2 = st.columns(2)
    with col1:
        with col1:
            image = Image.open(BytesIO(file_bytes))
            st.image(image, caption=uploaded_file.name)
    with col2:
        if st.button("Predict"):
            start_time = time.time()
            # file_bytes = uploaded_file.read() 
            result = predict_with_cotent(file_bytes, uploaded_file.name)
            time_taken = time.time() - start_time

            if "error" in result:
                st.error("Error in prediction: " + result["error"])
            else:
                st.json(result)
            st.write(f"Time taken for processing: {time_taken:.2f} seconds")
