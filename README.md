# Emotion Detection Web App   
**Author:** Jessica Ogbonna

This Flask-based web app detects human emotions from uploaded or webcam-captured images using a pretrained **DeepFace** model. It identifies emotions such as happiness, sadness, anger, surprise, and more.  

The app saves analyzed results, including the dominant emotion and probabilities, into a local **SQLite** database and displays annotated images with emotion labels.  

###  Tech Stack
- Flask (Backend Web Framework)  
- DeepFace (Pretrained Deep Learning Model)  
- TensorFlow + tf-keras (Model Backend)  
- OpenCV & Pillow (Image Handling)  
- SQLite (Local Storage)

###  Run Locally
```bash
pip install -r requirements.txt
python app.py