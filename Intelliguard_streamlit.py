import streamlit as st
import sys
import pathlib
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import face_recognition
import pandas as pd
import uuid
from dotenv import load_dotenv
import os
import boto3
import smtplib
from email.mime.text import MIMEText
from sqlalchemy import create_engine, text
from datetime import datetime
from langchain_experimental.sql import SQLDatabaseChain
from langchain.sql_database import SQLDatabase 
from langchain.llms import OpenAI
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import make_msgid
from email.mime.image import MIMEImage

# -------------------- Configuration --------------------
# Load environment variables from .env file
load_dotenv(dotenv_path="Password.env")

# Environment variables
DB_URI = os.getenv("DB_URI")
ALERT_EMAIL = os.getenv("ALERT_EMAIL")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = os.getenv("SMTP_PORT")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
AWS_ACCESS = os.getenv("AWS_ACCESS")
AWS_SECRET = os.getenv("AWS_SECRET")
AWS_REGION = os.getenv("AWS_REGION")
s3_bucket = os.getenv("s3_bucket")
openai_api_key = os.getenv("OPENAI_API_KEY")

# -------------------- Load Models --------------------

# Add YOLOv5 to the system path
YOLOV5_PATH = os.path.join(os.path.dirname(__file__), 'yolov5')
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

# Now import YOLOv5 modules
from yolov5.models.common import DetectMultiBackend

if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath

@st.cache_resource
def load_yolo_model():
    weights_path = 'C:\\Users\\AK\\Downloads\\colab_train2\\runs\\train\\colab_train2\\weights\\best.pt'
    model = DetectMultiBackend(weights_path, device='cpu') 
    return model

def load_known_faces():
    return {"admin": face_recognition.face_encodings(face_recognition.load_image_file("admin.jpg"))[0]}

model = load_yolo_model()
known_faces = load_known_faces()

# -------------------- Database Connection --------------------
print("DB_URI:", DB_URI)
engine = create_engine(DB_URI)

# -------------------- AWS Bucket Connection --------------------
s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS, aws_secret_access_key=AWS_SECRET, region_name=AWS_REGION)

# -------------------- Email Alert --------------------

def send_email_alert(to_email,labels_str,detected_img_cv):
     # Convert OpenCV image to bytes
    _, buffer = cv2.imencode('.png', detected_img_cv)
    img_bytes = buffer.tobytes()

    # Create a unique content ID for embedding
    image_cid = make_msgid(domain='xyz.com')  # You can use your domain here

    # Build HTML content
    html_message = f"""
    <html>
        <body>
            <h2 style="color: red;">🚨 PPE Violations Detected</h2>
            <p><strong>The following violations were detected:</strong></p>
            <p><strong>Violations:</strong> {labels_str}</p>
            <p><img src="cid:{image_cid}" alt="Detection Image" width="600"/></p>
            <p style="font-size: small; color: gray;">This is an automated alert from the PPE Monitoring System.</p>
        </body>
    </html>
    """
    # Compose email with HTML content
    msg = MIMEMultipart("alternative")
    msg['Subject'] = "🚨 PPE Violation Alert"
    msg['From'] = EMAIL_USER
    msg['To'] = to_email
    
    # Create a multipart/related container for HTML + image
    msg_related = MIMEMultipart('related')

    # Attach the HTML content
    msg_html = MIMEText(html_message, "html")
    msg.attach(msg_html)
    msg_related.attach(msg_html)

    # Attach the image to the related container
    img = MIMEImage(img_bytes, 'png')
    img.add_header('Content-ID', image_cid)
    img.add_header('Content-Disposition', 'inline', filename='detected.png')
    msg_related.attach(img)

    # Attach the related container to the root message
    msg.attach(msg_related)

    # Send the email
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)

# -------------------- Upload image to S3-bucket --------------------
def upload_to_s3(image_bytes, prefix="ppe_violation"):
    key = f"{prefix}/{uuid.uuid4()}.jpg"
    s3.put_object(Bucket=s3_bucket, Key=key, Body=image_bytes, ContentType='image/jpeg')
    return f"https://{s3_bucket}.s3.amazonaws.com/{key}"    

# -------------------- Streamlit UI --------------------
st.title("IntelliGuard PPE Detection System")

with st.expander("🔒 Admin Login"):
    pin = st.text_input("Enter PIN", type="password")
    face_image = st.file_uploader("Upload Face Image", type=["jpg", "png"], key="face")

if st.button("Login"):
    if pin == "1234" and face_image is not None:
        img = face_recognition.load_image_file(face_image)
        encodings = face_recognition.face_encodings(img)
        if encodings and np.linalg.norm(encodings[0] - known_faces["admin"]) < 0.5:
            st.session_state["authenticated"] = True
            st.success("Access granted")
        else:
            st.error("Face not recognized")
    else:
        st.error("Invalid login")

if st.session_state.get("authenticated"):
    uploaded_file = st.file_uploader("Upload Image for Detection", type=["jpg", "png", "jpeg"], key="img")

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0)

        with torch.no_grad():
           output = model(img_tensor)[0] 

        output = np.array(output)
        detections = output[0]
        classes = ['glove', 'goggles', 'helmet', 'mask', 'no-suit', 'no_glove',
           'no_goggles', 'no_helmet', 'no_mask', 'no_shoes', 'shoes', 'suit']

        violations = []
        event_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        # Collect unique labels (first instance per label)
        seen_labels = set()
        for i, det in enumerate(detections):
            if det.shape[0] >= 6:
               x1, y1, x2, y2 = map(int, det[:4])
               conf = float(det[4])
               cls = int(np.argmax(det[5:]))
               label = classes[cls] if cls < len(classes) else f"unknown_{cls}"

               # Skip duplicates
               if label in seen_labels:
                continue
               seen_labels.add(label)

               cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
               cv2.putText(img_cv, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
               violations.append({"label": label, "confidence": conf})

        _, buffer = cv2.imencode(".jpg", img_cv)
        image_bytes = buffer.tobytes()
        image_url = upload_to_s3(image_bytes)

        with engine.begin() as conn:
        # Insert metadata
            conn.execute(
                text("""
                     INSERT INTO violations_meta (id, timestamp, anomaly_detected, image_path)
                     VALUES (:id, :timestamp, :anomaly_detected, :image_path)
                """),
               {
                "id": event_id,
                "timestamp": timestamp,
                "anomaly_detected": bool(violations),
                "image_path": image_url
               }
          )

        # Prepare and insert violation details
            rows = [{"id": event_id, "label": v["label"], "confidence": v["confidence"]} for v in violations]
        
            if rows:  # Only insert if there are violations
                conn.execute(
                text("""
                INSERT INTO violation_details (id, label, confidence)
                VALUES (:id, :label, :confidence)
                """), rows
        ) 
                 
        if violations:
            # Extract all labels
            all_labels = [v['label'] for v in violations]

            # Get unique labels preserving order
            unique_labels = list(dict.fromkeys(all_labels))

            no_violations = [v for v in unique_labels if "no" in v.lower()]

            # Join them as a string for email content
            labels_str = ",".join(no_violations)
            
            if labels_str:
                # Send email alert
                send_email_alert(ALERT_EMAIL, labels_str, img_cv)
                st.success("Alert email sent successfully!")
            else:
                # No critical violations
                st.warning("No alert email sent. Violations are not critical.")
                
        st.image(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)), caption='Detection Result', use_container_width=True)

# -------------------- Download log as CSV --------------------
    with st.expander("📊 View Logs & Export"):
        df_logs = pd.read_sql("SELECT * FROM violations_meta", engine)
        st.dataframe(df_logs)
        st.download_button("Download Logs", df_logs.to_csv().encode(), file_name="violations.csv", mime="text/csv")

# -------------------- Ask question to AI reg violation--------------------
    with st.expander("🤖 Ask the System (LangChain Agent)"):
        query = st.text_input("Ask a question about violations:")
        if query:
            db = SQLDatabase.from_uri(DB_URI)
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
            agent = SQLDatabaseChain.from_llm(llm, db, verbose=True)
            st.write(agent.run(query))


