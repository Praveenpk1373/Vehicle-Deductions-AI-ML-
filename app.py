import streamlit as st
import cv2
import tempfile
import time
import pandas as pd
import numpy as np
import plotly.express as px
from ultralytics import YOLO
from datetime import datetime
from fpdf import FPDF
import os


MODEL_PATH = 'yolov8n.pt'
SUPPORTED_CLASSES = [
    "car", "bus", "truck", "motorbike", "bicycle", 
    "auto-rickshaw", "van", "ambulance", "scooter", "motorcycle"
]
LINE_POSITION = 300
CONFIDENCE_THRESHOLD = 0.3

st.set_page_config(page_title="Vehicle Detection & Society Analysis", layout="wide")
st.title("Vehicle Detection & Society Traffic Insights")
st.markdown("Upload a video to detect and analyze traffic using AI. Includes social impact insights!")

# --- VIDEO UPLOAD ---
uploaded_file = st.file_uploader("📤 Upload your traffic video", type=["mp4", "avi", "mov"])
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # --- DETECTION ---
    st.markdown("### 🔍 Detection Progress")
    progress = st.progress(0)
    stframe = st.empty()

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = YOLO(MODEL_PATH)
    vehicle_counts = {}
    seen_ids = set()
    frame_id = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        results = model.track(frame, persist=True)[0]
        boxes = results.boxes
        annotated_frame = frame.copy()
        cv2.line(annotated_frame, (0, LINE_POSITION), (width, LINE_POSITION), (0, 255, 255), 2)

        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else None
            class_name = model.names[cls_id]

            # Handle scooter vs motorcycle aliasing
            if class_name == "motorcycle":
                class_name = "scooter"

            if class_name in SUPPORTED_CLASSES and confidence > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if track_id is not None:
                    obj_uid = f"{class_name}-{track_id}"
                    if cy > LINE_POSITION and obj_uid not in seen_ids:
                        vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1
                        seen_ids.add(obj_uid)

                color = (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} ({confidence:.2f})"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        stframe.image(annotated_frame, channels="BGR", use_container_width=True)
        progress.progress(min(1.0, frame_id / total_frames))

    cap.release()
    total_time = time.time() - start_time
    fps = frame_id / total_time

    # --- RESULTS ---
    st.success("✅ Detection Completed")
    st.markdown(f"🕒 **Processing Time:** {total_time:.2f} seconds")
    st.markdown(f"🎥 **Frames Processed:** {frame_id} ({fps:.2f} FPS)")

    # Prepare dataframe
    df = pd.DataFrame({
        "Vehicle Type": list(vehicle_counts.keys()),
        "Count": list(vehicle_counts.values())
    }).sort_values(by="Count", ascending=False)

    st.markdown("### 📊 Vehicle Count Summary")
    st.dataframe(df)

    # Plot chart
    fig = px.bar(df, x="Vehicle Type", y="Count", color="Vehicle Type", title="Vehicle Count Chart")
    st.plotly_chart(fig)

    # Save CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"vehicle_report_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    with open(csv_filename, "rb") as f:
        st.download_button("⬇️ Download Report (CSV)", f, file_name=csv_filename)

    # --- SOCIETY ANALYSIS ---
    total_vehicles = df["Count"].sum()
    congestion_level = "High" if total_vehicles > 30 else "Moderate" if total_vehicles > 10 else "Low"
    noise_count = df[df["Vehicle Type"].isin(["truck", "bus", "auto-rickshaw"])].sum()["Count"]
    noise_level = "High" if noise_count > 10 else "Medium"
    most_common = df.loc[df["Count"].idxmax()]["Vehicle Type"] if not df.empty else "None"

    st.markdown("### 🧠 Society Impact Analysis")
    st.markdown(f"**Traffic Congestion:** 🚦 {congestion_level}")
    st.markdown(f"**Estimated Noise Level:** 🔊 {noise_level}")
    st.markdown(f"**Most Common Vehicle:** 🏍️ {most_common}")

    # --- PDF REPORT ---
    pdf_filename = f"vehicle_report_{timestamp}.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Traffic Analysis Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Total Vehicles Detected: {total_vehicles}", ln=True)
    pdf.cell(0, 10, f"Most Common Vehicle: {most_common}", ln=True)
    pdf.cell(0, 10, f"Traffic Congestion: {congestion_level}", ln=True)
    pdf.cell(0, 10, f"Estimated Noise Level: {noise_level}", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Vehicle Count Summary:", ln=True)
    pdf.set_font("Arial", "", 12)
    for _, row in df.iterrows():
        pdf.cell(0, 10, f"{row['Vehicle Type']}: {row['Count']}", ln=True)

    pdf.output(pdf_filename)

    with open(pdf_filename, "rb") as f:
        st.download_button("📄 Download PDF Report", f, file_name=pdf_filename)
