# Production Quality Detection System

This project is an embedded system for real-time product quality supervision.  
It uses **Raspberry Pi 5 + Arducam Camera** with **YOLOv8 (TensorFlow)** to detect defects in production lines.  
A **web interface with Firebase** is used for remote monitoring.  

## 🛠️ Technologies
- Raspberry Pi 5
- Python, OpenCV, TensorFlow, YOLOv8
- Firebase (web dashboard)
- Linux Embedded

## 📂 Project Structure
- `/src` → source code
- `/models` → YOLOv8  tensorflow trained models
- `/docs` → documentation and diagrams

## 🚀 How to run
```bash
git clone https://github.com/siwar-zahi/production-quality-detection.git
cd src
python main.py
📊 Results

95% accuracy for defect detection

Real-time monitoring at 15 FPS
