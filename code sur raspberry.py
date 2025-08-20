import cv2
import numpy as np
import time
import threading
import queue
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import datetime
from PIL import Image, ImageDraw
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import ssl

# === CONFIGURATION ===
DETECTION_MODEL_PATH = "/home/pi/modele_detection1.pt"
CLASSIFICATION_MODEL_PATH = "/home/pi/mon_modele.h5"
CLASS_INTERVAL = 1.0
DETECTION_CONFIDENCE = 0.5
MAX_QUEUE_SIZE = 1

EMAIL_CONFIG = {
    'sender': 'siwarzahi10@gmail.com',
    'password': 'pelr ennj odam nmrs',
    'receiver': 'siwarzahi10@gmail.com',
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 465
}

# === CHARGEMENT DES MODÈLES ===
try:
    detection_model = YOLO(DETECTION_MODEL_PATH)
    classification_model = load_model(CLASSIFICATION_MODEL_PATH)
    print("Modèles chargés avec succès")
except Exception as e:
    print(f"Erreur de chargement: {e}")
    exit()

# === INITIALISATION CAMÉRA ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la caméra")
    exit()

# === STRUCTURES PARTAGÉES ===
frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
detection_event = threading.Event()
latest_results = None
lock = threading.Lock()

def preprocess_for_classification(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return (frame / 255.0).astype(np.float32)[np.newaxis, ...]

def detection_worker():
    global latest_results
    while not detection_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            results = detection_model(frame, conf=DETECTION_CONFIDENCE, imgsz=320, device='cpu', verbose=False)
            with lock:
                latest_results = results[0] if results else None
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Erreur détection: {e}")

def draw_enhanced_detections(frame, results):
    if not results:
        return frame
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls_id = int(box.cls[0])
        label = f"{results.names[cls_id]} {conf:.2f}"
        color = (0, 165, 255) if "soudure" in results.names[cls_id].lower() else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

def generate_report(frame, results, confidence):
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_image_path = os.path.join(report_dir, f"defaut_detecte_{timestamp}.png")
    report_text_path = os.path.join(report_dir, f"defaut_detecte_{timestamp}.txt")

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls_id = int(box.cls[0])
        label = f"{results.names[cls_id]} {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 20), label, fill="red")

    pil_img.save(report_image_path)

    with open(report_text_path, "w") as f:
        f.write(f"# Rapport de détection\n\nGénéré le {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Confiance globale: {confidence:.2f}\n---\n\n## Détails :\n")
        for i, box in enumerate(results.boxes, 1):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls_id = int(box.cls[0])
            f.write(f"* Défaut #{i}\n  - Type: {results.names[cls_id]}\n  - Confiance: {conf*100:.2f}%\n  - Position: ({x1}, {y1}) à ({x2}, {y2})\n\n")

    return report_image_path, report_text_path

def send_email_with_report(report_image_path, report_text_path):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_CONFIG['sender']
    msg['To'] = EMAIL_CONFIG['receiver']
    msg['Subject'] = f"Défaut détecté - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"

    msg.attach(MIMEText("Bonjour,\n\nUn défaut a été détecté. Veuillez trouver ci-joint :\n- L'image annotée\n- Le rapport détaillé\n\nCordialement,\nSystème automatisé", 'plain'))

    with open(report_image_path, 'rb') as f:
        img = MIMEImage(f.read())
        img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(report_image_path))
        msg.attach(img)

    with open(report_text_path, 'rb') as f:
        txt = MIMEText(f.read().decode('utf-8'), 'plain', 'utf-8')
        txt.add_header('Content-Disposition', 'attachment', filename=os.path.basename(report_text_path))
        msg.attach(txt)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'], context=context) as server:
            server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
            server.send_message(msg)
        print("Email envoyé avec succès")
    except Exception as e:
        print(f"Erreur email: {e}")

# === THREAD DE DÉTECTION ===
detection_thread = threading.Thread(target=detection_worker)
detection_thread.daemon = True
detection_thread.start()

last_class_time = 0
status = "INITIALISATION"
confidence = 0.0
frame_count = 0
start_time = time.time()
global_start_time = start_time

# === BOUCLE PRINCIPALE ===
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur de capture")
            break

        display_frame = frame.copy()

        # Amélioration qualité
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        display_frame = cv2.filter2D(display_frame, -1, sharpen_kernel)
        display_frame = cv2.convertScaleAbs(display_frame, alpha=1.2, beta=10)

        current_time = time.time()
        if current_time - last_class_time > CLASS_INTERVAL:
            try:
                pred = classification_model.predict(preprocess_for_classification(frame), verbose=0)[0][0]
                is_conforme = pred <= 0.5
                status = "CONFORME" if is_conforme else "NON CONFORME"
                confidence = (1 - pred) if is_conforme else pred
                last_class_time = current_time

                if not is_conforme:
                    if frame_queue.empty():
                        frame_queue.put(frame.copy())
                    time.sleep(0.5)
                    with lock:
                        if latest_results:
                            report_img, report_txt = generate_report(frame, latest_results, confidence)
                            email_thread = threading.Thread(
                                target=send_email_with_report,
                                args=(report_img, report_txt)
                            )
                            email_thread.start()
            except Exception as e:
                print(f"Erreur classification: {e}")
                status = "ERREUR"

        # Statut
        status_color = (0, 255, 0) if status == "CONFORME" else (0, 0, 255)
        cv2.putText(display_frame, f"{status} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Détéctions + FPS uniquement si non conforme
        if status == "NON CONFORME":
            with lock:
                if latest_results:
                    display_frame = draw_enhanced_detections(display_frame, latest_results)

            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)  # FPS en jaune

        # Temps écoulé permanent
        exec_time = time.time() - global_start_time
        mins, secs = divmod(int(exec_time), 60)
        cv2.putText(display_frame, f"Temps écoulé: {mins:02d}:{secs:02d}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 0), 1)

        cv2.imshow("Inspection USB - RPi5", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    detection_event.set()
    detection_thread.join(timeout=1)
    cap.release()
    cv2.destroyAllWindows()
    total_exec_time = time.time() - global_start_time
    print(f"Arrêt du programme - Temps total: {total_exec_time:.2f} secondes")
