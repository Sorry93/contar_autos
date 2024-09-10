import cv2

# Carga el video
capture = cv2.VideoCapture("autos.mp4")

# Carga el modelo de detección
carros = cv2.CascadeClassifier("cars.xml")

while True:
    ret, frames = capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars = carros.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(30, 30))

    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Redimensiona el marco a un tamaño específico (por ejemplo, 640x480)
    resized_frame = cv2.resize(frames, (640, 480))

    # Muestra la imagen redimensionada en una ventana llamada "Video"
    cv2.imshow("Video", resized_frame)

    if cv2.waitKey(33) == 27:  # Espera por 33 ms o hasta que presiones ESC
        break

capture.release()
cv2.destroyAllWindows()

