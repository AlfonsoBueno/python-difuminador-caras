import cv2 as cv

clasificador = "haarcascade_frontalface_default.xml"
video = "video.mp4"

# Importamos el video 
cap = cv.VideoCapture(video)

# Obtenemos el ancho y el alto del video original
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Definimos el codec y creamos el objeto VideoWriter
# que es el video que se creará de salida
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
out = cv.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# Verificamos si la apertura del video y el objeto VideoWriter fueron exitosos
if not cap.isOpened():
    print("Error al abrir el archivo de video")
if not out.isOpened():
    print("Error al abrir el archivo de salida")

# Comienza el bucle del video
while cap.isOpened():
    ret, frame = cap.read()

    # Si el frame es correcto, ret es True; si es False es que ya ha dejado de leerlos
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Se basa a b&n porque así trabaja mejor
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    #Detecta las caras
    face_cascade = cv.CascadeClassifier(clasificador)

    # Guarda las caras
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=12, minSize=(45, 45))

    # Emborrona las caras
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        blurred_face = cv.resize(cv.resize(face, (w // 12, h // 12)), (w, h))
        frame[y:y + h, x:x + w] = blurred_face

    # Escribimos el frame procesado en el objeto VideoWriter
    out.write(frame)
    
    # Esto es para verlo por pantalla
    cv.imshow('frame', frame)

    # Si pulsas q se sale
    if cv.waitKey(1) == ord('q'):
        break

# Liberamos los objetos cap y out, y destruimos todas las ventanas
out.release()
cap.release()
cv.destroyAllWindows()
