import tensorflow as tf
import cv2
import numpy as np


model = tf.keras.models.load_model("trained model/trained_model.h5")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  
    loss="categorical_crossentropy",  
    metrics=["accuracy"]
)

IMG_SIZE = 224 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break  
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))  
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  

    predictions = model.predict(img)
    class_index = np.argmax(predictions)  
    confidence = np.max(predictions)  

    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
              "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    label = labels[class_index] if class_index < len(labels) else "Unknown"

    cv2.putText(frame, f"{label} ({confidence*100:.2f}%)", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

cap.release()
cv2.destroyAllWindows()
