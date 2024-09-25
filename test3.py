import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="driving_behavior_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get the input shape
input_shape = input_details[0]['shape']

# Define the class names (adjust these based on your specific classes)
class_names = ['DangerousDriving', 'Distracted', 'Drinking', 'SafeDriving', 'SleepyDriving', 'Yawn']

def preprocess_image(frame, target_size):
    # Resize the frame
    resized = cv2.resize(frame, target_size)
    # Convert to RGB (OpenCV uses BGR by default)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize the image
    normalized = rgb / 255.0
    # Add batch dimension
    input_data = np.expand_dims(normalized, axis=0).astype(np.float32)
    return input_data

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Preprocess the frame
    input_data = preprocess_image(frame, (input_shape[1], input_shape[2]))

    # Set the tensor to point to the input data
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Process the results
    class_index = np.argmax(output_data[0])
    class_name = class_names[class_index]
    confidence = output_data[0][class_index]
    
    # Display the result on the frame
    cv2.putText(frame, f'{class_name}: {confidence:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Driver Drowsiness Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()