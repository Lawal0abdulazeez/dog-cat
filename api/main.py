

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import numpy as np
from PIL import Image
import io

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../models/1")

CLASS_NAMES = ["Cat", "Dog"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image



@app.post("/predict")
async def predict(
    files: List[UploadFile] = File(...)
):
    # Process the uploaded images
    processed_images = []
    for file in files:
        image = read_file_as_image(await file.read())

        # Resize the image to match the expected input shape
        resized_img = Image.fromarray(image).resize((180, 180))

        # Convert the image to a numpy array
        img_array = np.array(resized_img)

        processed_images.append(img_array)

    # Convert the list of processed images to a numpy array
    processed_img_batch = np.array(processed_images)

    # Make predictions with the model
    predictions = MODEL.predict(processed_img_batch)

    # Assuming CLASS_NAMES and the read_file_as_image function are defined elsewhere in your code
    predicted_classes = [CLASS_NAMES[np.argmax(pred)] for pred in predictions]
    confidences = [float(np.max(pred)) for pred in predictions]

    results = []
    for predicted_class, confidence in zip(predicted_classes, confidences):
        results.append({'class': predicted_class, 'confidence': confidence})

    return JSONResponse(content=results)


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)