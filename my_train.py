from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('ultralytics/cfg/models/v8/myyolov8.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

if __name__ == '__main__':
    # Use the model
    results = model.train(data='ultralytics/cfg/datasets/mycoco128.yaml', epochs=50)  # train the model
    results = model.val()  # evaluate model performance on the validation set
    results = model('ultralytics-main/data/mycoco/images/0a71179f4c0c38ab8309bbf313ea69b2.jpg') # predict on an image
    success = YOLO("yolov8n.pt").export(format="onnx")  # export a model to ONNX format