from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('/root/code/ultralytics/cfg/models/v8/myyolov8.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

if __name__ == '__main__':
    # Use the model
    results = model.train(data='/root/code/ultralytics/cfg/datasets/mycoco128.yaml', epochs=300)  # train the model
    results = model.val()  # evaluate model performance on the validation set
    success = YOLO("yolov8n.pt").export(format="onnx")  # export a model to ONNX format