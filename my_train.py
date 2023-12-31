from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('/root/code/myyolo/ultralytics/cfg/models/v8/myyolov8.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('/root/code/myyolo/yolov8n.pt')

if __name__ == '__main__':
    # Use the model
    results = model.train(data='/root/code/myyolo/ultralytics/cfg/datasets/mycoco128.yaml',workers=1, epochs=300,batch=32)  # train the model
    success = YOLO('/root/code/myyolo/yolov8n.pt').export(format="onnx")  # export a model to ONNX format
