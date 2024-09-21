from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt") # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model

model.train(data="data.yaml", epochs=10)
metrics = model.val()

results = model("C:/Users/praph/Downloads/TestImage1.jpg",show = True,conf = 0.8) 
# results = model.train(data="datasets/data.yaml", epochs=3)  # train the model

# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format 