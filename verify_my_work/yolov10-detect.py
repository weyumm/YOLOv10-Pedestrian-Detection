import cv2
import supervision as sv
from ultralytics import YOLOv10

# 初始化 YOLOv10 模型
# Initialize the YOLOv10 model
model = YOLOv10(f'yolov10n.pt')

# 创建边界框注释器
# Create a bounding box annotator
bounding_box_annotator = sv.BoundingBoxAnnotator()

# 创建标签注释器
# Create a label annotator
label_annotator = sv.LabelAnnotator()  

# 打开摄像头
# Open the webcam
cap = cv2.VideoCapture(0)

# 设置视频帧的宽度和高度
# Set the frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 检查摄像头是否成功打开
# Check if the webcam was successfully opened
if not cap.isOpened():
    print("can't open the webcam")

while True:
    # 读取一帧视频
    # Read a frame from the video
    ret, frame = cap.read()  
    if not ret:
        break
    
    # 使用模型进行推断
    # Perform inference using the model
    results = model(frame)[0]
    
    # 从模型结果中获取检测结果
    # Get detections from the model results
    detections = sv.Detections.from_ultralytics(results)
    
    # 注释边界框
    # Annotate bounding boxes
    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    
    # 注释标签
    # Annotate labels
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
    
    # 显示注释后的图像
    # Display the annotated image
    cv2.imshow('Webcam', annotated_image)
    
    # 等待用户按键
    # Wait for a key press
    k = cv2.waitKey(1)
    
    # 按下 'Esc' 键退出程序
    # Exit the program if 'Esc' key is pressed
    if k % 256 == 27:
        print("Escape hit, closing...")
        break   

# 释放视频捕捉对象
# Release the video capture object
cap.release()

# 关闭所有OpenCV窗口
# Destroy all OpenCV windows
cv2.destroyAllWindows()
