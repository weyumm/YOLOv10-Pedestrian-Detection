import cv2
import os
import supervision as sv
from ultralytics import YOLOv10

# 加载 YOLOv10 模型
# Load the YOLOv10 model
model = YOLOv10('yolov10n.pt')

# 创建注释器对象
# Create annotator objects
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 视频文件路径（假设视频文件在当前文件夹下的 datavideo 文件夹中）
# Video file path (assuming the video file is in the 'datavideo' folder in the current directory)
video_path = 'datavideo/test002.mp4'

# 打开视频文件
# Open the video file
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the video file.")
    exit()

# 循环读取并处理视频帧
# Loop to read and process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 对当前帧进行目标检测
    # Perform object detection on the current frame
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # 标注带有边界框的图像
    # Annotate the image with bounding boxes
    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    
    # 标注带有标签的图像
    # Annotate the image with labels
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
    
    # 显示标注后的图像
    # Display the annotated image
    cv2.imshow('Video', annotated_image)
    
    # 检测按键输入，如果按下了 ESC 键则退出循环
    # Check for key input; if ESC key is pressed, exit the loop
    k = cv2.waitKey(1)
    if k % 256 == 27:
        print("Escape hit, closing...")
        break

# 释放资源
# Release resources
cap.release()
cv2.destroyAllWindows()
