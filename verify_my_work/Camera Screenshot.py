import cv2
import os

# 创建视频捕捉对象
# Create a video capture object
cap = cv2.VideoCapture(0)

# 设置视频帧的宽度和高度
# Set the frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# 检查摄像头是否成功打开
# Check if the webcam was successfully opened
if not cap.isOpened():
    print("can't open the webcam")

# 获取视频帧的宽度和高度
# Get the width and height of the video frame
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("pix:".format(width, height))

# 创建输出目录，如果不存在则创建
# Create an output directory if it does not exist
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# 初始化图像计数器
# Initialize the image counter
img_counter = 0

while True:
    # 读取一帧视频
    # Read a frame from the video
    ret, frame = cap.read()  
    if not ret:
        break

    # 显示摄像头画面
    # Display the webcam feed
    cv2.imshow('Webcam', frame)

    # 等待用户按键
    # Wait for a key press
    k = cv2.waitKey(1)
    
    # 按下 'Esc' 键退出程序
    # Exit the program if 'Esc' key is pressed
    if k % 256 == 27:
        print("Escape hit, closing...")
        break
    
    # 按下 's' 键保存当前帧
    # Save the current frame if 's' key is pressed
    elif k % 256 == ord('s'):
        img_name = os.path.join(output_dir, "opencv_frame_{}.png".format(img_counter))
        cv2.imwrite(img_name, frame)
        print("{} saved".format(img_name))
        img_counter += 1 

# 释放视频捕捉对象
# Release the video capture object
cap.release()

# 关闭所有OpenCV窗口
# Destroy all OpenCV windows
cv2.destroyAllWindows()
