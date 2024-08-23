import cv2
import os

# 定义视频的路径
# Define the path to the video
video_path = "C:/Users/114514/Desktop/1-videoedge-recognize/merge.mp4"

# 创建一个VideoCapture对象
# Create a VideoCapture object
video = cv2.VideoCapture(video_path)

# 获取视频的帧率（fps）和总帧数
# Get video frame rate (fps) and total frame count
fps = int(video.get(cv2.CAP_PROP_FPS))
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frames_per_second = 3  # 每秒提取的帧数
# Number of frames to extract per second

# 保存帧图像的目录
# Directory to save the frames
output_dir = "C:/Users/114514/Desktop/1-videoedge-recognize/lunkuo1/"
os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在则创建
# Create the directory if it doesn't exist

# 遍历视频并保存帧
# Step through the video and save frames
for i in range(0, frame_count, round(fps / frames_per_second)):
    # 设置视频文件的当前帧位置
    # Set the current frame position of the video file
    video.set(cv2.CAP_PROP_POS_FRAMES, i)
    
    # 读取帧
    # Read the frame
    success, frame = video.read()
    
    # 检查帧是否成功读取
    # Check if the frame was successfully read
    if not success:
        break  # 如果读取失败，退出循环
        # Break the loop if reading fails
    
    # 创建帧文件名
    # Create the frame file name
    frame_filename = os.path.join(output_dir, f"{i}.jpg")
    
    # 保存帧为图像
    # Save the frame as an image
    cv2.imwrite(frame_filename, frame)

# 释放视频捕捉对象
# Release the video capture object
video.release()

print("Frames have been extracted and saved.")
# Frames have been extracted and saved.
