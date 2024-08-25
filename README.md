# Research on Real-time Pedestrian Detection Algorithm of YOLOv10 under Complex Lighting and Occlusion Conditions
The paper has been submitted, and this is the code for the paper. As a junior undergraduate student not majoring in a computer science related field, I have always encountered a lot of difficulties when trying to reproduce the results of some projects or papers. Therefore, when implementing some of the code myself, I would like more novices to be able to see the results quickly and intuitively, instead of being stuck in a painful debug of configuring the environment and so on. The purpose of this document is to allow more newbies to follow the document step by step, until all the projects are completed quickly and the paper reproduced.

## Step1: Preparation
- Hardware: If you want to simply verify the effect, even a CPU will do; if you want to train your own YOLO target detection model, and apply it in “video game game auto-targeting”, or “video game daily quests auto-scrubbing”, then I suggest you to train your own YOLO target detection model, and apply it in “video game game auto-targeting”, or “video game daily quests auto-scrubbing”. If you want to train your own YOLO target detection model for “auto-targeting enemies in video games”, or “auto-scouring daily quests in video games”, then I suggest you to use GPUs with N graphics cards, or cloud GPUs for deployment.
- Software: Before deploying YOLOv10 locally, you need to install Python (version greater than or equal to 3.10) and Git, these two are essential, if you have the foundation, you can consider using Anaconda or Miniconda, or even Docker.

## Step2: Deployment Phase (Windows)
### Step2.1: Open Powershell or cmd
- First of all, you can create a new folder in D drive (**newbies please make sure your environment variables allow Python to read & run the files in D drive**), and then name this folder as yolo10PD (you can play around with the name for this step).

<div align=center>
<img src="https://github.com/weyumm/YOLOv10-Pedestrian-Detection/blob/main/docs_and_imgs/1-Create%20new%20folder.png" width="720" height="420"> 
</div>
- Then, if you're on Win11, you can launch PowerShell by right-clicking on an empty space in the folder once you're in that folder, and then launching PowerShell. or you can use the win key + x, and then select Terminal Administrator, and utilize the shortcut to, well, summon the terminal.
<div align=center>
<img src="docs_and_imgs/2-Open Powershell.png" width="720" height="420"> 
</div>

And, Powershell looks like this:
<div align=center>
<img src="docs_and_imgs/3-Powershell looks like this.png" width="720" height="420"> 
</div>
- Of course, if you're on Win11 and below, you can also choose to open the command line by going into the folder, clicking on the box at the top that displays the path, and typing cmd. Powershell has slightly different commands than cmd, so asking AI is a good option if you run into some reported errors.
<div align=center>
<img src="docs_and_imgs/4-Open cmd.png" width="720" height="420"> 
</div>
And, cmd looks like this:
<div align=center>
<img src="docs_and_imgs/5-cmd looks like this.png" width="720" height="420"> 
</div>

Of course, if you will jump to the directory by cd command, it is also possible.
### Step2.2: Create and activate the virtual environment
- Here, you just need to copy and paste the code line by line into the terminal and run it, and then a change similar to the one shown in the picture means your operation is successful!
- For Windows system, when you use Powershell, please paste the first code first.
```
    python -m venv yolovenv
```
 Then, to activate it, enter the following code 【again, this is the Powershell activation command, the activation environment in cmd makes a difference】
```
    yolovenv/Scripts/activate
```
You will see the green virtual environment as shown highlighted in Powershell, meaning the operation was successful.
<div align=center>
<img src="docs_and_imgs/6-Activate the environment in Powershell.png" width="720" height="420"> 
</div>
- For Windows, when you use Powershell, the operation is similar but slightly different, again first copy the code below
 ```
    python -m venv yolovenv
```
At this point, you want to activate the virtual environment using the call command by copying the following code
```
    call yolovenv\Scripts\activate.bat
```
Similarly, if you see something similar to the image below, it means that you created the virtual environment and activated it successfully using cmd.
<div align=center>
<img src="docs_and_imgs/7-Activate the environment in cmd.png" width="720" height="420"> 
</div>
<div align=center>
<img src="docs_and_imgs/8-Activate environment successfully in cmd.png" width="720" height="420"> 
</div>
Tip: My virtual environment is called “VLyoloPD”, while the virtual environment created after copying and pasting my code box is called “yolovenv”, don't get confused.

### Step2.3: Install third-party libraries
First of all, let's use Windows system as an example, copy and paste the code of **pip install** line by line.
- The first piece of code is to install the necessary base libraries.
```
    pip install supervision labelme labelme2yolo huggingface_hub google-cloud-audit-log
```
- The second code, if your device has NVIDIA graphics card, then you can consider installing pytorch, but it should be noted that, please go to the official website to install the appropriate version of your computer torch. here code is only as an example, if the installation of this step is particularly fast, then it is very likely that there are some problems with this step of the operation, because the torch library is very large, it will take a long time to install it. It will take a long time to install. If you don't have a NVIDIA graphics card, and you just want to see the effect of pedestrian detection, then you don't need to run this code.
```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- The third piece of code, which is the open source code for installing YOLOv10, is an essential step, but many problems may be encountered.
```
    pip install git+https://github.com/THU-MIG/yolov10.git
```
If this code fails to execute, consider installing using SSH.
First, make sure you've generated an SSH key and added the public key to your GitHub account. You can generate the SSH key (if you haven't already) using the following command:
```
    ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```
Then, add the generated public key (usually the contents of the ~/.ssh/id_rsa.pub file) to your GitHub account (look for video tutorials to follow along).
Finally, run this code
```
    pip install git+ssh://git@github.com/THU-MIG/yolov10.git
```
- The fourth piece of code is to run the PYQT interface that I've wrapped up
```
    pip install pyside6 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### Step2.4: Install third-party libraries (short tutorial)
If you want to run the code with CPU, just copy the following three lines:
```
    pip install supervision labelme labelme2yolo huggingface_hub google-cloud-audit-log
    pip install git+https://github.com/THU-MIG/yolov10.git
    pip install pyside6 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
If you want to run the code on the GPU, just copy the following four lines:
```
    pip install supervision labelme labelme2yolo huggingface_hub google-cloud-audit-log
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # The second line depends on your device model, download pytorch yourself from the official website
    pip install git+https://github.com/THU-MIG/yolov10.git
    pip install pyside6 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
If you're experiencing difficulty downloading at home, try Gitee.
```
    pip install supervision labelme labelme2yolo huggingface_hub google-cloud-audit-log
    pip install git+https://gitee.com/weyumm/yolov10
    pip install pyside6 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
If you are using Linux, the operation is similar to Windows, the code is shown below.
```
  python -m venv yolovenv
  source yolovenv/bin/activate
  pip install supervision labelme labelme2yolo huggingface_hub google-cloud-audit-log
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install git+https://github.com/THU-MIG/yolov10.git
  pip install pyside6 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
After the above process, we have the virtual environment ready.
Please remember to **activate the environment** every time you are ready to do any work with YOLOv10!
In Windows, you need to cd to the yolo10PD folder (created by Step2.1) and then activate the virtual environment using the following:

Powershell for Windows:
 ```
    yolovenv/Scripts/activate
 ```
cmd in Windows：
 ```
    call yolovenv\Scripts\activate.bat
 ```
Linux：
 ```
    source yolovenv/bin/activate
 ```
## Step3: Downloading the dataset, model & testing video (release) in progress
## Step3.1: Models
The models are divided into two sets, model_base (here is the pre-training model for YOLOv10, totaling 6, as well as a benchmark model for YOLOv8).

The other set is model_PD (here is the pre-training model for pedestrian detection, also 6, pre-trained by me in combination with the visual big model).

For details, you can check the Release section.

### Step3.2: Dataset
The original dataset is from Caltech pedestrian detection dataset, which mainly includes

Training set + test set: seq format data;
Pedestrian labeling data:vbb(video bounding box) format data, this format data is mainly the pedestrian bounding box in dataset 1. Since we need mainly image format data for training, we need to convert the data in .seq .vbb format to image.

[Visit the Caltech website](https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)

[Google Drive for this dataset]([https://github.com](https://drive.google.com/drive/folders/1CzNwN9QUXvJLYQzzkk-EVDLB0B6bAwJv))
Here is the dataset download script
 ```
    #!/bin/bash

# # Get files from Google Drive
annolist=(https://drive.google.com/file/d/1EsAL5Q9FfOQls28qYmr2sO6rha1d4YVz/view?usp=sharing)
for dir in ${annolist[@]};do
    echo ${dir}
    echo ${dir:32:33}
    URL="https://drive.google.com/u/0/uc?export=download&id=${dir:32:33}"
    wget --load-cookies /tmp/cookies.txt "https://drive.google.com/u/0/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${dir:32:33}" -O anno.zip && rm -rf /tmp/cookies.txt
    unzip anno.zip
done
rm -rf anno.zip

# USA set00-set10
setlist=(https://drive.google.com/file/d/1tPeaQr1cVmSABNCJQsd8OekOZIjpJivj/view?usp=sharing
https://drive.google.com/file/d/1apo5VxoZA5m-Ou4GoGR_voUgLN0KKc4g/view?usp=sharing
https://drive.google.com/file/d/1yvfjtQV6EnKez6TShMZQq_nkGyY9XA4q/view?usp=sharing
https://drive.google.com/file/d/1jvF71hw4ztorvz0FWurtyCBs0Dy_Fh0A/view?usp=sharing
https://drive.google.com/file/d/11Q7uZcfjHLdwpLKwDQmr5gT8LoGF82xY/view?usp=sharing
https://drive.google.com/file/d/1Q0pnxM5cnO8MJJdqzMGIEryZaEKk_Un_/view?usp=sharing
https://drive.google.com/file/d/1ft6clVXKdaxFGeihpth_jdBQxOIirSk7/view?usp=sharing
https://drive.google.com/file/d/1-E_B3iAPQKTvkZ8XyuLcE2Lytog3AofW/view?usp=sharing
https://drive.google.com/file/d/1oXCaTPOV0UYuxJJrxVtY9_7byhOLTT8G/view?usp=sharing
https://drive.google.com/file/d/1f0mpL2C2aRoF8bVex8sqWaD8O3f9ZgfR/view?usp=sharing
https://drive.google.com/file/d/18TvsJ5TKQYZRlj7AmcIvilVapqAss97X/view?usp=sharing
)

for setdir in ${setlist[@]};do
    echo ${setdir}
    echo ${setdir:32:33}
    URL="https://drive.google.com/u/0/uc?export=download&id=${setdir:32:33}"
    wget --load-cookies /tmp/cookies.txt "https://drive.google.com/u/0/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${setdir:32:33}" -O set.tar && rm -rf /tmp/cookies.txt
    tar -xvf set.tar
done
rm -rf set.tar
 ```
seq turn into jpg
```
#!/usr/bin/env python
# encoding: utf-8
# Deal with .seq format for video sequence
# The .seq file is combined with images,
# so I split the file into several images with the image prefix
# "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46".

import os.path
import fnmatch
import shutil


def open_save(file, savepath):
    """
    read .seq file, and save the images into the savepath

    :param file: .seq文件路径
    :param savepath: 保存的图像路径
    :return:
    """

    # 读入一个seq文件，然后拆分成image存入savepath当中
    f = open(file, 'rb+')
    # 将seq文件的内容转化成str类型
    string = f.read().decode('latin-1')

    # splitstring是图片的前缀，可以理解成seq是以splitstring为分隔的多个jpg合成的文件
    splitstring = "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46"

    # split函数做一个测试,因此返回结果的第一个是在seq文件中是空，因此后面省略掉第一个
    """
    >>> a = ".12121.3223.4343"
    >>> a.split('.')
    ['', '12121', '3223', '4343']
    """
    # split .seq file into segment with the image prefix
    strlist = string.split(splitstring)
    f.close()
    count = 0
    # delete the image folder path if it exists
    # if os.path.exists(savepath):
    #     shutil.rmtree(savepath)
    # create the image folder path
    # if not os.path.exists(savepath):
    #     os.makedirs(savepath)
    # deal with file segment, every segment is an image except the first one
    for img in strlist:
        filename = str(count) + '.jpg'
        filenamewithpath = savepath + '_' + filename #os.path.join(savepath, filename)
        # abandon the first one, which is filled with .seq header
        if count > 0:
            i = open(filenamewithpath, 'wb+')
            i.write(splitstring.encode('latin-1'))
            i.write(img.encode('latin-1'))
            i.close()
        count += 1


if __name__ == "__main__":
    rootdir = "/workspace/dataset/zfjuan/data/CaltechPedestrian/"
    saveroot = "/workspace/dataset/zfjuan/data/CaltechPedestrian/caltech_voc/JPEGImages"

    # walk in the rootdir, take down the .seq filename and filepath
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            # check .seq file with suffix
            # fnmatch 全称是 filename match，主要是用来匹配文件名是否符合规则的
            if fnmatch.fnmatch(filename, '*.seq'):
                # take down the filename with path of .seq file
                thefilename = os.path.join(parent, filename)
                # create the image folder by combining .seq file path with .seq filename
                parent_path = parent
                parent_path = parent_path.replace('\\', '/')
                thesavepath = saveroot + '/' + parent_path.split('/')[-1] + '_' + filename.split('.')[0]
                print("Filename=" + thefilename)
                print("Savepath=" + thesavepath)
                open_save(thefilename, thesavepath)


```

The expanded dataset Campus is photographed by itself, and when you want to use it, please remember to integrate Campus into the Caltech dataset by dividing it into a training set, a test set, and a cross-validation set.

### Step3.3: Test Videos
The test videos are again four in number and are already in Release. Compress them with datavideo.7z.

**This part, the model, dataset, and test videos you're going to download are all organized in Release

If, however, you're not too good with Github, you can get these in my web share.

## Step4: Start reproducing my work

```
    git clone https://github.com/weyumm/YOLOv10-Pedestrian-Detection
```
First clone my code locally, then open verify_my_work and start the initial reproduction.

There are four files in this folder, I call them “Dock Four”, I believe they can bring you infinite happiness. Each py file has its own skill, each py file has its own trick, its fighting spirit and patience are even more amazing, and the careful bilingual comments will bring you surprises!

 
- First, Camera Screenshot.py is for taking screenshots, you just have to activate the environment and cd to the verify_my_work directory and then run the Python file in the terminal.
<div align=center>
<img src="verify_my_work/9-Take screenshots.png" width="720" height="420"> 
</div>
As you can see in the picture, when the activation of the environment and the cd command are done, you need to type in the terminal:
```
    python Camera Screenshot.py
```
I have written comments in English and Chinese for each line of this code, but I will repeat the procedure again here:
1. Press 'Esc' to exit the program
2. Press 's' to save the current frame.
- Second, Video Frame-cutting.py can output a locally stored video after frame-cutting, but please remember to open the Python file and change the path yourself.
<div align=center>
<img src="verify_my_work/10-Modify the video path.png" width="720" height="420"> 
</div>
```
    python Video Frame-cutting.py
```
- Third, yolov10-detect.py can open the local computer's webcam and then detect the screen in front of the webcam in real time.
```
    python yolov10-detect.py
```
- Fourth, yolov10-detect-video.py can call the weight file that has been trained and then detect the frame of this video, also remember to modify the path.
<div align=center>
<img src="verify_my_work/11-Modify the model path.png" width="720" height="420"> 
</div>
```
    python yolov10-detect-video.py
```
- Finally, putting the model you downloaded with the test video in the verify_my_work folder like in the image below will save you a lot of unnecessary tweaking of environment variables.
<div align=center>
<img src="verify_my_work/12-initial state.png" width="720" height="420"> 
</div>
<div align=center>
<img src="verify_my_work/13-Assembly complete.png" width="720" height="420"> 
</div>
Then, please try to do a pedestrian detection in the terminal, using commands such as python yolov10-detect-video.py, if the above operations, you are correct, then you will see the pop-up window of the successful execution, here is an example of the test video test003.mp4:
<div align=center>
<img src="docs_and_imgs/14-Example test chart.png" width="720" height="420"> 
</div>
<div align=center>
<img src="docs_and_imgs/15-stop detection.png" width="720" height="420"> 
</div>
Press Ctrl+c to stop detection
## Step5: Using the wrapper
First of all, when you clone my project, please cd to the appropriate directory:
```
    cd D:\yolo10PD\pyqt
```
Then, if you are a Chinese user, run this code in a virtual environment：
```
    python base_camera_cn.py
```
Please follow the order in the figure, first select the weight file, then initialize the weight file, and then try 【photo inspection and display the results】, 【video inspection and display the results】, 【camera real-time inspection and display the results】 in turn
<div align=center>
<img src="docs_and_imgs/16-GUI screen cn.png" width="720" height="420"> 
</div>
if you are a English user, run this code in a virtual environment：
```
    python base_camera_en.py
```
<div align=center>
<img src="docs_and_imgs/17-GUI screen en.png" width="720" height="420"> 
</div>
Please follow the order in the figure, first select the weight file, then initialize the weight file, and then try 【photo inspection and display the results】, 【video inspection and display the results】, 【camera real-time inspection and display the results】 in turn


## Step5: Train your own model
You can use the cloud platform, Roboflow for training, [Roboflow official website](roboflow.com “Click to visit Roboflow”)
You can also train locally with the labelme plugin, this part has already downloaded the relevant libraries when configuring the environment, you just need to run the following code
```
    labelme2yolo --json_dir  D:\yolo10PD\output_images
```
However, please note that Roboflow cloud platform can automatically resize the image to 640x640, but the local labelme does not, so please manually resize the image to 640x640 yourself.
Then, **change the paths in the data.yaml file** in the Campus dataset to the absolute paths where you put your own training set, test set, and validation set.
Finally, enter the training instructions in the terminal:
```
    yolo detect train data=D:/yolo10PD/dataset/Campus/data.yaml model=yolov10n.pt epochs=10 batch=8 imgsz=640 device=0
```
The path corresponding to data is exactly the absolute path where the yaml file is located. model can specify the (YOLOn, s, m, b, l, x) model you want to use, and is compatible from YOLOv8 to YOLOv10.

If you have one NVIDIA graphics card, you can write device=0. If you have two, write device=1.

If you only use the CPU, then you can remove device and not write
```
    yolo detect train data=D:/yolo10PD/dataset/Campus/data.yaml model=yolov10n.pt epochs=10 batch=8 imgsz=640
```
Above, that is the complete process, I plan to record the operation of the video, to assist newcomers to learn better.
————————————————————————————————————————————————————————————————

# 复杂光照和遮挡条件下 YOLOv10 的行人实时检测算法研究
论文已投递，这是论文的代码，作为一名非计算机科学相关专业的低年级本科生，我在尝试复现一些项目或者论文结果时总遇到许多困难。因此，当自己实现了一些代码后，我希望更多新手能快速而直观地看到成效，而不是被困于配置环境等等的痛苦debug中。本说明文档旨在让更多的新手，也能随着文档一步步操作，直至快速完成所有项目与论文复现。

## Step1：准备阶段
- 硬件：如果想要简单验证效果，哪怕是CPU也可以；如果你希望训练自己的YOLO目标检测模型，应用在“电子游戏的游戏自动锁定敌方”，亦或者“电子游戏的每日任务自动刷副本”，那么我建议你使用带有N显卡的GPU，亦或者使用云GPU进行部署。
- 软件：本地部署YOLOv10前，你需要先安装好Python（版本大于等于3.10）和Git，这两个是必备的，如果你有基础，可以考虑使用Anaconda或者Miniconda，甚至Docker等。

## Step2：部署阶段（Windows）
### Step2.1：打开Powershell或cmd
- 首先，你可以在D盘新建一个文件夹（**新手请确保你的环境变量能让Python读取与运行D盘的文件**），然后将这个文件夹命名为yolo10PD（这一步的名字可以自行发挥）。

<div align=center>
<img src="https://github.com/weyumm/YOLOv10-Pedestrian-Detection/blob/main/docs_and_imgs/1-Create%20new%20folder.png" width="720" height="420"> 
</div>
- 然后，如果你是Win11的系统，可以通过在进入该文件夹后，右键文件夹中空白的地方，然后启动PowerShell。或者可以使用win键+x，然后选择终端管理员，利用快捷键，召唤终端。
<div align=center>
<img src="docs_and_imgs/2-Open Powershell.png" width="720" height="420"> 
</div>
  
并且，Powershell看起来是这样：
<div align=center>
<img src="docs_and_imgs/3-Powershell looks like this.png" width="720" height="420"> 
</div>
- 当然，如果你是Win11及以下的系统，也可以选择进入该文件夹后，点击上方显示路径的框格，然后输入cmd，打开命令行。Powershell与cmd的命令略有不同，如果你遇到了一些报错，询问AI是一个很好的选择。
<div align=center>
<img src="docs_and_imgs/4-Open cmd.png" width="720" height="420"> 
</div>
并且，cmd看起来是这样：
<div align=center>
<img src="docs_and_imgs/5-cmd looks like this.png" width="720" height="420"> 
</div>

当然，如果你会通过cd指令来跳转目录，也是可以的。
### Step2.2：创建并激活虚拟环境
- 在这里，你只要逐行复制粘贴代码到终端里运行，然后出现类似如图所示的变化，意味着你的操作成功了
- 对于Windows系统，当你使用Powershell时，请先粘贴第一段代码
  ```
    python -m venv yolovenv
  ```
  然后，激活它，输入以下代码【再次提示，这是Powershell的激活指令，cmd中激活环境有所区别】
  ```
    yolovenv/Scripts/activate
  ```
  你将在Powershell中看到如图所示的绿色虚拟环境被高亮显示，意味着操作成功。
<div align=center>
<img src="docs_and_imgs/6-Activate the environment in Powershell.png" width="720" height="420"> 
</div>
- 对于Windows系统，当你使用Powershell时，操作类似，但略有区别，同样先复制下方代码
  ```
    python -m venv yolovenv
  ```
  此时，你要使用call指令来激活虚拟环境，复制下方代码
  ```
    call yolovenv\Scripts\activate.bat
  ```
  同样的，如果你看到类似如下图的内容，意味着你创建了虚拟环境，并使用cmd成功激活。
<div align=center>
<img src="docs_and_imgs/7-Activate the environment in cmd.png" width="720" height="420"> 
</div>
<div align=center>
<img src="docs_and_imgs/8-Activate environment successfully in cmd.png" width="720" height="420"> 
</div>
  温馨提示：我的虚拟环境名字叫"VLyoloPD"，而复制粘贴我代码框后，创建的虚拟环境名字叫"yolovenv"，不要搞混。

### Step2.3：安装第三方库
首先，我们还是用Windows系统做例子，一行行复制粘贴**pip install**的代码
- 第一段代码是安装必备的基础库
```
    pip install supervision labelme labelme2yolo huggingface_hub google-cloud-audit-log
```
- 第二段代码，如果你的设备有NVIDIA的显卡，那么可以考虑安装pytorch。但需要注意的是，请前往官网安装适合自己电脑版本的torch。这里的代码仅作为示例，如果这一步安装起来特别快，那么很有可能这一步的操作有些问题，因为torch库很大，要安装的话还是比较久的。如果没有NVIDIA的显卡，只想简单看看行人检测的效果，那么这一段代码可以不用运行。
```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- 第三段代码，是安装YOLOv10的开源代码，这一步是必不可少的，但也可能遇到许多问题。
```
    pip install git+https://github.com/THU-MIG/yolov10.git
```
如果这段代码执行失败，可以考虑使用SSH进行安装。
首先，确保你已经生成了 SSH 密钥，并将公钥添加到你的 GitHub 账户中。你可以使用以下命令生成 SSH 密钥（如果你还没有）：
```
    ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```
然后，将生成的公钥（通常是 ~/.ssh/id_rsa.pub 文件中的内容）添加到你的 GitHub 账户中（可以找视频教程跟着操作）。
最后，运行这段代码
```
    pip install git+ssh://git@github.com/THU-MIG/yolov10.git
```
- 当然，即便你开了Watt Toolkit(Steam++)或者FastGithub之类的工具进行加速，但你的电脑仍然无法顺利把YOLOv10的代码clone到本地后pip install，那么，你也可以考虑使用Gitee或者kkgithub等镜像网站，然后安装到本地。
- 第四段代码，是为了运行我封装好的PYQT界面
```
    pip install pyside6 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Step2.4：安装第三方库(简明教程)
如果你想用CPU跑代码，只需要复制以下三行：
```
    pip install supervision labelme labelme2yolo huggingface_hub google-cloud-audit-log
    pip install git+https://github.com/THU-MIG/yolov10.git
    pip install pyside6 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
如果你想用GPU跑代码，只需要复制以下四行：
```
    pip install supervision labelme labelme2yolo huggingface_hub google-cloud-audit-log
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    #第二行要根据自己的设备型号来，在官网自己下载pytorch
    pip install git+https://github.com/THU-MIG/yolov10.git
    pip install pyside6 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
如果你在国内遇到难以下载的情况，可以试试Gitee。
```
    pip install supervision labelme labelme2yolo huggingface_hub google-cloud-audit-log
    pip install git+https://gitee.com/weyumm/yolov10
    pip install pyside6 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
如果你使用Linux系统，操作与Windows系统是差不多的，代码如下所示。
```
  python -m venv yolovenv
  source yolovenv/bin/activate
  pip install supervision labelme labelme2yolo huggingface_hub google-cloud-audit-log
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install git+https://github.com/THU-MIG/yolov10.git
  pip install pyside6 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
经过上述流程后，我们已经把虚拟环境准备完毕了。
在你每次准备使用YOLOv10进行任何工作时，请记得**激活环境**！！
在Windows中，你需要在cd到yolo10PD文件夹（Step2.1建立的），然后使用以下方式激活虚拟环境：

Windows的Powershell：
 ```
    yolovenv/Scripts/activate
 ```
Windows的cmd：
 ```
    call yolovenv\Scripts\activate.bat
 ```
Linux：
 ```
    source yolovenv/bin/activate
 ```
## Step3：下载数据集、模型与测试视频(release)中
### Step3.1：模型
模型分为两套，分别是model_base（这里是YOLOv10的预训练模型，共计6个，以及一个YOLOv8的基准模型）。

另一套是model_PD（这里是做行人检测的预训练模型，同样是6个，由我结合视觉大模型预训练而成。）

详细内容可以查看Release部分。

### Step3.2：数据集
原始数据集来自Caltech行人检测数据集，该数据集主要包括

训练集+测试集：seq格式的数据；
行人标签数据:vbb(video bounding box)格式的数据，该格式数据主要是数据集1中的行人bounding box。由于我们training时需要的主要是图像格式的数据，所以需要将.seq .vbb这两个格式的数据转换为图像。

[访问Caltech官网](https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)

[该数据集的谷歌网盘]([https://github.com](https://drive.google.com/drive/folders/1CzNwN9QUXvJLYQzzkk-EVDLB0B6bAwJv))
以下为数据集下载脚本
 ```
    #!/bin/bash

# # Get files from Google Drive
annolist=(https://drive.google.com/file/d/1EsAL5Q9FfOQls28qYmr2sO6rha1d4YVz/view?usp=sharing)
for dir in ${annolist[@]};do
    echo ${dir}
    echo ${dir:32:33}
    URL="https://drive.google.com/u/0/uc?export=download&id=${dir:32:33}"
    wget --load-cookies /tmp/cookies.txt "https://drive.google.com/u/0/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${dir:32:33}" -O anno.zip && rm -rf /tmp/cookies.txt
    unzip anno.zip
done
rm -rf anno.zip

# USA set00-set10
setlist=(https://drive.google.com/file/d/1tPeaQr1cVmSABNCJQsd8OekOZIjpJivj/view?usp=sharing
https://drive.google.com/file/d/1apo5VxoZA5m-Ou4GoGR_voUgLN0KKc4g/view?usp=sharing
https://drive.google.com/file/d/1yvfjtQV6EnKez6TShMZQq_nkGyY9XA4q/view?usp=sharing
https://drive.google.com/file/d/1jvF71hw4ztorvz0FWurtyCBs0Dy_Fh0A/view?usp=sharing
https://drive.google.com/file/d/11Q7uZcfjHLdwpLKwDQmr5gT8LoGF82xY/view?usp=sharing
https://drive.google.com/file/d/1Q0pnxM5cnO8MJJdqzMGIEryZaEKk_Un_/view?usp=sharing
https://drive.google.com/file/d/1ft6clVXKdaxFGeihpth_jdBQxOIirSk7/view?usp=sharing
https://drive.google.com/file/d/1-E_B3iAPQKTvkZ8XyuLcE2Lytog3AofW/view?usp=sharing
https://drive.google.com/file/d/1oXCaTPOV0UYuxJJrxVtY9_7byhOLTT8G/view?usp=sharing
https://drive.google.com/file/d/1f0mpL2C2aRoF8bVex8sqWaD8O3f9ZgfR/view?usp=sharing
https://drive.google.com/file/d/18TvsJ5TKQYZRlj7AmcIvilVapqAss97X/view?usp=sharing
)

for setdir in ${setlist[@]};do
    echo ${setdir}
    echo ${setdir:32:33}
    URL="https://drive.google.com/u/0/uc?export=download&id=${setdir:32:33}"
    wget --load-cookies /tmp/cookies.txt "https://drive.google.com/u/0/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${setdir:32:33}" -O set.tar && rm -rf /tmp/cookies.txt
    tar -xvf set.tar
done
rm -rf set.tar
 ```
seq转为jpg
```
#!/usr/bin/env python
# encoding: utf-8
# Deal with .seq format for video sequence
# The .seq file is combined with images,
# so I split the file into several images with the image prefix
# "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46".

import os.path
import fnmatch
import shutil


def open_save(file, savepath):
    """
    read .seq file, and save the images into the savepath

    :param file: .seq文件路径
    :param savepath: 保存的图像路径
    :return:
    """

    # 读入一个seq文件，然后拆分成image存入savepath当中
    f = open(file, 'rb+')
    # 将seq文件的内容转化成str类型
    string = f.read().decode('latin-1')

    # splitstring是图片的前缀，可以理解成seq是以splitstring为分隔的多个jpg合成的文件
    splitstring = "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46"

    # split函数做一个测试,因此返回结果的第一个是在seq文件中是空，因此后面省略掉第一个
    """
    >>> a = ".12121.3223.4343"
    >>> a.split('.')
    ['', '12121', '3223', '4343']
    """
    # split .seq file into segment with the image prefix
    strlist = string.split(splitstring)
    f.close()
    count = 0
    # delete the image folder path if it exists
    # if os.path.exists(savepath):
    #     shutil.rmtree(savepath)
    # create the image folder path
    # if not os.path.exists(savepath):
    #     os.makedirs(savepath)
    # deal with file segment, every segment is an image except the first one
    for img in strlist:
        filename = str(count) + '.jpg'
        filenamewithpath = savepath + '_' + filename #os.path.join(savepath, filename)
        # abandon the first one, which is filled with .seq header
        if count > 0:
            i = open(filenamewithpath, 'wb+')
            i.write(splitstring.encode('latin-1'))
            i.write(img.encode('latin-1'))
            i.close()
        count += 1


if __name__ == "__main__":
    rootdir = "/workspace/dataset/zfjuan/data/CaltechPedestrian/"
    saveroot = "/workspace/dataset/zfjuan/data/CaltechPedestrian/caltech_voc/JPEGImages"

    # walk in the rootdir, take down the .seq filename and filepath
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            # check .seq file with suffix
            # fnmatch 全称是 filename match，主要是用来匹配文件名是否符合规则的
            if fnmatch.fnmatch(filename, '*.seq'):
                # take down the filename with path of .seq file
                thefilename = os.path.join(parent, filename)
                # create the image folder by combining .seq file path with .seq filename
                parent_path = parent
                parent_path = parent_path.replace('\\', '/')
                thesavepath = saveroot + '/' + parent_path.split('/')[-1] + '_' + filename.split('.')[0]
                print("Filename=" + thefilename)
                print("Savepath=" + thesavepath)
                open_save(thefilename, thesavepath)


```

扩充数据集Campus由自己拍摄，当你想要使用时，请记得把Campus融入Caltech数据集，划分为训练集、测试集以及交叉验证集。

### Step3.3：测试视频
测试视频同样有四个，并且已经在Release中了。用datavideo.7z压缩。

**这一部分，你要下载的模型、数据集、测试视频都整理在Release中了**

如果，你不太会使用Github，可以在我的网盘分享中拿到这些内容。


## Step4：开始复现我的工作
```
    git clone https://github.com/weyumm/YOLOv10-Pedestrian-Detection
```
先把我的代码clone到本地，然后打开verify_my_work，开始初步的复现工作。

这个文件夹有四个文件，我把他们叫做“码头四”，相信它们能给你带来无限快乐。每段py文件都身怀绝技，每段py文件都有独门绝招，斗志和耐性更是技惊四座，精心的双语注释更给你带来意外惊喜呀！！

- 第一，Camera Screenshot.py是用来截图的，你只要激活环境，然后cd到verify_my_work目录，然后在终端里运行Python文件即可。
<div align=center>
<img src="verify_my_work/9-Take screenshots.png" width="720" height="420"> 
</div>
如图所示，当激活环境和cd指令完成后，你需要在终端里输入：
```
    python Camera Screenshot.py
```
此代码的每一行，我都撰写了中英文注释，但在此再次重复一下操作：
1. 按下 'Esc' 键退出程序
2. 按下 's' 键保存当前帧
- 第二，Video Frame-cutting.py可以把存储在本地的一段视频进行抽帧后输出，但请务必记得打开Python文件，自己修改下路径。
<div align=center>
<img src="verify_my_work/10-Modify the video path.png" width="720" height="420"> 
</div>
```
    python Video Frame-cutting.py
```
- 第三，yolov10-detect.py可以打开本地电脑的摄像头，然后实时检测摄像头前的画面。
```
    python yolov10-detect.py
```
- 第四，yolov10-detect-video.py可以调用已经被训练好的权重文件，然后检测这段视频的画面，也要记得修改下路径。
<div align=center>
<img src="verify_my_work/11-Modify the model path.png" width="720" height="420"> 
</div>
```
    python yolov10-detect-video.py
```
- 最后，把你下载的模型与测试视频，像下图一样放在verify_my_work文件夹下，能省去很多不必要的调整环境变量的麻烦。
<div align=center>
<img src="verify_my_work/12-initial state.png" width="720" height="420"> 
</div>
<div align=center>
<img src="verify_my_work/13-Assembly complete.png" width="720" height="420"> 
</div>
那么，请试着在终端里，使用python yolov10-detect-video.py等指令进行行人检测吧，如果上述操作，你都是正确的，那么你将会看到执行成功的弹窗，这里以测试视频 test003.mp4为例：
<div align=center>
<img src="docs_and_imgs/14-Example test chart.png" width="720" height="420"> 
</div>
<div align=center>
<img src="docs_and_imgs/15-stop detection.png" width="720" height="420"> 
</div>
按Ctrl+c停止检测
## Step5：使用封装程序
首先，当你clone我的项目后，请cd到相应目录：
```
    cd D:\yolo10PD\pyqt
```
然后，如果你是中文版用户，请在虚拟环境中运行
```
    python base_camera_cn.py
```
请按照图中的顺序，先选择权重文件，然后初始化权重文件，然后依次尝试【照片检验并显示结果】、【视频检验并显示结果】、【摄像头实时检验并显示结果】
<div align=center>
<img src="docs_and_imgs/16-GUI screen cn.png" width="720" height="420"> 
</div>
如果你是英文版用户，请在虚拟环境中运行
```
    python base_camera_en.py
```
<div align=center>
<img src="docs_and_imgs/17-GUI screen en.png" width="720" height="420"> 
</div>
请按照图中的顺序，先选择权重文件，然后初始化权重文件，然后依次尝试【照片检验并显示结果】、【视频检验并显示结果】、【摄像头实时检验并显示结果】

## Step5：训练自己的模型
可以使用云端平台，Roboflow进行训练，[Roboflow官网](roboflow.com "点击访问 Roboflow")
也可以在本地用labelme插件来训练，这部分已经在配置环境时下载了相关的库，你只需要运行以下代码
```
    labelme2yolo --json_dir  D:\yolo10PD\output_images
```
但要注意的是，Roboflow云端平台可以自动把图像尺寸调整为640x640，但本地的labelme并不可以，因此请自己手动将图片调整为640x640的尺寸。
然后，修改Campus数据集中的data.yaml文件中的路径，改为你自己放训练集、测试集、验证集的绝对路径。
最后，在终端中输入训练指令：
```
    yolo detect train data=D:/yolo10PD/dataset/Campus/data.yaml model=yolov10n.pt epochs=10 batch=8 imgsz=640 device=0
```
data所对应的路径，正是yaml文件所在的绝对路径，model可以指定你要使用的（YOLOn、s、m、b、l、x）模型，从YOLOv8到YOLOv10都兼容。

如果你有一块NVIDIA显卡，可以写device=0，有两块，就写device=1。

如果你只用CPU，那么可以去掉device不写
```
    yolo detect train data=D:/yolo10PD/dataset/Campus/data.yaml model=yolov10n.pt epochs=10 batch=8 imgsz=640
```
以上，即是完整流程了，我计划再录制操作视频，辅助新手们更好地学习。
