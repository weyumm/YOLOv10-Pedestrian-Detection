# YOLOv10-Pedestrian-Detection




# 复杂光照和遮挡条件下 YOLOv10 的行人实时检测算法研究
论文已投递，这是论文的代码，作为一名初学者，本说明文档旨在让更多的新手，也能随着文档一步步操作，直至快速完成所有项目与论文复现。

## Step1：准备阶段
- 硬件：如果想要简单验证效果，哪怕是CPU也可以；如果你希望训练自己的YOLO目标检测模型，应用在“电子游戏的游戏自动锁定敌方”，亦或者“电子游戏的每日任务自动刷副本”，那么我建议你使用带有N显卡的GPU，亦或者使用云GPU进行部署。
- 软件：本地部署YOLOv10前，你需要先安装好Python（版本大于等于3.10）和Git，这两个是必备的，如果你有基础，可以考虑使用Anaconda或者Miniconda，甚至Docker等。

## Step2：部署阶段（Windows）
### Step2.1：打开Powershell或cmd
- 首先，你可以在D盘新建一个文件夹（**新手请确保你的环境变量能让Python读取与运行D盘的文件**），然后将这个文件夹命名为yolo10PD（这一步的名字可以自行发挥）。

<div align=center>
<img src="https://github.com/weyumm/YOLOv10-Pedestrian-Detection/blob/main/docs_and_imgs/1-Create%20new%20folder.png" width="720" height="420"> 
</div>
- 然后，如果你是Win11的系统，可以通过在进入该文件夹后，右键文件夹中空白的地方，然后启动PowerShell。
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

### Step2.3：安装第三方库(简明教程)
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
