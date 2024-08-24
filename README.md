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
