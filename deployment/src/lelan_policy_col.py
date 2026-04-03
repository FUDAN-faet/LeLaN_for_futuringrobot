#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
# 为了直接复用 train/ 目录下的模型定义与工具，这里把训练代码目录加入 Python 搜索路径。
sys.path.insert(0, '../../train')

# ROS 相关依赖：负责接收图像、发布速度。
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

# PyTorch：负责模型推理。
import torch
import torch.nn.functional as F

# 其余通用依赖：图像处理、配置读取、文本编码、命令行参数等。
import os
import cv2
import yaml
import clip
import argparse
import numpy as np
from PIL import ImageDraw
from PIL import Image as PILImage
from cv_bridge import CvBridge, CvBridgeError
import torchvision.transforms as T

# 工具函数：
# - msg_to_pil: ROS Image -> PIL Image
# - transform_images_lelan: 把图像整理成 LeLaN 需要的输入格式
# - load_model: 根据 yaml 和权重加载模型
# - pil2cv/cv2pil: PIL 和 OpenCV 之间相互转换
from utils import msg_to_pil, to_numpy, transform_images, transform_images_lelan, load_model, pil2cv, cv2pil

# -----------------------------
# 命令行参数
# -----------------------------
# 这个脚本的目标很明确：
# 输入“目标物体的文本描述 + 实时相机图像”，
# 输出“机器人底盘速度 /cmd_vel”。
parser = argparse.ArgumentParser(description="give prompts")
# -p / --prompt: 目标物体的文本提示，例如 "chair"、"office chair"。
parser.add_argument('-p', '--prompt', type=str, help="prompts of the target objects")
# -c / --config: 训练时对应的 yaml 配置，推理时也要用它来恢复模型结构。
parser.add_argument('-c', '--config', type=str, default = "../../train/config/lelan.yaml", help="path for the config file (.yaml)")
# -m / --model: 训练好的模型权重路径。
parser.add_argument('-m', '--model', type=str, default = "../model_weights/wo_col_loss_wo_temp.pth", help="path for the config file (.yaml)")
# -r / --ricoh: 是否使用 Ricoh Theta S 全景相机；如果不是，就按普通相机走。
parser.add_argument('-r', '--ricoh', type=bool, default = True, help="True: Ricoh Theta S, False: others")
args = parser.parse_args()

# 只在第一次推理时做一次文本编码，后续重复使用。
flag_once = 0
# 这些变量用于带历史图像版本的 LeLaN。
store_hist = 0
init_hist = 0
image_hist = []

# -----------------------------
# 相机话题名
# -----------------------------
# 这里是硬编码的话题名，后续迁移到 ROS2 时很适合改成参数。
TOPIC_NAME_CAMERA = '/cv_camera_node/image_raw'
#TOPIC_NAME_CAMERA = '/usb_cam/image_raw'

# -----------------------------
# Ricoh Theta S 的图像裁剪参数
# -----------------------------
# 这组参数用于从全景图里裁出实际用于导航的局部视野。
xc = 310
yc = 321
yoffset = 310 
xoffset = 310
xyoffset = 280
xplus = 661
XY = [(xc-xyoffset, yc-xyoffset), (xc+xyoffset, yc+xyoffset)]

# -----------------------------
# 模型配置与权重
# -----------------------------
# 这里会读取训练时的 yaml，以恢复模型结构。
# 这个脚本既可以加载 lelan，也可以加载 lelan_col。
model_config_path = args.config     # We provide two sample yaml files, "../../train/config/lelan_col.yaml" or "../../train/config/lelan.yaml"
#ckpth_path = "/mnt/sdb/models/wo_col_loss_wo_temp.pth" 
ckpth_path = args.model
# Please down load our checkpoints, with_col_loss.pth, wo_col_loss.pth, and wo_col_loss_wo_temp.pth
# with_col_loss.pth: finetuned model considering collision avoindace loss. We feed the history of image (1 second ago) as well as the current image and the prompt
# wo_col_loss.pth: trained model NOT considering collision avoindace loss. We feed the history of image (1 second ago) as well as the current image and the prompt
# wo_col_loss_wo_temp.pth: trained model NOT considering collision avoindace loss. We feed the current image and the prompt. Simplest model with our core idea.

#ckpth_path = "/mnt/sdb/models/3.pth"

# 默认优先用 GPU 推理。
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 读取模型参数。
with open(model_config_path, "r") as f:
    model_params = yaml.safe_load(f)

# 检查权重文件是否存在。
if os.path.exists(ckpth_path):
    print(f"Loading model from {ckpth_path}")
else:
    raise FileNotFoundError(f"Model weights not found at {ckpth_path}")

# 根据是否为全景相机，提示用户当前采用哪条预处理路径。
if args.ricoh:
    print("Reading", TOPIC_NAME_CAMERA, "as a spherical camera")
else:
    print("Reading", TOPIC_NAME_CAMERA, "as NOT a spherical camera (canocical or fisheye camera)")

# 真正加载模型：
# - 根据 yaml 组装网络结构
# - 再把 checkpoint 权重灌进去
model = load_model(
    ckpth_path,
    model_params,
    device,
)
model = model.to(device)
model.eval()  # 推理模式：关闭 dropout / 使用推理行为。

def preprocess_camera(msg):
    global pub, bridge

    # 这一步只做图像预处理，不做模型推理。
    # 如果是 Ricoh 全景相机，需要先裁掉无效区域，再做旋转和翻转。
    if args.ricoh:
        cv2_msg_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        pil_img = cv2pil(cv2_msg_img)
        fg_img = PILImage.new('RGBA', pil_img.size, (0, 0, 0, 255))
        draw=ImageDraw.Draw(fg_img)
        draw.ellipse(XY, fill = (0, 0, 0, 0))
        pil_img.paste(fg_img, (0, 0), fg_img.split()[3])
        cv2_img = pil2cv(pil_img)
        cv_cutimg = cv2_img[yc-xyoffset:yc+xyoffset, xc-xyoffset:xc+xyoffset]
        cv_cutimg = cv2.transpose(cv_cutimg)
        cv_cutimg = cv2.flip(cv_cutimg, 1)
    else:
        # 普通相机直接取原图。
        cv_cutimg = bridge.imgmsg_to_cv2(msg, "bgr8")

    # 预处理后的图像重新发布到 /image_processed，
    # 后续真正的 LeLaN 推理只订阅这个预处理后的话题。
    msg_img = bridge.cv2_to_imgmsg(cv_cutimg, 'bgr8')
    pub.publish(msg_img)

def callback_lelan(msg_1):
    global init_hist, image_hist
    global flag_once, feat_text

    # 真正的 LeLaN 推理逻辑都在这个回调里。
    if True:
        # ROS Image -> PIL，后续交给图像变换函数。
        im = msg_to_pil(msg_1)
        #im_crop = im.crop((0, 0, 560, 560))  

        # 第一次收到图像时，先把历史图像队列用当前帧填满。
        # 这样 lelan_col 在刚启动时也能拿到固定长度的历史输入。
        if init_hist == 0:
            for ih in range(10):
                image_hist.append(im)
            init_hist = 1

        # im_obs 是“历史图像 + 当前图像”的输入组合。
        # 对于 lelan 来说，最终只用当前图像；
        # 对于 lelan_col，会结合历史图像与当前图像。
        im_obs = [image_hist[9], im]
        # 转成模型需要的张量格式。
        obs_images, obs_current = transform_images_lelan(im_obs, model_params["image_size"], center_crop=False)              
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1)
        batch_obs_images = obs_images.to(device)
        batch_obs_current = obs_current.to(device)

        with torch.no_grad():
            # 文本编码只在开始时做一次。
            # 因为目标 prompt 通常在整个导航过程中不变。
            if flag_once == 0:
                obj_inst = args.prompt    #"office chair"

                batch_obj_inst = clip.tokenize(obj_inst).to(device)            
                feat_text = model("text_encoder", inst_ref=batch_obj_inst)
            else:
                flag_once = 1

            # -----------------------------
            # LeLaN 主推理
            # -----------------------------
            # lelan_col: 使用历史图像 + 当前图像 + 文本
            # lelan:     使用当前图像 + 文本
            if model_params["model_type"] == "lelan_col":
                obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, feat_text=feat_text.to(torch.float32), current_img=batch_obs_current) 
            elif model_params["model_type"] == "lelan":
                obsgoal_cond = model("vision_encoder", obs_img=batch_obs_current, feat_text=feat_text.to(torch.float32)) 

            # 速度头输出的是未来一小段控制序列。
            # 这里部署时只取第一个时刻的线速度和角速度来执行。
            linear_vel, angular_vel = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

        # 取出当前时刻的控制量。
        vt = linear_vel.cpu().numpy()[0,0]
        wt = angular_vel.cpu().numpy()[0,0]
        print("linear vel.", vt, "angular vel.", wt)

        # -----------------------------
        # 控制后处理：限幅
        # -----------------------------
        # 防止模型输出过大，直接把底盘打飞。
        maxv = 0.3
        maxw = 0.5           
        msg_pub = Twist()

        # 如果线速度和角速度都在允许范围内，直接发布。
        if np.absolute(vt) <= maxv:
            if np.absolute(wt) <= maxw:
                msg_pub.linear.x = vt
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = wt
            else:
                # 角速度过大时，按曲率近似缩放，尽量保持转弯半径一致。
                rd = vt/wt
                msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = maxw * np.sign(wt)
        else:
            # 线速度过大时，同样进行限幅。
            if np.absolute(wt) <= 0.001:
                msg_pub.linear.x = maxv * np.sign(vt)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = 0.0
            else:
                rd = vt/wt
                if np.absolute(rd) >= maxv / maxw:
                    # 优先限制线速度。
                    msg_pub.linear.x = maxv * np.sign(vt)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxv * np.sign(wt) / np.absolute(rd)
                else:
                    # 否则按角速度限幅调整线速度。
                    msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxw * np.sign(wt)

        # 发布到底盘控制话题。
        pub_vel.publish(msg_pub)

        # 更新历史图像队列：把当前帧塞到最前面，最旧的一帧丢掉。
        image_histx = [im] + image_hist[0:9]
        image_hist = image_histx

def subscriber_callback(msg):
    global latest_image
    # 不在订阅回调里直接跑推理，而是先缓存“最新一帧”。
    # 这样可以避免消息一来就同步阻塞。
    latest_image = msg

def timer_callback(_):
    global latest_image
    # 定时器驱动推理：
    # 如果缓存里有最新图像，就取出来做一次 LeLaN 推理。
    if latest_image is not None:
        callback_lelan(latest_image)
        latest_image = None

# cv_bridge 负责 ROS Image 与 OpenCV/PIL 的互转。
bridge = CvBridge()
latest_image = None

if __name__ == '__main__':    
    # -----------------------------
    # ROS 节点初始化
    # -----------------------------
    rospy.init_node('LeLaN_col', anonymous=False)

    # 订阅两个话题：
    # 1) 原始相机图像 -> preprocess_camera
    # 2) 预处理后图像 -> subscriber_callback
    rospy.Subscriber('/image_processed', Image, subscriber_callback)
    rospy.Subscriber(TOPIC_NAME_CAMERA, Image, preprocess_camera)

    # 定时器周期性触发推理，形成“图像 -> 推理 -> 控制”的闭环。
    rospy.Timer(rospy.Duration(0.1), timer_callback)

    # 发布两个话题：
    # 1) /cmd_vel：给底盘的速度命令
    # 2) /image_processed：预处理后的图像
    pub_vel = rospy.Publisher('/cmd_vel', Twist,queue_size=1) #velocities for the robot control
    pub = rospy.Publisher("/image_processed", Image, queue_size = 1)

    print('waiting message .....')
    # 进入 ROS 事件循环。
    rospy.spin()
