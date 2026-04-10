#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LeLaN 策略的 ROS 部署入口。

这个脚本负责“最后一段”目标靠近：
1. 持续读取当前相机画面，
2. 用 CLIP 对语言 prompt 编码，
3. 用 LeLaN 预测一小段未来线速度/角速度，
4. 只执行当前时刻的第一步控制并发布到 /cmd_vel。
"""

import sys
sys.path.insert(0, '../../train')

#ROS
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

#torch
import torch
import torch.nn.functional as F

#others
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

from utils import msg_to_pil, to_numpy, transform_images, transform_images_lelan, load_model, pil2cv, cv2pil

# 部署参数：目标文本、配置文件、模型权重、相机类型。
parser = argparse.ArgumentParser(description="give prompts")
parser.add_argument('-p', '--prompt', type=str, help="prompts of the target objects")
parser.add_argument('-c', '--config', type=str, default = "../../train/config/lelan.yaml", help="path for the config file (.yaml)")
parser.add_argument('-m', '--model', type=str, default = "../model_weights/wo_col_loss_wo_temp.pth", help="path for the config file (.yaml)")
parser.add_argument('-r', '--ricoh', type=bool, default = True, help="True: Ricoh Theta S, False: others")
args = parser.parse_args()

flag_once = 0
store_hist = 0
init_hist = 0
image_hist = []

# 相机输入 topic。实际部署时通常要按机器人环境修改。
TOPIC_NAME_CAMERA = '/cv_camera_node/image_raw'
#TOPIC_NAME_CAMERA = '/usb_cam/image_raw'

# Ricoh Theta S 圆形成像区域的遮罩和裁切参数。
xc = 310
yc = 321
yoffset = 310 
xoffset = 310
xyoffset = 280
xplus = 661
XY = [(xc-xyoffset, yc-xyoffset), (xc+xyoffset, yc+xyoffset)]

# 按启动参数加载策略配置和权重。
model_config_path = args.config     # We provide two sample yaml files, "../../train/config/lelan_col.yaml" or "../../train/config/lelan.yaml"
#ckpth_path = "/mnt/sdb/models/wo_col_loss_wo_temp.pth" 
ckpth_path = args.model
# Please down load our checkpoints, with_col_loss.pth, wo_col_loss.pth, and wo_col_loss_wo_temp.pth
# with_col_loss.pth: finetuned model considering collision avoindace loss. We feed the history of image (1 second ago) as well as the current image and the prompt
# wo_col_loss.pth: trained model NOT considering collision avoindace loss. We feed the history of image (1 second ago) as well as the current image and the prompt
# wo_col_loss_wo_temp.pth: trained model NOT considering collision avoindace loss. We feed the current image and the prompt. Simplest model with our core idea.

#ckpth_path = "/mnt/sdb/models/3.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(model_config_path, "r") as f:
    model_params = yaml.safe_load(f)

if os.path.exists(ckpth_path):
    print(f"Loading model from {ckpth_path}")
else:
    raise FileNotFoundError(f"Model weights not found at {ckpth_path}")

if args.ricoh:
    print("Reading", TOPIC_NAME_CAMERA, "as a spherical camera")
else:
    print("Reading", TOPIC_NAME_CAMERA, "as NOT a spherical camera (canocical or fisheye camera)")
        
model = load_model(
    ckpth_path,
    model_params,
    device,
)
model = model.to(device)
model.eval()  

def preprocess_camera(msg):
    global pub, bridge

    # Ricoh 的原始画面是圆形成像，所以这里会先做遮罩，再裁成中心方形视图。
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
        cv_cutimg = bridge.imgmsg_to_cv2(msg, "bgr8")
        
    msg_img = bridge.cv2_to_imgmsg(cv_cutimg, 'bgr8')
    pub.publish(msg_img)
        
def callback_lelan(msg_1):
    global init_hist, image_hist
    global flag_once, feat_text

    # 这个回调完成一次完整的闭环策略推理。
    if True:
        im = msg_to_pil(msg_1)

        # 用第一帧初始化历史缓存，避免时序模型刚启动时没有上一帧可用。
        if init_hist == 0:
            for ih in range(10):
                image_hist.append(im)
            init_hist = 1

        # transform_images_lelan 会返回两部分：
        # - 低分辨率的历史帧堆叠，给时序分支用
        # - 全分辨率当前帧，给 FiLM 条件视觉编码器用
        im_obs = [image_hist[9], im]
        obs_images, obs_current = transform_images_lelan(im_obs, model_params["image_size"], center_crop=False)              
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1)
        batch_obs_images = obs_images.to(device)
        batch_obs_current = obs_current.to(device)
        
        with torch.no_grad():
            # 同一次导航里目标文本通常不变，所以 prompt 特征会被重复复用。
            if flag_once == 0:
                obj_inst = args.prompt    #"office chair"
                                                                                                                                                                                                                                         
                batch_obj_inst = clip.tokenize(obj_inst).to(device)            
                feat_text = model("text_encoder", inst_ref=batch_obj_inst)
            else:
                flag_once = 1

            # 在线有两个版本：
            # - lelan：当前帧 + prompt
            # - lelan_col：短历史帧 + 当前帧 + prompt
            if model_params["model_type"] == "lelan_col":
                obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, feat_text=feat_text.to(torch.float32), current_img=batch_obs_current) 
            elif model_params["model_type"] == "lelan":
                obsgoal_cond = model("vision_encoder", obs_img=batch_obs_current, feat_text=feat_text.to(torch.float32)) 
            linear_vel, angular_vel = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

        # 策略会预测一小段控制序列，但部署时只执行第一步，下一次定时器再重算。
        vt = linear_vel.cpu().numpy()[0,0]
        wt = angular_vel.cpu().numpy()[0,0]
        print("linear vel.", vt, "angular vel.", wt)
        # 按机器人安全上限裁剪速度，同时尽量保留原始输出对应的转弯半径。
        maxv = 0.3
        maxw = 0.5           
        msg_pub = Twist()        
        if np.absolute(vt) <= maxv:
            if np.absolute(wt) <= maxw:
                msg_pub.linear.x = vt
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = wt
            else:
                rd = vt/wt
                msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                msg_pub.linear.y = 0.0
                msg_pub.linear.z = 0.0
                msg_pub.angular.x = 0.0
                msg_pub.angular.y = 0.0
                msg_pub.angular.z = maxw * np.sign(wt)
        else:
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
                    msg_pub.linear.x = maxv * np.sign(vt)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxv * np.sign(wt) / np.absolute(rd)
                else:
                    msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxw * np.sign(wt)

        # 发布给机器人底盘的最终控制命令。
        pub_vel.publish(msg_pub)

        # 更新历史缓存，让下一次推理能把这帧当作时序上下文。
        image_histx = [im] + image_hist[0:9]
        image_hist = image_histx

def subscriber_callback(msg):
    global latest_image
    # 这里只保留最新的一张处理后图像，真正推理由定时器统一触发。
    latest_image = msg
    
def timer_callback(_):
    global latest_image
    # 将相机回调频率和策略运行频率解耦。
    if latest_image is not None:
        callback_lelan(latest_image)
        latest_image = None    

bridge = CvBridge()
latest_image = None

if __name__ == '__main__':    
    # ROS 数据流：相机 -> 预处理 -> 最新图像缓存 -> 定时器触发策略。
    rospy.init_node('LeLaN_col', anonymous=False)

    # subscribe of topics
    rospy.Subscriber('/image_processed', Image, subscriber_callback)
    rospy.Subscriber(TOPIC_NAME_CAMERA, Image, preprocess_camera)
    rospy.Timer(rospy.Duration(0.1), timer_callback)
            
    # publisher of topics
    pub_vel = rospy.Publisher('/cmd_vel', Twist,queue_size=1) #velocities for the robot control
    pub = rospy.Publisher("/image_processed", Image, queue_size = 1)
	
    print('waiting message .....')
    rospy.spin()
