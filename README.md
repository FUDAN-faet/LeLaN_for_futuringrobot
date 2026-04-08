# 本项目基于开源项目Learining A Language-conditioned Navigation Policy（后续均称为lelan）修改，用于轮式底盘机器人的控制

# lelan节点启动指令
cd ~/navi_ws/src/LeLaN
source ~/venv/lelan_jazzy_env/bin/activate
source /opt/ros/jazzy/setup.bash

export PYTHONPATH=/home/zme/navi_ws/src/LeLaN/diffusion_policy:$PYTHONPATH

python lelan_ros2/lelan_ros2/lelan_policy_node.py --ros-args \
  -p image_topic:=/camera/color/image_raw \
  -p cmd_vel_topic:=/cmd_vel_test \
  -p prompt:=chair \
  -p config_path:=/home/zme/navi_ws/src/LeLaN/train/config/lelan.yaml \
  -p model_path:=/home/zme/navi_ws/src/LeLaN/deployment/model_weights/wo_col_loss_wo_temp.pth \
  -p use_ricoh:=false \
  -p timer_period:=0.1 \
  -p max_linear_vel:=0.3 \
  -p max_angular_vel:=0.5

# 系统相机启动指令
source /opt/ros/jazzy/setup.bash
source ~/navi_ws/install/setup.bash

ros2 run usb_cam usb_cam_node_exe --ros-args \
  --params-file ~/navi_ws/config/usb_cam_lelan.yaml \
  -r image_raw:=/camera/color/image_raw

> 可以利用笔记本自带的摄像头查看lelan输出的控制速度，控制输出的捕获话题为：ros2 topic echo /cmd_vel_test

### 2026/04/08已经在本机i7 14650HX + 16GB + 5060-8G上试运行成功了该项目，启动指令如上。