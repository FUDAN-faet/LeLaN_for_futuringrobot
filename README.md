# LeLaN ROS2 Jazzy 最小运行说明

当前仓库推荐使用“分离运行时”的方式：

- `conda lelan` 负责运行模型推理服务
- ROS 2 Jazzy 的系统 Python 负责运行 `rclpy` bridge 节点

这样拆分是有意为之。因为 Jazzy 使用的是 Python 3.12，而你当前的 `lelan` conda 环境使用的是 Python 3.8，所以不建议把 `rclpy` 和 Torch 推理栈强行塞进同一个解释器里。

## 1. 修复 `import lelan`

```bash
source /home/zme/anaconda3/etc/profile.d/conda.sh
conda activate lelan
cd ~/navi_ws/src/lelan_for_futuringrobot
python -m pip install -e train/
python -c "import lelan; import vint_train; print('lelan ok')"
```

## 2. 准备权重

仓库本身不跟踪 `deployment/model_weights/` 目录，所以你需要自己创建这个目录，并把下面这些权重之一放进去：

- 带碰撞损失的最小部署方案：
  - `deployment/model_weights/with_col_loss.pth`
  - 配置文件：`train/config/lelan_col.yaml`
- 带历史帧但不带碰撞损失的方案：
  - `deployment/model_weights/wo_col_loss.pth`
  - 配置文件：`train/config/lelan_col.yaml`
- 最简单的“提示词到速度”模型：
  - `deployment/model_weights/wo_col_loss_wo_temp.pth`
  - 配置文件：`train/config/lelan.yaml`

```bash
cd ~/navi_ws/src/lelan_for_futuringrobot
mkdir -p deployment/model_weights
bash deployment/download_weights.sh
```

## 3. 在 conda 环境中启动推理服务

下面是带碰撞损失模型的示例：

```bash
source /home/zme/anaconda3/etc/profile.d/conda.sh
conda activate lelan
cd ~/navi_ws/src/lelan_for_futuringrobot
python deployment/src/lelan_inference_server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --prompt chair \
  --config train/config/lelan_col.yaml \
  --model deployment/model_weights/with_col_loss.pth
```

如果你目前只有 `wo_col_loss_wo_temp.pth`，就改成下面这条：

```bash
python deployment/src/lelan_inference_server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --prompt chair \
  --config train/config/lelan.yaml \
  --model deployment/model_weights/wo_col_loss_wo_temp.pth
```

## 4. 在 ROS 2 中启动相机节点

```bash
source /opt/ros/jazzy/setup.bash
ros2 run image_tools cam2image --ros-args \
  -r image:=/camera/color/image_raw \
  -p device_id:=0 \
  -p width:=640 \
  -p height:=480 \
  -p show_camera:=true
```

## 5. 启动 ROS 2 bridge 节点

```bash
source /opt/ros/jazzy/setup.bash
cd ~/navi_ws/src/lelan_for_futuringrobot
python3 lelan_ros2/lelan_ros2/lelan_policy_node.py --ros-args \
  -p inference_backend:=socket \
  -p server_host:=127.0.0.1 \
  -p server_port:=8765 \
  -p image_topic:=/camera/color/image_raw \
  -p cmd_vel_topic:=/cmd_vel_test \
  -p prompt:=chair \
  -p use_ricoh:=false \
  -p timer_period:=0.1 \
  -p max_linear_vel:=0.3 \
  -p max_angular_vel:=0.5
```

## 6. 验证输出

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic echo /cmd_vel_test
```

## 备注

- `python3 lelan_ros2/lelan_ros2/lelan_policy_node.py` 是当前最短、最直接的可用入口。
- `ros2 run lelan_ros2 lelan_policy_node` 只是可选方式，前提是你已经通过 `colcon build` 安装了 `lelan_ros2`。
- 如果你使用的是 Ricoh Theta 这类全景输入，相应参数改成 `-p use_ricoh:=true`。
- `lelan_col` / `with_col_loss.pth` 这条路线仍然依赖 conda 环境里的 `efficientnet_pytorch`。
- 如果你只是想先验证整条链能不能打通，`wo_col_loss_wo_temp.pth` 是最轻量的一条路线。
- CLIP 模型权重会缓存到 `./.cache/clip` 下，所以第一次启动通常会比后续启动慢一些。
- 对于当前这台机器的环境组合（`torch 2.4.1+cu121` + RTX 5060 Ti），LeLaN 现在会自动回退到 CPU，因为这版 Torch 还不支持 `sm_120`。如果你想显式写出来，可以在推理服务命令里加 `--device cpu`。
- 如果你已经装好了 `usb_cam`，也可以直接替换掉 `cam2image`，只要它发布的是 `/camera/color/image_raw` 即可。

## 7. Gazebo 仿真方案（推荐：TurtleBot 4）

这个仓库本身没有给 LeLaN 单独提供一个专用仿真包。当前 ROS 2 节点实际只需要两样输入输出：

- 一个 `sensor_msgs/msg/Image` 类型的 RGB 图像话题
- 一个 `geometry_msgs/msg/Twist` 类型的速度控制话题

因此，最简单的仿真路线不是自己从头搭机器人，而是直接使用一个已经具备“相机 + `cmd_vel`”能力的官方移动机器人仿真器。对于 Ubuntu 24.04 + ROS 2 Jazzy，推荐直接使用官方的 TurtleBot 4 Gazebo 仿真：

- 用户手册：<https://turtlebot.github.io/turtlebot4-user-manual/software/turtlebot4_simulator.html>
- 仿真总览：<https://turtlebot.github.io/turtlebot4-user-manual/software/simulation.html>
- 源码仓库：<https://github.com/turtlebot/turtlebot4_simulator>

推荐它的原因：

- 它是 ROS 2 Jazzy 官方维护路线之一
- 它已经包含 Gazebo Harmonic 和 ROS-Gazebo bridge 支持
- 它能直接提供移动底盘和 RGB 相机流，正好符合 `lelan_policy_node.py` 的需求
- 你不需要为了接仿真去改 LeLaN 代码

### 7.1 安装 TurtleBot 4 Gazebo 仿真器

先安装 Gazebo Harmonic 和官方 TurtleBot 4 仿真包：

```bash
sudo apt-get install curl
sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install gz-harmonic ros-jazzy-turtlebot4-simulator ros-jazzy-irobot-create-nodes
```

如果你的电脑同时还连着别的 ROS 设备，建议在仿真前加上：

```bash
export ROS_LOCALHOST_ONLY=1
```

### 7.2 启动 Gazebo 仿真

先启动仿真器本体。为了验证 LeLaN，建议先关闭 `nav2` 和 `slam`，避免别的节点和 LeLaN 抢速度控制权。

推荐的第一组启动参数：

```bash
source /opt/ros/jazzy/setup.bash
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py \
  model:=standard \
  world:=warehouse \
  rviz:=true \
  nav2:=false \
  slam:=false
```

说明：

- `world:=warehouse` 比较适合第一轮验证，场景里有目标物也有一定空间
- 如果后面想测试更强一点的绕障碍能力，可以改成 `world:=maze`
- 在验证 LeLaN 输出 `/cmd_vel` 时，不要同时启动 Nav2

### 7.3 找到仿真里的真实话题名

不要先入为主地认为仿真相机一定就是 `/camera/color/image_raw`。你的 LeLaN bridge 默认确实是这个值，但 TurtleBot 4 仿真很可能会带命名空间，或者使用别的相机命名风格。

Gazebo 启动后，先查真实话题：

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic list | rg 'image|camera|cmd_vel'
```

然后确认消息类型：

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic type <SIM_IMAGE_TOPIC>
ros2 topic type <SIM_CMD_TOPIC>
```

你希望看到的是：

- `<SIM_IMAGE_TOPIC>` 的类型是 `sensor_msgs/msg/Image`
- `<SIM_CMD_TOPIC>` 的类型是 `geometry_msgs/msg/Twist`

你也可以顺手检查图像话题是否真的在发：

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic hz <SIM_IMAGE_TOPIC>
```

对于这个仓库，LeLaN 当前只需要 RGB 图像流，不需要深度图。

### 7.4 启动 LeLaN 推理服务

在另一个终端中，从 `lelan` conda 环境启动推理服务。

推荐使用带避障损失的模型组合：

```bash
source /home/zme/anaconda3/etc/profile.d/conda.sh
conda activate lelan
cd ~/navi_ws/src/lelan_for_futuringrobot
python deployment/src/lelan_inference_server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --prompt chair \
  --config train/config/lelan_col.yaml \
  --model deployment/model_weights/with_col_loss.pth \
  --device cpu
```

说明：

- 如果你想测试“看到目标后靠近，同时尽量考虑避障”，优先用 `with_col_loss.pth` + `train/config/lelan_col.yaml`
- 对你现在这台机器来说，`--device cpu` 是最稳妥的显式写法，因为当前 Torch 版本还不完全支持这张新显卡
- 如果你只是想先把整条链打通，也可以退回 `wo_col_loss_wo_temp.pth` + `train/config/lelan.yaml`

### 7.5 让 LeLaN ROS 2 Bridge 接仿真相机

再开一个终端，启动 ROS 2 bridge 节点。

第一轮验证时，不要直接发到底盘控制话题，先发到测试话题：

```bash
source /opt/ros/jazzy/setup.bash
cd ~/navi_ws/src/lelan_for_futuringrobot
python3 lelan_ros2/lelan_ros2/lelan_policy_node.py --ros-args \
  -p inference_backend:=socket \
  -p server_host:=127.0.0.1 \
  -p server_port:=8765 \
  -p image_topic:=<SIM_IMAGE_TOPIC> \
  -p cmd_vel_topic:=/cmd_vel_test \
  -p prompt:=chair \
  -p use_ricoh:=false \
  -p timer_period:=0.3 \
  -p request_timeout:=10.0 \
  -p max_linear_vel:=0.15 \
  -p max_angular_vel:=0.30
```

这里这些参数是有意这么设置的：

- `cmd_vel_topic:=/cmd_vel_test`：先确认 LeLaN 能不能出速度，避免直接驱动仿真机器人
- `timer_period:=0.3`：比默认频率低一点，更适合 CPU 推理
- `request_timeout:=10.0`：第一次联调时不容易因为慢启动而误判超时
- `use_ricoh:=false`：普通前视 RGB 相机都应该关掉它

### 7.6 验证 LeLaN 是否成功输出速度

再开一个终端：

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic echo /cmd_vel_test
```

你可以把下面这些现象视为成功：

- 仿真相机确实在稳定发图
- LeLaN bridge 没有持续报推理错误
- `/cmd_vel_test` 能看到非零的 `linear.x` 和/或 `angular.z`
- 当目标物体出现在画面中时，输出速度会发生变化

到这一步，其实你已经完成了最关键的验证：

- 仿真相机 -> LeLaN -> 速度输出

### 7.7 让仿真机器人真正动起来

只有当 `/cmd_vel_test` 看起来正常之后，再把 LeLaN 接到仿真器的真实控制话题。

重新启动 bridge，把：

```bash
-p cmd_vel_topic:=/cmd_vel_test
```

改成：

```bash
-p cmd_vel_topic:=<SIM_CMD_TOPIC>
```

在这么做之前，先确认没有别的节点也在发速度命令：

- 不要同时开 Nav2
- 如果有键盘遥控，先停掉
- 验证阶段尽量保证系统里只有一个速度命令源

### 7.8 推荐的 Gazebo 联调顺序

建议你按下面这个顺序来，不要一上来就直接自动跑：

1. 在 `warehouse` 世界里生成机器人。
2. 让机器人前方出现一个比较显眼、比较容易识别的物体。
3. 先用简单英文提示词，例如 `chair`、`box`、`plant`、`bottle`。
4. 先确认 `/cmd_vel_test` 有输出。
5. 再切到仿真器真实的 `cmd_vel` 话题。
6. 观察机器人是否会朝目标靠近，以及速度是否被 `max_linear_vel` 和 `max_angular_vel` 正常限制。

### 7.9 常见问题排查

如果 Gazebo 已经起来了，但 LeLaN 没有输出速度：

- 先重新确认相机话题名是不是填对了，这是最常见的问题
- 确认相机话题类型是 `sensor_msgs/msg/Image`，不要误用压缩图像话题
- 确认目标物体真的在 RGB 画面里能看到
- 先用简单英文提示词，不要一开始就用长句或者中文提示词
- 确认推理服务还活着，并且还在监听 `127.0.0.1:8765`
- 如果 CPU 推理偏慢，把 `timer_period` 提高到 `0.5`，把 `request_timeout` 提高到 `15.0`

如果 Gazebo 里机器人不响应 LeLaN：

- 确认你已经把 `cmd_vel_topic` 从 `/cmd_vel_test` 改成仿真器真实控制话题
- 确认没有 Nav2、teleop 或其他节点还在同时发速度

如果你最后不想用 TurtleBot 4，而想换别的仿真器：

- 也完全可以，只要它能提供一个 ROS RGB 图像话题和一个 `Twist` 控制话题
- LeLaN 这一侧并不强依赖 TurtleBot 4
- 这里只是因为 TurtleBot 4 是 ROS 2 Jazzy 下比较省事、官方支持也比较完整的 Gazebo 路线
