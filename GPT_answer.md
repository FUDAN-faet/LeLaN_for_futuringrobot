很好，参数已经足够了。
你这个 ROS2 节点的参数名就是：

* `image_topic`
* `cmd_vel_topic`
* `prompt`
* `config_path`
* `model_path`
* `use_ricoh`
* `timer_period`
* `max_linear_vel`
* `max_angular_vel`

所以你现在已经可以**直接试运行**了。

---

# 先跑最简单版本

第一次本机测试，先用：

* 配置：`lelan.yaml`
* 权重：`wo_col_loss_wo_temp.pth`

这样最轻、最容易先跑通。

你的文件路径就是：

```bash id="ps6s3c"
/home/zme/navi_ws/src/LeLaN/train/config/lelan.yaml
/home/zme/navi_ws/src/LeLaN/deployment/model_weights/wo_col_loss_wo_temp.pth
```

---

# 四个终端这样开

## 终端 1：本机摄像头发布图像

让话题名直接匹配节点默认值 `/camera/color/image_raw`：

```bash id="c1pr2q"
source /opt/ros/jazzy/setup.bash
source ~/navi_ws/install/setup.bash

ros2 run image_tools cam2image --ros-args -r image:=/camera/color/image_raw
```

---

## 终端 2：确认相机图像正常

```bash id="e6kxyy"
source /opt/ros/jazzy/setup.bash
source ~/navi_ws/install/setup.bash

ros2 run image_tools showimage --ros-args -r image:=/camera/color/image_raw
```

如果这一步看不到图，先不要跑 LeLaN。

---

## 终端 3：启动 LeLaN 节点

直接用这条：

```bash id="x30ng1"
source /opt/ros/jazzy/setup.bash
source ~/navi_ws/install/setup.bash

ros2 run lelan_ros2 lelan_policy_node --ros-args \
  -p image_topic:="/camera/color/image_raw" \
  -p cmd_vel_topic:="/cmd_vel_test" \
  -p prompt:="chair" \
  -p config_path:="/home/zme/navi_ws/src/LeLaN/train/config/lelan.yaml" \
  -p model_path:="/home/zme/navi_ws/src/LeLaN/deployment/model_weights/wo_col_loss_wo_temp.pth" \
  -p use_ricoh:=false \
  -p timer_period:=0.1 \
  -p max_linear_vel:=0.3 \
  -p max_angular_vel:=0.5
```

这里我把输出话题改成了：

```bash id="7wgubk"
/cmd_vel_test
```

这样不会误发到真实机器人控制链。

---

## 终端 4：看速度输出

```bash id="dxsk9j"
source /opt/ros/jazzy/setup.bash
source ~/navi_ws/install/setup.bash

ros2 topic echo /cmd_vel_test
```

---

# 你现在该怎么测

拿一个简单目标到镜头前，比如：

* 椅子：`chair`
* 杯子：`cup`
* 瓶子：`bottle`

先用 `chair` 试。
观察 `/cmd_vel_test`：

### 你应该重点看

* 目标不在画面里时，输出不要一直乱跳
* 目标进入画面中央时，`linear.x` 应该有明显变化
* 目标偏左/偏右时，`angular.z` 应该跟着变化

如果这几条成立，说明：

**相机输入 → LeLaN 推理 → `Twist` 输出**

这条近程链已经打通了。

---

# 如果节点没有输出，立刻查这三件事

## 1）节点有没有真的启动

```bash id="jlwm34"
ros2 node list
```

你应该看到：

```bash id="boqvxp"
/lelan_policy_node
```

## 2）节点订阅和发布了什么

```bash id="pjlwmv"
ros2 node info /lelan_policy_node
```

重点看：

* Subscriptions 里有没有 `/camera/color/image_raw`
* Publishers 里有没有 `/cmd_vel_test`

## 3）话题有没有数据

```bash id="qfz989"
ros2 topic hz /camera/color/image_raw
ros2 topic info /cmd_vel_test
```

---

# 如果你想用 launch，也可以试一下源码里的 launch 文件

你 grep 已经看到了本地源码里有：

```bash id="ao4q1h"
~/navi_ws/src/LeLaN/lelan_ros2/launch/lelan_policy.launch.py
```

虽然它没被安装到 `share/lelan_ros2`，但你仍然可以直接用源码路径试试：

```bash id="03kw05"
python3 ~/navi_ws/src/LeLaN/lelan_ros2/launch/lelan_policy.launch.py
```

不过我还是建议你**先用 `ros2 run`**，因为你现在最重要的是排除参数和模型路径问题。

---

# 跑通后第二步怎么做

等最简单版本跑通，再测带碰撞规避训练的版本：

```bash id="9m7xva"
ros2 run lelan_ros2 lelan_policy_node --ros-args \
  -p image_topic:="/camera/color/image_raw" \
  -p cmd_vel_topic:="/cmd_vel_test" \
  -p prompt:="chair" \
  -p config_path:="/home/zme/navi_ws/src/LeLaN/train/config/lelan_col.yaml" \
  -p model_path:="/home/zme/navi_ws/src/LeLaN/deployment/model_weights/with_col_loss.pth" \
  -p use_ricoh:=false \
  -p timer_period:=0.1 \
  -p max_linear_vel:=0.3 \
  -p max_angular_vel:=0.5
```

这样你可以比较：

* `wo_col_loss_wo_temp.pth`：最简单、最轻
* `with_col_loss.pth`：带碰撞规避蒸馏，理论上更稳一些

---

# 我建议你现在就执行的顺序

按这个顺序来：

1. 终端 1 跑 `cam2image`
2. 终端 2 跑 `showimage`
3. 终端 3 跑我给你的 `lelan_policy_node` 命令
4. 终端 4 `echo /cmd_vel_test`

然后把**终端 3 的启动日志**和**终端 4 的几行输出**发给我。
如果有报错，我可以直接帮你定位到下一步。
