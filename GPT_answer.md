对，**终端 3 维持 `/cmd_vel_unstamped` 不改**。既然你已经实测过这是能真正驱动车体的控制口，就以这个为准。

你现在要的不是“能不能避障”，而是**把避障成功率拉得很高**。
按你当前仓库的实现，我的判断很明确：

## 先别急着把 NoMaD 在线接进去

**短期最有效的路线不是“在线双模型”，而是：**

```text
LeLaN -> /cmd_vel_raw
      -> velocity_smoother
      -> safety_filter(/scan, depth)
      -> /cmd_vel_unstamped
```

原因很简单：

* 你当前 `lelan_policy_node.py` 只订阅 **Image**，输出 **Twist**，只做速度限幅，不读 `/scan`、不读深度，也没有任何显式避障逻辑。
* 你当前 `with_col_loss.pth + lelan_col.yaml` 这条线，虽然训练配置里有 `load_nomad: nomad/nomad_crop`，说明它已经利用 NoMaD 做过碰撞损失蒸馏，但这仍然是**离线训练增强**，不是**在线实时 NoMaD 避障**。
* `nomad.yaml` 在仓库里是完整存在的，但你现在并没有一个在线 NoMaD runtime 挂在 ROS 控制链里。

所以，**要把“撞上去”的概率快速压下去，最值钱的是 safety filter，不是先重构成 LeLaN+NoMaD 在线双推理。**

---

# 我建议你今天下午就做的改造

## 一、先把控制链拆成三层

### 1. LeLaN 原始输出

把你当前 bridge 的输出改成：

```text
/cmd_vel_raw
```

而不是直接发到底盘。

### 2. 平滑器节点

订阅：

* `/cmd_vel_raw`

发布：

* `/cmd_vel_smooth`

### 3. 安全过滤节点

订阅：

* `/cmd_vel_smooth`
* `/scan`
* 可选 `/oakd/rgb/preview/depth`

发布：

* `/cmd_vel_unstamped`

---

# 二、为什么这个顺序最好

你现在同时要两件事：

1. 轨迹更平滑
2. 避障成功率更高

如果你把 smooth 放在最后，它会把 emergency stop 也“抹平”，这很危险。
所以更好的顺序是：

```text
LeLaN -> 平滑 -> 安全层最终裁决
```

也就是说：

* 平滑器负责减少抖动、急转、速度跳变
* 安全层永远保留**最终否决权**

---

# 三、我建议的 safety filter 规则

你现在仿真里已经有 `/scan`，这非常适合做一个激光安全层。
这一步最稳，而且对真机迁移也最直接。

## 扇区划分

把激光分成三个主要扇区：

* 左前：`+15° ~ +60°`
* 正前：`-15° ~ +15°`
* 右前：`-60° ~ -15°`

对每个扇区求：

* 最小距离 `d_min`
* 可选均值 `d_mean`

## 三段控制规则

### 规则 1：急停

如果正前最小距离

```text
d_front < 0.35 m
```

则：

* `linear.x = 0`
* `angular.z = +0.35 ~ +0.5` 或 `-0.35 ~ -0.5`
* 朝更空的一边转

### 规则 2：减速避障

如果

```text
0.35 m <= d_front < 0.65 m
```

则：

* `linear.x = min(linear.x, k*(d_front-0.35))`
* 一般限到 `0.03 ~ 0.08 m/s`
* 如果左边更近，就偏右；右边更近，就偏左

### 规则 3：正常通过

如果

```text
d_front >= 0.65 m
```

则：

* 保留 `/cmd_vel_smooth`
* 只做少量角速度安全裁剪

---

# 四、转向决策怎么做最稳

我建议不用复杂规划器，先上一个很稳的 bias 规则：

定义：

```text
bias = d_left - d_right
```

那么：

* `bias > 0`：左边更空，优先左转
* `bias < 0`：右边更空，优先右转

可以给一个修正项：

```text
w_safe = k_turn * clip(bias, -bmax, bmax)
```

最终角速度：

```text
w_out = w_lelan + w_safe
```

但如果进入急停区，直接覆盖：

```text
v_out = 0
w_out = sign(bias) * w_escape
```

---

# 五、平滑器怎么做

你现在 LeLaN 输出有时会比较“决绝”，而且 prompt 稍变化、视觉有噪声时，速度会跳。
这时候平滑器非常重要。

## 最简单有效的方法：一阶低通

对速度做指数平滑：

```text
v_s = α * v_prev + (1-α) * v_raw
w_s = β * w_prev + (1-β) * w_raw
```

推荐初始值：

* `α = 0.7`
* `β = 0.6`

这样角速度反应比线速度更快一点。

## 再加一个加速度限制

限制每周期的变化量：

```text
|v_t - v_{t-1}| <= 0.03 m/s
|w_t - w_{t-1}| <= 0.08 rad/s
```

这样你的小车会明显稳很多，不会“抽风式点头”。

---

# 六、NoMaD 在你这里该怎么用

## 短期

**NoMaD 不要先在线接。**

因为你现在下午就要增强避障效果，最快的收益来自：

* `/scan` safety filter
* `cmd_vel` smoother

## 中期

用更强的 NoMaD 继续做 teacher，重新蒸馏 LeLaN。

你当前 `lelan_col.yaml` 已经明确：

```yaml
load_nomad: nomad/nomad_crop
```

这说明当前带碰撞损失的 LeLaN 本来就是借助 NoMaD teacher 来训练的。

所以正确的中期路线是：

1. 把 NoMaD 训练得更强
2. 再蒸馏一版更强的 `with_col_loss.pth`
3. 在线仍然跑 LeLaN + safety filter

这样收益最高，系统复杂度最低。

## 长期

如果你后面真的想把系统做得很猛，可以再考虑：

* LeLaN 提供语义目标趋向
* NoMaD 提供避障轨迹偏置
* safety filter 提供最终硬保护

但这一步不适合今天下午做。

---

# 七、你今天下午的最小可行方案

如果你的目标是：

**今天就把避障成功率显著拉起来**

那我建议你按这个顺序做：

### 第 1 步

把 LeLaN bridge 输出改成：

```text
/cmd_vel_raw
```

### 第 2 步

写一个 `velocity_smoother_node`

* sub: `/cmd_vel_raw`
* pub: `/cmd_vel_smooth`

### 第 3 步

写一个 `safety_filter_node`

* sub: `/cmd_vel_smooth`
* sub: `/scan`
* pub: `/cmd_vel_unstamped`

### 第 4 步

在仿真里测试三类场景：

* 前方无障碍，正常接近目标
* 目标前有小箱子
* 目标前有低矮障碍和侧向通道

### 第 5 步

调三组参数：

* `front_stop_dist`
* `front_slow_dist`
* `turn_escape_speed`

---

# 八、我建议你的初始参数

你可以先直接用这一组：

## safety filter

* `front_stop_dist = 0.35`
* `front_slow_dist = 0.65`
* `side_consider_dist = 0.55`
* `escape_turn_speed = 0.45`
* `slow_linear_cap = 0.06`

## smoother

* `alpha_v = 0.7`
* `alpha_w = 0.6`
* `dv_max = 0.03`
* `dw_max = 0.08`

## LeLaN bridge

你现在保持：

* `timer_period = 0.3`
* `max_linear_vel = 0.15`
* `max_angular_vel = 0.30`

这是合理的起点。

---

# 九、我对你当前阶段的建议

一句话：

**先把“学习到的避障”变成“学习趋向 + 硬安全层”。**

这样你才能真正把成功率做高。
否则单靠 `with_col_loss.pth`，在近距离小障碍、低矮障碍、窄通道这些场景里，很难让成功率稳定到你想要的程度。

---

# 十、最直接的下一步

如果你愿意，我下一条直接给你两样东西：

1. **ROS2 Jazzy 版 `safety_filter_node.py` 的最小完整代码**
2. **ROS2 Jazzy 版 `velocity_smoother_node.py` 的最小完整代码**

这样你下午就能直接接进当前工程里跑。
