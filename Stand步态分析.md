# Stand 步态模式详解

## 一、Stand 模式的特点

### 最简单的情况
- ✅ **所有4只脚始终接触地面**
- ✅ **没有摆动相位**（swing phase = 0）
- ✅ **接触调度固定不变**
- ✅ **只需要平衡和速度控制**

### 与 Walk/Trot 的对比

| 特性 | Stand | Walk | Trot |
|------|-------|------|------|
| 接触脚数 | 4 | 3 | 2 |
| 步态切换 | ❌ 无 | ✅ 有 | ✅ 有 |
| 摆动约束 | ❌ 无 | ✅ 有 | ✅ 有 |
| 复杂度 | 低 | 中 | 高 |

---

## 二、Stand 模式下的接触调度

### 接触调度矩阵

```python
# 4只脚 × 14个节点（预测步数）
contact_schedule = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # FL (前左)
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # FR (前右)
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # RL (后左)
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # RR (后右)
]

# 摆动调度（全为0，因为没有摆动（离地））
swing_schedule = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # FL
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # FR
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # RL
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # RR
]
```

---

## 三、Stand 模式下的约束

### 1. 接触约束（所有脚都激活）

```python
for idx in range(4):  # 4只脚
    f_e = forces[idx*3 : (idx+1)*3]  # 每只脚的力 [fx, fy, fz]
    in_contact = 1  # Stand模式下始终为1
    
    # ✅ 摩擦锥约束
    f_normal = f_e[2]  # fz
    f_tangent_square = f_e[0]**2 + f_e[1]**2  # fx² + fy²
    
    # 法向力必须 >= 0（只能推地面）
    f_normal >= 0
    
    # 切向力 <= μ * 法向力（摩擦系数 μ=0.9）
    sqrt(fx² + fy²) <= 0.9 * fz
    
    # ✅ 零速度约束（脚不能滑动）
    vel_xy = [vx, vy] == [0, 0]  # xy方向速度为0
    vel_z = 0                     # z方向速度为0
```

### 2. 动力学约束

```python
# RNEA 动力学方程
tau_rnea = rnea(q, v, a, forces)

# 基座无外力矩（浮动基座）
tau_rnea[0:6] == 0

# 关节力矩匹配
tau_rnea[6:] == tau_j
```

### 3. 关节限制

```python
# 位置限制
-π/2 <= q_joint <= π/2

# 速度限制
-10 rad/s <= v_joint <= 10 rad/s

# 力矩限制（前3个节点）
-40 Nm <= tau_joint <= 40 Nm
```

---

## 四、Stand 模式下的优化目标

### 期望状态

```python
# 你设置的跟踪目标
base_vel_des = [0.1, 0, 0, 0, 0, 0]  # 向前 0.1 m/s

# 期望状态
x_des = [
    q0,              # 标称站立姿态
    [0.1, 0, 0],     # 基座线速度：向前 0.1 m/s
    [0, 0, 0],       # 基座角速度：不旋转
    [0] * 16         # 关节速度：全为0
]
```

### 期望接触力（重力补偿）

```python
# 总重力
f_gravity = 9.81 * mass  # 假设 mass = 50 kg → 490.5 N

# 前后重量分配（假设前后比例 0.5:0.5）
f_per_foot = f_gravity / 4  # 每只脚 122.6 N

# 期望力（每只脚）
f_des = [
    [0, 0, 122.6],  # FL: 只有z方向力
    [0, 0, 122.6],  # FR
    [0, 0, 122.6],  # RL
    [0, 0, 122.6],  # RR
]
```

### 目标函数

```python
# 最小化
obj = Σ (状态误差² + 控制输入误差²)

# 展开
obj = Σ [
    # 状态误差
    (base_pos - base_pos_des)² * Q_pos +
    (base_vel - [0.1, 0, 0, 0, 0, 0])² * Q_vel +
    (joint_pos - q0)² * Q_joint_pos +
    (joint_vel - 0)² * Q_joint_vel +
    
    # 控制输入误差
    (a - 0)² * R_acc +
    (f - f_des)² * R_force +
    (tau - 0)² * R_torque
]
```

---

## 五、Stand 模式的物理意义

### 优化器在做什么？

想象机器人站在地面上，你要求它向前移动 0.1 m/s：

1. **优化器需要找到**：
   - 每只脚施加多大的力？
   - 每个关节需要多大的力矩？
   - 基座如何加速？

2. **约束条件**：
   - 4只脚都不能离地（接触约束）
   - 4只脚都不能滑动（零速度约束）
   - 每只脚的力必须在摩擦锥内
   - 动力学方程必须满足
   - 关节不能超限

3. **优化目标**：
   - 尽量达到 0.1 m/s 的速度
   - 尽量保持站立姿态
   - 尽量减小控制输入（节能）

---

## 六、调试 Stand 模式的技巧

### 1. 查看接触调度

```python
# 在 update_gait_sequence 后添加
contact_schedule = ocp.opti.value(ocp.contact_schedule)
print("Contact schedule:")
print(contact_schedule)
# 应该全是 1
```

### 2. 查看期望力

```python
# 在 setup_targets 后添加
f_des = ocp.opti.value(ocp.f_des)
print("Desired forces:")
print(f_des.reshape(4, 3))  # 4只脚 × 3维力
# 应该看到每只脚约 [0, 0, 120] N
```

### 3. 查看求解结果

```python
# 在 MPC 循环中
forces_sol = ocp.forces_sol[0]  # 第一步的解
print("Solved forces:")
print(forces_sol.reshape(4, 3))
# 看看是否接近期望力
```

### 4. 可视化基座轨迹

```python
# 收集基座位置
base_positions = []
for q in ocp.q_sol:
    base_pos = q[:3]  # [x, y, z]
    base_positions.append(base_pos)

import matplotlib.pyplot as plt
base_positions = np.array(base_positions)
plt.plot(base_positions[:, 0], label='x')  # 应该逐渐增加
plt.plot(base_positions[:, 1], label='y')  # 应该接近0
plt.plot(base_positions[:, 2], label='z')  # 应该保持恒定
plt.legend()
plt.show()
```

---

## 七、常见问题

### Q1: 为什么约束违反（CV）很大？

```
CV (inf norm):  206.01
```

这表示某些约束没有完全满足。可能原因：
- 求解器迭代次数不够（max_iter=10 太少）
- 初始猜测不好
- 约束之间冲突

**解决方法**：
```python
# 增加迭代次数
"fatrop.max_iter": 50,  # 从 10 改为 50

# 放松容差
"fatrop.tol": 1e-2,  # 从 1e-3 改为 1e-2
```

### Q2: Stand 模式下机器人应该怎么运动？

- **理想情况**：基座平滑加速到 0.1 m/s，然后匀速前进
- **实际情况**：可能会有小幅振荡，因为优化不完美

### Q3: 为什么需要 MPC？站立不是很简单吗？

即使是站立，也需要：
- 实时调整接触力以保持平衡
- 补偿模型误差和外部扰动
- 平滑地加速/减速

---

## 八、下一步学习建议

### 掌握 Stand 后，可以尝试：

1. **修改速度**
   ```python
   base_vel_des = [0, 0.1, 0, 0, 0, 0]  # 向左移动
   base_vel_des = [0, 0, 0, 0, 0, 0.1]  # 原地旋转
   ```

2. **查看力分布**
   - 前进时，后脚的力会增大（推进）
   - 旋转时，左右脚的力会不对称

3. **理解权重的作用**
   ```python
   Q_vel_diag[0] = 20000  # 增大x方向速度权重
   # 机器人会更积极地跟踪x方向速度
   ```

4. **然后再看 Walk 模式**
   - 理解步态切换
   - 理解摆动约束

---

## 九、关键代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| 步态定义 | `utils/gait_sequence.py` | - |
| 接触约束 | `optimization/ocp.py` | 145-192 |
| 期望目标 | `optimization/ocp_whole_body_rnea.py` | 80-98 |
| 动力学约束 | `optimization/ocp_whole_body_rnea.py` | 124-157 |

---

**总结**：Stand 模式是最简单的情况，专注理解：
1. 接触力如何分配
2. 动力学方程如何满足
3. 优化器如何平衡多个目标

掌握这些后，Walk/Trot 只是增加了时间维度的步态切换！
