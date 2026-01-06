# OCP 构建与求解流程说明

本文档详细说明 `main.py` 中如何一步步构建最优控制问题（OCP）并进行求解。

---

## 一、程序入口：`main()` 函数

### 1.1 初始化机器人模型

```python
# 设置步态序列
robot.set_gait_sequence(gait_type, gait_period)
robot_instance = robot.robot
model = robot.model
data = robot.data
q0 = robot.q0

# 计算初始状态的所有动力学项
pin.computeAllTerms(model, data, q0, np.zeros(model.nv))
```

**作用**：
- 初始化机器人的步态（walk/trot/stand）
- 获取 Pinocchio 模型和数据
- 计算初始姿态的动力学信息

---

## 二、构建 OCP：`make_ocp()` 函数调用链

### 2.1 调用工厂函数 `make_ocp()`

**位置**：`main.py` 第 183-190 行

```python
ocp = make_ocp(
    dynamics=dynamics,              # "whole_body_rnea"
    dyn_args=DYN_ARGS[dynamics],    # {"include_acc": True}
    robot=robot,
    nodes=nodes,                     # 14
    tau_nodes=tau_nodes,            # 3
    warm_start=warm_start,          # True
)
```

**作用**：根据动力学类型创建对应的 OCP 对象

---

### 2.2 工厂函数内部流程

**位置**：`optimization/ocp_factory.py`

```python
def make_ocp(dynamics, dyn_args, **kwargs):
    # 1. 选择 OCP 类
    ocp_classes = {
        "whole_body_rnea": OCPWholeBodyRNEA,
        # ... 其他动力学类型
    }
    
    # 2. 合并参数
    args = dyn_args.copy()
    args.update(kwargs)
    
    # 3. 实例化 OCP 对象
    ocp = ocp_classes[dynamics](**args)
    
    # 4. 构建优化问题
    ocp.setup_problem()
    
    # 5. 设置权重
    ocp.set_weights()
    
    return ocp
```

---

### 2.3 OCP 对象实例化：`OCPWholeBodyRNEA.__init__()`

**位置**：`optimization/ocp_whole_body_rnea.py` 第 10-26 行

```python
def __init__(self, robot, nodes, tau_nodes, warm_start, include_acc=True):
    # 1. 调用父类初始化
    super().__init__(robot, nodes, tau_nodes, warm_start)
    
    # 2. 创建动力学对象
    self.dyn = DynamicsWholeBodyTorque(self.model, self.mass, self.ee_frames)
    
    # 3. 设置标称状态
    self.x_nom = np.concatenate((self.robot.q0, [0] * self.nv))
    
    # 4. 初始化解存储
    self.tau_sol = []
    
    # 5. 设置是否包含加速度
    self.include_acc = include_acc
```

**父类 `OCP.__init__()` 做了什么**：
- 存储机器人模型、数据、步态序列
- 初始化维度信息（nq, nv, nf, nj）
- 创建 CasADi Opti 对象：`self.opti = ca.Opti()`
- 初始化解存储列表

---

### 2.4 构建优化问题：`ocp.setup_problem()`

**位置**：`optimization/ocp.py` 第 42-48 行

```python
def setup_problem(self):
    self.setup_variables()     # 步骤 1
    self.setup_parameters()    # 步骤 2
    self.setup_targets()       # 步骤 3
    self.setup_constraints()   # 步骤 4
    obj = self.setup_objective()  # 步骤 5
    self.opti.minimize(obj)    # 步骤 6
```

#### 步骤 1：`setup_variables()` - 设置决策变量

**位置**：`optimization/ocp_whole_body_rnea.py` 第 59-78 行

```python
def setup_variables(self):
    # 状态维度
    self.nx = self.nq + self.nv  # 位置 + 速度
    self.ndx_opt = self.nv * 2   # 位置增量 + 速度增量
    
    # 输入维度（每个节点可能不同）
    self.nu_opt = [
        self.na_opt + self.nf + self.nj  # 前 tau_nodes 个节点：加速度 + 力 + 力矩
        for _ in range(self.tau_nodes)
    ] + [
        self.na_opt + self.nf  # 后续节点：加速度 + 力（无力矩约束）
        for _ in range(self.nodes - self.tau_nodes)
    ]
    
    # 创建决策变量
    self.DX_opt = []  # 状态增量
    self.U_opt = []   # 控制输入
    for i in range(self.nodes):
        self.DX_opt.append(self.opti.variable(self.ndx_opt))
        self.U_opt.append(self.opti.variable(self.nu_opt[i]))
    self.DX_opt.append(self.opti.variable(self.ndx_opt))  # 终端状态
```

**创建的符号变量**：
- `DX_opt[0..nodes]`：状态增量（nodes+1 个）
- `U_opt[0..nodes-1]`：控制输入（nodes 个）

---

#### 步骤 2：`setup_parameters()` - 设置参数

**位置**：`optimization/ocp.py` 第 56-108 行

```python
def setup_parameters(self):
    # 状态和时间参数
    self.x_init = self.opti.parameter(self.nx)
    self.dt_min = self.opti.parameter(1)
    self.dt_max = self.opti.parameter(1)
    
    # 步态调度参数
    self.contact_schedule = self.opti.parameter(self.n_feet, self.nodes)
    self.swing_schedule = self.opti.parameter(self.n_feet, self.nodes)
    self.n_contacts = self.opti.parameter(1)
    self.swing_period = self.opti.parameter(1)
    self.swing_height = self.opti.parameter(1)
    self.swing_vel_limits = self.opti.parameter(2)
    
    # 优化权重参数
    self.Q_diag = self.opti.parameter(self.ndx_opt)
    self.R_diag = self.opti.parameter(self.nu_opt[0])
    
    # 跟踪目标参数
    self.base_vel_des = self.opti.parameter(6)
    self.arm_vel_des = self.opti.parameter(3)
    self.arm_force_des = self.opti.parameter(3)
    
    # 自适应时间步长
    ratio = self.dt_max / self.dt_min
    gamma = ratio ** (1 / (self.nodes - 1))
    self.dts = [self.dt_min * gamma**i for i in range(self.nodes)]
```

**创建的符号参数**：所有参数都是符号变量，稍后通过 `set_value()` 赋值

---

#### 步骤 3：`setup_targets()` - 设置期望目标

**位置**：`optimization/ocp_whole_body_rnea.py` 第 80-98 行

```python
def setup_targets(self):
    # 期望状态：标称位置 + 期望基座速度 + 零关节速度
    x_des = ca.vertcat(
        self.robot.q0,
        self.base_vel_des,
        [0] * self.nj
    )
    self.dx_des = self.dyn.state_difference()(self.x_init, x_des)
    
    # 期望接触力：重力补偿
    f_gravity = 9.81 * self.mass
    f_front = ca.repmat(ca.vertcat(0, 0, front_scaling * f_gravity / self.n_contacts), 2, 1)
    f_rear = ca.repmat(ca.vertcat(0, 0, rear_scaling * f_gravity / self.n_contacts), 2, 1)
    self.f_des = ca.vertcat(f_front, f_rear)
    
    # 期望输入：零加速度 + 重力补偿力 + 零力矩
    self.u_des = ca.vertcat([0] * self.na_opt, self.f_des, [0] * self.nj)
```

**定义的符号表达式**：
- `dx_des`：期望状态增量
- `u_des`：期望控制输入

---

#### 步骤 4：`setup_constraints()` - 设置约束

**位置**：`optimization/ocp.py` 第 120-257 行

主要约束包括：

1. **初始状态约束**
   ```python
   self.opti.subject_to(self.DX_opt[0] == [0] * self.ndx_opt)
   ```

2. **动力学约束**（在子类中实现）
   ```python
   self.setup_dynamics_constraints(i)  # 每个节点
   ```
   
   **位置**：`optimization/ocp_whole_body_rnea.py` 第 124-157 行
   ```python
   # 积分约束
   self.opti.subject_to(dq_next == dq + v * dt)
   self.opti.subject_to(dv_next == dv + a * dt)
   
   # RNEA 约束
   tau_rnea = self.dyn.rnea_dynamics()(q, v, a, forces)
   self.opti.subject_to(tau_rnea[:6] == [0] * 6)  # 基座无外力矩
   self.opti.subject_to(tau_rnea[6:] == tau_j)    # 关节力矩匹配
   ```

3. **接触约束**
   ```python
   # 摩擦锥
   self.opti.subject_to(in_contact * f_normal >= 0)
   self.opti.subject_to(in_contact * mu**2 * f_normal**2 >= in_contact * f_tangent_square)
   
   # 摆动时零力
   self.opti.subject_to((1 - in_contact) * f_e == [0] * 3)
   
   # 接触时零 xy 速度
   self.opti.subject_to(in_contact * vel_xy == [0] * 2)
   
   # z 方向速度跟踪
   self.opti.subject_to(in_contact * vel_z + (1 - in_contact) * vel_diff == 0)
   ```

4. **关节限制**
   ```python
   self.opti.subject_to(self.opti.bounded(pos_min, q_j, pos_max))
   self.opti.subject_to(self.opti.bounded(vel_min, v_j, vel_max))
   self.opti.subject_to(self.opti.bounded(tau_min, tau_j, tau_max))
   ```

---

#### 步骤 5：`setup_objective()` - 设置目标函数

**位置**：`optimization/ocp_whole_body_rnea.py` 第 100-122 行

```python
def setup_objective(self):
    obj = 0
    Q = ca.diag(self.Q_diag)  # 状态权重矩阵
    R = ca.diag(self.R_diag)  # 输入权重矩阵
    
    # 对每个节点累加代价
    for i in range(self.nodes):
        dx = self.DX_opt[i]
        u = self.U_opt[i]
        
        err_dx = dx - self.dx_des
        err_u = u - self.u_des
        
        obj += err_dx.T @ Q @ err_dx  # 状态误差代价
        obj += err_u.T @ R @ err_u    # 输入误差代价
    
    # 终端状态代价
    dx = self.DX_opt[self.nodes]
    err_dx = dx - self.dx_des
    obj += err_dx.T @ Q @ err_dx
    
    return obj
```

---

#### 步骤 6：最小化目标函数

```python
self.opti.minimize(obj)
```

**至此，优化问题构建完成！**

---

### 2.5 设置权重：`ocp.set_weights()`

**位置**：`optimization/ocp_whole_body_rnea.py` 第 28-57 行

```python
def set_weights(self):
    # 状态权重
    Q_base_pos_diag = [0, 0, 1000, 10000, 10000, 0]  # 重视高度和姿态
    Q_joint_pos_diag = [1000, 500, 500] * 4 + [100] * arm_joints
    Q_vel_diag = [2000, 2000, 1000, 1000, 1000, 2000] + [2]*12 + [10]*arm_joints
    
    # 输入权重
    R_diag = [1e-3]*na + [5e-4]*nf + [1e-4]*12 + [1e-2]*arm_joints
    
    # 赋值
    self.opti.set_value(self.Q_diag, Q_diag)
    self.opti.set_value(self.R_diag, R_diag)
```

---

## 三、设置 OCP 参数

### 3.1 设置时间参数

```python
ocp.set_time_params(dt_min, dt_max)
```

### 3.2 设置摆动参数

```python
ocp.set_swing_params(swing_height, swing_vel_limits)
```

### 3.3 设置跟踪目标

```python
ocp.set_tracking_targets(base_vel_des, arm_vel_des, arm_force_des)
```

---

## 四、MPC 循环：`mpc_loop(ocp)`

### 4.1 初始化

```python
x_init = ocp.x_nom  # 初始状态
t_current = 0       # 当前时间

# 更新参数
ocp.update_params(x_init, t_current)

# 初始化求解器
ocp.init_solver(solver, SOLVER_ARGS[solver])
```

---

### 4.2 MPC 主循环（每次迭代）

```python
for k in range(mpc_loops):
    # 1. 更新参数
    t_current = k * dt_min
    ocp.update_params(x_init, t_current)
    solver_params = ocp.get_solver_params()
    
    # 2. 求解优化问题
    sol_x = solver_function(*solver_params)
    
    # 3. 检查约束违反
    g, lbg, ubg = ocp.g_data(sol_x, stacked_params)
    cv = ocp.constr_viol_norm_inf(g, lbg, ubg)
    
    # 4. 提取解并更新状态
    ocp.retract_stacked_sol(sol_x, retract_all=False)
    dx_sol = ocp.DX_prev[1]
    x_init = ocp.dyn.state_integrate()(x_init, dx_sol)
```

---

### 4.3 参数更新详解：`ocp.update_params()`

**位置**：`optimization/ocp.py` 第 308-322 行

```python
def update_params(self, x_init, t_current):
    # 1. 更新初始状态
    self.update_initial_state(x_init)
    
    # 2. 更新步态序列
    self.update_gait_sequence(t_current)
    
    # 3. 热启动（如果启用）
    if self.warm_start:
        self.warm_start_variables()
```

**步态序列更新**：
```python
def update_gait_sequence(self, t_current):
    dts = [self.opti.value(self.dts[i]) for i in range(self.nodes)]
    contact_schedule, swing_schedule = self.gait_sequence.get_gait_schedule(
        t_current, dts, self.nodes
    )
    self.opti.set_value(self.contact_schedule, contact_schedule)
    self.opti.set_value(self.swing_schedule, swing_schedule)
    # ...
```

---

### 4.4 求解器初始化：`ocp.init_solver()`

**位置**：`optimization/ocp.py` 第 330-393 行

```python
def init_solver(self, solver, solver_args):
    # 存储约束数据
    self.g_data = ca.Function("g_data", [x, p], [g, lbg, ubg])
    
    if solver == "fatrop" or solver == "ipopt":
        # 设置求解器
        self.opti.solver(solver, opts)
        
        # 存储求解器参数
        self.solver_params = [
            self.x_init, self.dt_min, self.dt_max,
            self.contact_schedule, self.swing_schedule,
            # ...
        ]
        
        # 创建求解器函数
        self.solver_function = self.opti.to_function(
            "solver_function",
            self.solver_params,  # 输入
            [self.opti.x],       # 输出
        )
```

---

## 五、完整调用流程图

```
main()
  │
  ├─ robot.set_gait_sequence()
  │
  ├─ make_ocp()  ← 开始构建 OCP
  │   │
  │   ├─ OCPWholeBodyRNEA.__init__()
  │   │   ├─ OCP.__init__()
  │   │   │   └─ self.opti = ca.Opti()
  │   │   └─ self.dyn = DynamicsWholeBodyTorque()
  │   │
  │   ├─ ocp.setup_problem()
  │   │   ├─ setup_variables()      # 创建决策变量
  │   │   ├─ setup_parameters()     # 创建参数
  │   │   ├─ setup_targets()        # 定义期望目标
  │   │   ├─ setup_constraints()    # 添加约束
  │   │   │   └─ setup_dynamics_constraints()
  │   │   ├─ setup_objective()      # 定义目标函数
  │   │   └─ opti.minimize(obj)     # 设置优化目标
  │   │
  │   └─ ocp.set_weights()          # 设置权重矩阵
  │
  ├─ ocp.set_time_params()          # 设置时间参数
  ├─ ocp.set_swing_params()         # 设置摆动参数
  ├─ ocp.set_tracking_targets()     # 设置跟踪目标
  │
  └─ mpc_loop(ocp)  ← 开始 MPC 循环
      │
      ├─ ocp.update_params()        # 初始化参数
      ├─ ocp.init_solver()          # 初始化求解器
      │
      └─ for k in range(mpc_loops):
          ├─ ocp.update_params()    # 更新参数
          ├─ solver_function()      # 求解优化问题
          ├─ ocp.retract_stacked_sol()  # 提取解
          └─ state_integrate()      # 更新状态
```

---

## 六、关键概念总结

### 6.1 符号变量 vs 参数 vs 决策变量

| 类型 | 定义方式 | 何时赋值 | 例子 |
|------|---------|---------|------|
| **参数** | `opti.parameter(n)` | 每次 MPC 迭代前 | `x_init`, `contact_schedule` |
| **决策变量** | `opti.variable(n)` | 求解器优化计算 | `DX_opt`, `U_opt` |
| **符号表达式** | 符号运算 | 求解时自动计算 | `dx_des`, `f_des` |

### 6.2 状态表示

- **完整状态 `x`**：`[q, v]`（位置 + 速度）
- **状态增量 `dx`**：`[dq, dv]`（相对于初始状态的增量）
- **期望状态增量 `dx_des`**：期望达到的状态增量

### 6.3 MPC 滚动时域策略

1. 在当前状态求解未来 N 步的最优轨迹
2. 只执行第一步的控制输入
3. 状态前进一步
4. 重新求解（参数更新，步态相位前进）

---

## 七、总结

整个流程可以概括为：

1. **构建阶段**（一次性）：
   - 创建符号变量和参数
   - 定义约束和目标函数
   - 构建优化问题结构

2. **求解阶段**（每次 MPC 迭代）：
   - 更新参数值（初始状态、步态调度）
   - 调用求解器
   - 提取解并更新状态

这种设计使得优化问题只需构建一次，后续只需更新参数即可快速求解，大大提高了 MPC 的实时性能。
