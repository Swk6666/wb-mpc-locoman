import time
import numpy as np
import pinocchio as pin
import casadi as ca
import matplotlib.pyplot as plt

from args import *
from utils.robot import *
from utils.visualization import visualize_forces
from utils.trajectory import CircleTrajectory, HelixTrajectory
from optimization import make_ocp

# Robot params
robot = B2_Z1(reference_pose="standing_with_arm_up", arm_joints=6)
dynamics ="whole_body_rnea"  # see args.py for options

# Tracking targets
base_vel_des = np.array([0.2, 0.05, 0, 0, 0, 0])  # linear + angular velocity (静止站立)
arm_vel_des = np.array([0, 0, 0])             # arm EE velocity (relative to the base)
arm_force_des = np.array([0, 0, 0])           # arm EE force (global)

# 轨迹配置：yz平面半径0.2m的圆 + x方向0.1m/s的线性运动
# 带阻抗控制（PD）来补偿位置误差
arm_trajectory = CircleTrajectory(
    radius=0.20,                      # 圆的半径 0.2m
    period=3.0,                       # 画一圈的周期 3秒
    plane="yz",                       # 在yz平面画圆
    linear_vel=np.array([0, 0, 0]),   # 线性速度
    kp=np.array([20.0, 20.0, 20.0]),     # 位置增益（阻抗控制）
    kd=np.array([0.1, 0.1, 0.1]),     # 速度增益（阻抗控制）
)
use_impedance = True  # 是否启用阻抗控制

# OCP params
nodes = 14      # OCP nodes
tau_nodes = 3   # add torque limits for this many nodes
dt_min = 0.015  # initial time step
dt_max = 0.08   # final time step

# Gait params
gait_type = "walk"              # "trot", "walk" or "stand"
gait_period = 0.8               # seconds
swing_height = 0.07             # meters
swing_vel_limits = [0.1, -0.2]  # meters/second

# Solver
solver = "fatrop"  # see args.py for options
warm_start = True
compile_solver = False
load_compiled_solver = None  # None or <filename> in "codegen/lib/"

# MPC
mpc_loops = 400  # 6秒 / 0.015秒 = 400步

# Debug
plot = False  # plot joint positions, velocities, torques




def mpc_loop(ocp):
    """
    MPC主循环函数
    
    功能：执行模型预测控制的主循环，每次迭代求解一个优化问题并更新机器人状态
    
    参数：
        ocp: 最优控制问题对象（OCP）
    
    返回：
        ocp: 包含求解历史的OCP对象
    """
    # 用于记录性能统计的列表
    solve_times = []      # 记录每次求解耗时
    constr_viol = []      # 记录每次约束违反程度
    
    # ========== 初始化阶段 ==========
    # 设置初始状态为机器人的标称状态（站立姿态）
    x_init = ocp.x_nom
    # 当前仿真时间初始化为0
    t_current = 0
    # 更新OCP的参数（初始状态、接触序列等）
    ocp.update_params(x_init, t_current)
    
    # 初始化求解器（Fatrop/IPOPT/OSQP）
    ocp.init_solver(solver, SOLVER_ARGS[solver])
    # 如果需要，编译求解器为C代码（加速）
    if compile_solver:
        ocp.compile_solver()
    
    # ========== 根据求解器类型选择不同的求解路径 ==========
    if solver == "fatrop" or solver == "ipopt":
        # ===== Fatrop/IPOPT求解器路径 =====
        # 这两个求解器支持代码生成和外部函数调用
        
        # 获取求解器函数
        if load_compiled_solver:
            # 如果指定了编译好的求解器，从共享库加载
            # 这样可以避免Python开销，速度更快（约2倍加速）
            solver_function = ca.external("solver_function", "codegen/lib/" + load_compiled_solver)
        else:
            # 否则使用CasADi生成的求解器函数
            solver_function = ocp.solver_function
        
        # ===== MPC主循环 =====
        for k in range(mpc_loops):
            # --- 1. 更新优化问题参数 ---
            # 计算当前仿真时间
            t_current = k * dt_min
            
            # --- 计算轨迹的期望速度（带阻抗控制）---
            if use_impedance:
                # 获取当前末端位置和速度
                q_current = np.array(x_init[:ocp.nq]).flatten()
                v_current = np.array(x_init[ocp.nq:]).flatten()
                pin.framesForwardKinematics(ocp.model, ocp.data, q_current)
                pin.computeFrameJacobian(ocp.model, ocp.data, q_current, robot.arm_ee_frame, pin.LOCAL_WORLD_ALIGNED)
                
                # 当前末端位置（相对于基座）
                ee_pos = ocp.data.oMf[robot.arm_ee_frame].translation.copy()
                base_pos = ocp.data.oMf[ocp.model.getFrameId("base_link")].translation.copy()
                base_rot = ocp.data.oMf[ocp.model.getFrameId("base_link")].rotation.copy()
                ee_pos_rel = base_rot.T @ (ee_pos - base_pos)
                
                # 当前末端速度（相对于基座）
                J = pin.getFrameJacobian(ocp.model, ocp.data, robot.arm_ee_frame, pin.LOCAL_WORLD_ALIGNED)
                ee_vel = J @ v_current
                ee_vel_lin = ee_vel[:3]
                ee_vel_rel = base_rot.T @ ee_vel_lin
                
                # 带阻抗控制的速度命令
                arm_vel_des_current = arm_trajectory.get_velocity_with_impedance(
                    t_current, ee_pos_rel, ee_vel_rel
                )
            else:
                # 纯速度跟踪
                arm_vel_des_current = arm_trajectory.get_velocity(t_current)
            
            ocp.set_tracking_targets(base_vel_des, arm_vel_des_current, arm_force_des)
            
            # 更新OCP参数：初始状态、接触序列、期望轨迹等
            # 这会根据当前时间更新步态相位、接触脚等信息
            ocp.update_params(x_init, t_current)
            # 获取所有求解器参数（打包成列表）
            solver_params = ocp.get_solver_params()
            
            # --- 2. 求解优化问题 ---
            # 记录求解开始时间
            start_time = time.time()
            # 调用求解器函数，输入当前参数，输出最优解
            # sol_x包含所有节点的状态和控制输入
            sol_x = solver_function(*solver_params)
            # 记录求解结束时间
            end_time = time.time()
            # 计算求解耗时
            sol_time = end_time - start_time
            solve_times.append(sol_time)
            print("Solve time (ms): ", sol_time * 1000)
            
            # --- 3. 检查约束违反程度 ---
            # 获取当前的参数值
            stacked_params = ocp.opti.value(ocp.opti.p)
            # 计算约束函数值和上下界
            # g: 约束函数值, lbg: 下界, ubg: 上界
            g, lbg, ubg = ocp.g_data(sol_x, stacked_params)
            # 计算约束违反的无穷范数（最大违反量）
            # 理想情况下应该接近0，表示所有约束都满足
            cv = ocp.constr_viol_norm_inf(g, lbg, ubg)
            constr_viol.append(cv)
            print("CV (inf norm): ", cv)
            
            # --- 4. 提取解并更新状态 ---
            # 从堆叠的解向量中提取状态和控制输入
            # retract_all=False 表示只提取第一步的解（MPC的滚动时域策略）
            ocp.retract_stacked_sol(sol_x, retract_all=False)
            # 获取下一步的状态增量（第二个节点的解）
            # 注意：DX_prev[0]是当前状态，DX_prev[1]是下一步状态
            dx_sol = ocp.DX_prev[1]
            # 使用动力学积分更新状态
            # 这相当于执行一步控制，让机器人从x_init运动到新状态
            x_init = ocp.dyn.state_integrate()(x_init, dx_sol)
    
    else:
        # ===== OSQP或其他求解器路径 =====
        # 这些求解器使用CasADi的Opti接口直接求解
        
        for k in range(mpc_loops):
            # --- 1. 更新优化问题参数 ---
            t_current = k * dt_min
            
            # --- 计算轨迹的期望速度（带阻抗控制）---
            if use_impedance:
                # 获取当前末端位置和速度
                q_current = np.array(x_init[:ocp.nq]).flatten()
                v_current = np.array(x_init[ocp.nq:]).flatten()
                pin.framesForwardKinematics(ocp.model, ocp.data, q_current)
                pin.computeFrameJacobian(ocp.model, ocp.data, q_current, robot.arm_ee_frame, pin.LOCAL_WORLD_ALIGNED)
                
                # 当前末端位置（相对于基座）
                ee_pos = ocp.data.oMf[robot.arm_ee_frame].translation.copy()
                base_pos = ocp.data.oMf[ocp.model.getFrameId("base_link")].translation.copy()
                base_rot = ocp.data.oMf[ocp.model.getFrameId("base_link")].rotation.copy()
                ee_pos_rel = base_rot.T @ (ee_pos - base_pos)
                
                # 当前末端速度（相对于基座）
                J = pin.getFrameJacobian(ocp.model, ocp.data, robot.arm_ee_frame, pin.LOCAL_WORLD_ALIGNED)
                ee_vel = J @ v_current
                ee_vel_lin = ee_vel[:3]
                ee_vel_rel = base_rot.T @ ee_vel_lin
                
                # 带阻抗控制的速度命令
                arm_vel_des_current = arm_trajectory.get_velocity_with_impedance(
                    t_current, ee_pos_rel, ee_vel_rel
                )
            else:
                # 纯速度跟踪
                arm_vel_des_current = arm_trajectory.get_velocity(t_current)
            
            ocp.set_tracking_targets(base_vel_des, arm_vel_des_current, arm_force_des)
            
            ocp.update_params(x_init, t_current)
            
            # --- 2. 求解优化问题 ---
            # 直接调用ocp.solve()，内部会处理求解器调用
            ocp.solve(retract_all=False)
            # 记录求解时间和约束违反
            solve_times.append(ocp.solve_time)
            constr_viol.append(ocp.constr_viol)
            
            # --- 3. 更新状态 ---
            dx_sol = ocp.DX_prev[1]
            x_init = ocp.dyn.state_integrate()(x_init, dx_sol)
    
    # ========== 统计信息 ==========
    # 计算预测时域的总长度（所有时间步之和）
    T = sum([ocp.opti.value(dt) for dt in ocp.dts])
    
    # 打印性能统计
    total_solve_time = sum(solve_times)
    total_sim_time = mpc_loops * dt_min
    print("************** STATS **************")
    print("Avg solve time (ms): ", np.average(solve_times) * 1000)  # 平均求解时间
    print("Std solve time (ms): ", np.std(solve_times) * 1000)      # 求解时间标准差
    print("Total solve time (s): ", total_solve_time)               # 总求解时间
    print("Total sim time (s): ", total_sim_time)                   # 总仿真时间
    print("Realtime factor: ", total_sim_time / total_solve_time)   # 实时性因子 (>1 表示比实时快)
    print("Avg CV (inf norm): ", np.average(constr_viol))           # 平均约束违反
    print("Horizon length (s): ", T)                                # 预测时域长度
    
    return ocp


def main():
    # Initialize robot
    robot.set_gait_sequence(gait_type, gait_period)
    robot_instance = robot.robot
    model = robot.model
    data = robot.data
    q0 = robot.q0
    print("Robot model: ", model)

    pin.computeAllTerms(model, data, q0, np.zeros(model.nv))

    # Setup OCP
    ocp = make_ocp(
        dynamics=dynamics,
        dyn_args=DYN_ARGS[dynamics],
        robot=robot,
        nodes=nodes,
        tau_nodes=tau_nodes,
        warm_start=warm_start,
    )
    ocp.set_time_params(dt_min, dt_max)
    ocp.set_swing_params(swing_height, swing_vel_limits)
    ocp.set_tracking_targets(base_vel_des, arm_vel_des, arm_force_des)

    # Run MPC
    ocp = mpc_loop(ocp)

    if plot:
        # Plot joint positions, velocities, torques
        if hasattr(ocp, "tau_sol"):
            tau_j_sol = ocp.tau_sol
        else:
            # Compute from RNEA
            tau_j_sol = []
            for k in range(len(ocp.q_sol)):
                q = ocp.q_sol[k].flatten()
                v = ocp.v_sol[k].flatten()
                a = ocp.a_sol[k].flatten()
                forces = ocp.forces_sol[k].flatten()

                tau_rnea = ocp.dyn.rnea_dynamics()(q, v, a, forces)
                tau_rnea = np.array(tau_rnea).flatten()
                tau_j = tau_rnea[6:]
                tau_j_sol.append(tau_j)

        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        labels = ["FL hip", "FL thigh", "FL calf", "FR hip", "FR thigh", "FR calf",
                  "RL hip", "RL thigh", "RL calf", "RR hip", "RR thigh", "RR calf",
                  "Arm 1", "Arm 2", "Arm 3", "Arm 4"]

        axs[0].set_title("Joint positions (q)")
        for j in range(robot.nj):
            # Ignore base (quaternion)
            axs[0].plot([q[7 + j] for q in ocp.q_sol], label=labels[j])
        axs[0].set_xlabel("Time step")
        axs[0].set_ylabel("Position (rad)")

        axs[1].set_title("Joint velocities (v)")
        for j in range(robot.nj):
            # Ignore base
            axs[1].plot([v[6 + j] for v in ocp.v_sol], label=labels[j])
        axs[1].set_xlabel("Time step")
        axs[1].set_ylabel("Velocity (rad/s)")

        axs[2].set_title("Joint torques (tau)")
        for j in range(robot.nj):
            axs[2].plot([tau[j] for tau in tau_j_sol], label=labels[j])
        axs[2].set_xlabel("Time step")
        axs[2].set_ylabel("Torque (Nm)")

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1, 0.5))

        plt.tight_layout(rect=[0, 0, 0.88, 1])  # adjust for legend
        plt.show()

    # 提取机械臂末端轨迹（惯性系/全局坐标系）
    arm_ee_positions = []
    base_positions = []
    for q in ocp.q_sol:
        pin.framesForwardKinematics(model, data, q.flatten())
        # 全局位置
        ee_pos = data.oMf[robot.arm_ee_frame].translation.copy()
        base_pos = data.oMf[model.getFrameId("base_link")].translation.copy()
        arm_ee_positions.append(ee_pos)
        base_positions.append(base_pos)
    arm_ee_positions = np.array(arm_ee_positions)
    base_positions = np.array(base_positions)
    
    print("末端轨迹形状:", arm_ee_positions.shape)
    print(f"起始点（惯性系）: X={arm_ee_positions[0, 0]:.4f}, Y={arm_ee_positions[0, 1]:.4f}, Z={arm_ee_positions[0, 2]:.4f}")
    print(f"终止点（惯性系）: X={arm_ee_positions[-1, 0]:.4f}, Y={arm_ee_positions[-1, 1]:.4f}, Z={arm_ee_positions[-1, 2]:.4f}")
    
    # 生成期望轨迹（惯性系）
    # 使用初始末端位置作为轨迹起点
    arm_trajectory.reset(arm_ee_positions[0])
    desired_positions = []
    for k in range(mpc_loops):
        t = k * dt_min
        pos_des = arm_trajectory.get_position(t, arm_ee_positions[0])
        desired_positions.append(pos_des)
    desired_positions = np.array(desired_positions)
    
    # 绘制末端轨迹（3D，惯性系）
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 实际轨迹（蓝色）
    ax.plot(arm_ee_positions[:, 0], arm_ee_positions[:, 1], arm_ee_positions[:, 2], 
            'b-', linewidth=2, label='Actual Trajectory')
    ax.scatter(arm_ee_positions[0, 0], arm_ee_positions[0, 1], arm_ee_positions[0, 2], 
               c='g', s=100, marker='o', label='Start')
    ax.scatter(arm_ee_positions[-1, 0], arm_ee_positions[-1, 1], arm_ee_positions[-1, 2], 
               c='r', s=100, marker='o', label='End')
    
    # 期望轨迹（红色虚线）
    ax.plot(desired_positions[:, 0], desired_positions[:, 1], desired_positions[:, 2], 
            'r--', linewidth=2, alpha=0.7, label='Desired Trajectory')
    
    # 基座轨迹（灰色）
    ax.plot(base_positions[:, 0], base_positions[:, 1], base_positions[:, 2], 
            'gray', linewidth=1, alpha=0.5, label='Base Trajectory')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Arm End-Effector Trajectory (Inertial Frame)')
    
    # 强制XYZ刻度一致
    all_positions = np.vstack([arm_ee_positions, desired_positions, base_positions])
    max_range = np.max([
        all_positions[:, 0].max() - all_positions[:, 0].min(),
        all_positions[:, 1].max() - all_positions[:, 1].min(),
        all_positions[:, 2].max() - all_positions[:, 2].min()
    ]) / 2
    mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) / 2
    mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) / 2
    mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.legend()
    plt.show()

    # Visualize robot
    robot_instance.initViewer()
    robot_instance.loadViewerModel("pinocchio")
    robot_instance.display(q0)
    viewer = robot_instance.viewer
    for _ in range(50):
        for (q, forces) in zip(ocp.q_sol, ocp.forces_sol):
            robot_instance.display(q)
            visualize_forces(viewer, robot, model, data, q, forces)
            time.sleep(dt_min)


if __name__ == "__main__":
    main()
