import time
import numpy as np
import casadi as ca
import osqp
from scipy import sparse

from utils.gait_sequence import *
from dynamics import DynamicsCentroidalVel


class OCP:
    """最优控制问题（Optimal Control Problem）基类"""
    
    def __init__(self, robot, nodes, tau_nodes, warm_start):
        """初始化OCP对象
        
        参数:
            robot: 机器人对象，包含模型、数据和步态序列
            nodes: OCP的节点数（预测时域的离散点数）
            tau_nodes: 前几个节点添加力矩限制
            warm_start: 是否使用热启动（用上一次的解初始化）
        """
        # ========== 机器人模型相关 ==========
        self.robot = robot
        self.model = robot.model  # Pinocchio模型
        self.data = robot.data    # Pinocchio数据
        self.gait_sequence = robot.gait_sequence  # 步态序列
        self.foot_frames = robot.foot_frames      # 脚部坐标系ID列表
        self.arm_ee_frame = robot.arm_ee_frame    # 机械臂末端执行器坐标系ID
        
        # 所有末端执行器坐标系（脚+机械臂）
        self.ee_frames = self.foot_frames.copy()
        if self.arm_ee_frame:
            self.ee_frames.append(self.arm_ee_frame)        
        self.arm_joints = robot.arm_joints  # 机械臂关节数
        self.n_feet = len(self.foot_frames)  # 脚的数量

        # ========== 维度信息 ==========
        self.nq = robot.nq  # 广义坐标维度（位置）
        self.nv = robot.nv  # 广义速度维度
        self.nf = robot.nf  # 接触力维度
        self.nj = robot.nj  # 关节数量

        # ========== OCP配置 ==========
        self.nodes = nodes  # 预测时域节点数
        self.tau_nodes = tau_nodes  # 前几个节点添加力矩限制
        self.warm_start = warm_start  # 是否使用热启动
        self.mass = self.data.mass[0]  # 机器人总质量
        self.opti = ca.Opti()  # CasADi优化接口

        # ========== 存储求解结果 ==========
        self.q_sol = []  # 位置轨迹
        self.v_sol = []  # 速度轨迹
        self.a_sol = []  # 加速度轨迹
        self.forces_sol = []  # 接触力轨迹

    def setup_problem(self):
        """构建完整的优化问题
        
        按顺序设置：决策变量 -> 参数 -> 目标 -> 约束 -> 目标函数
        """
        self.setup_variables()     # 1. 设置决策变量（状态、控制输入等）
        self.setup_parameters()    # 2. 设置参数（初始状态、步态调度等）
        self.setup_targets()       # 3. 设置期望状态和控制输入
        self.setup_constraints()   # 4. 设置约束（动力学、接触、关节限制等）
        obj = self.setup_objective()  # 5. 设置目标函数
        self.opti.minimize(obj)    # 6. 最小化目标函数

    def setup_variables(self):
        """
        Initialize decision variables.
        """
        pass

    def setup_parameters(self):
        """
        设置所有OCP共用的参数
        这些参数在优化过程中保持不变，但可以在每次MPC迭代时更新
        """
        # ========== 状态和时间参数 ==========
        self.x_init = self.opti.parameter(self.nx)  # 初始状态（位置+速度）
        self.dt_min = self.opti.parameter(1)  # 第一个时间步长（用于仿真）
        self.dt_max = self.opti.parameter(1)  # 最后一个时间步长
        
        # ========== 步态调度参数 ==========
        # 接触调度：每个脚在每个节点是否接触地面（0=摆动，1=接触）
        self.contact_schedule = self.opti.parameter(self.n_feet, self.nodes)
        # 摆动调度：每个脚在摆动相位中的进度（0=开始摆动，1=结束摆动）
        self.swing_schedule = self.opti.parameter(self.n_feet, self.nodes)
        self.n_contacts = self.opti.parameter(1)  # 当前接触地面的脚的数量
        self.swing_period = self.opti.parameter(1)  # 摆动周期（秒）
        self.swing_height = self.opti.parameter(1)  # 最大摆动高度（米）
        self.swing_vel_limits = self.opti.parameter(2)  # 摆动起始和结束时的速度限制

        # ========== 优化权重参数 ==========
        self.Q_diag = self.opti.parameter(self.ndx_opt)  # 状态误差权重对角矩阵
        self.R_diag = self.opti.parameter(self.nu_opt[0])  # 控制输入权重对角矩阵

        # ========== 跟踪目标参数 ==========
        self.base_vel_des = self.opti.parameter(6)  # 期望基座速度（线速度3维 + 角速度3维）
        self.arm_vel_des = self.opti.parameter(3)  # 期望机械臂末端执行器线速度
        self.arm_force_des = self.opti.parameter(3)  # 期望机械臂末端执行器力

        # ========== 自适应时间步长 ==========
        # 计算从dt_min到dt_max的指数增长时间步长序列
        ratio = self.dt_max / self.dt_min  # 时间步长比率
        gamma = ratio ** (1 / (self.nodes - 1))  # 增长因子（每步乘以gamma）
        self.dts = [self.dt_min * gamma**i for i in range(self.nodes)]  # 生成时间步长列表

    def setup_targets(self):
        """
        Determine desired state and input.
        """
        pass

    def setup_objective(self):
        """
        设置默认目标函数：使用状态权重矩阵Q和控制输入权重矩阵R的二次型
        目标：最小化状态误差和控制输入误差
        """
        obj = 0
        Q = ca.diag(self.Q_diag)  # 状态权重对角矩阵
        R = ca.diag(self.R_diag)  # 控制输入权重对角矩阵
        
        # 对每个节点累加代价
        for i in range(self.nodes):
            # 跟踪期望状态和控制输入
            dx = self.DX_opt[i]  # 当前节点的状态增量
            u = self.U_opt[i]    # 当前节点的控制输入
            err_dx = dx - self.dx_des  # 状态误差
            err_u = u - self.u_des     # 控制输入误差
            # 二次型代价：误差的加权平方和
            obj += err_dx.T @ Q @ err_dx  # 状态误差代价
            obj += err_u.T @ R @ err_u    # 控制输入误差代价

        # 终端状态代价（最后一个节点）
        dx = self.DX_opt[self.nodes]
        err_dx = dx - self.dx_des
        obj += err_dx.T @ Q @ err_dx  # 终端状态误差代价

        return obj

    def setup_constraints(self, mu=0.9):
        """
        设置所有OCP共用的约束条件
        动力学约束在子类中实现
        
        参数:
            mu: 摩擦系数（默认0.9）
        """
        # ========== 初始状态约束 ==========
        # 第一个节点的状态增量必须为0（从给定初始状态开始）
        self.opti.subject_to(self.DX_opt[0] == [0] * self.ndx_opt)

        # ========== 机械臂末端执行器全局速度目标计算 ==========
        if self.arm_ee_frame:
            # 从初始状态提取信息
            q_0 = self.x_init[:self.nq]  # 初始位置
            arm_pos_0 = self.dyn.get_frame_position(self.arm_ee_frame)(q_0)  # 机械臂末端位置
            base_pos_0 = self.dyn.get_base_position()(q_0)  # 基座位置
            base_rot_0 = self.dyn.get_base_rotation()(q_0)  # 基座旋转矩阵

            # 将相对于基座的期望速度转换为全局坐标系
            arm_vel_des_global = base_rot_0 @ self.arm_vel_des
            arm_vel_des_global[2] = self.arm_vel_des[2]  # 保持z方向速度
            arm_vel_des_global += self.base_vel_des[:3]  # 添加基座线速度

            # 考虑基座角速度对末端执行器的影响
            base_ang_vel = self.base_vel_des[3:]  # 基座角速度
            arm_pos_rel = arm_pos_0 - base_pos_0  # 末端相对于基座的位置
            ang_vel_correction = ca.cross(base_ang_vel, arm_pos_rel)  # 角速度引起的线速度
            arm_vel_des_global += ang_vel_correction

        # ========== 对每个节点设置约束 ==========
        for i in range(self.nodes):
            # 获取当前节点的状态和输入信息
            q = self.get_q(i)        # 位置
            v = self.get_v(i)        # 速度
            forces = self.get_forces(i)  # 接触力

            # 动力学约束（在子类中实现）
            self.setup_dynamics_constraints(i)

            # ========== 接触和摆动约束 ==========
            for idx, frame_id in enumerate(self.foot_frames):
                f_e = forces[idx * 3 : (idx + 1) * 3]  # 提取该脚的接触力（3维）

                # 获取接触和摆动信息
                in_contact = self.contact_schedule[idx, i]  # 是否接触（0或1）
                swing_phase = self.swing_schedule[idx, i]   # 摆动相位（0到1）

                # --- 接触约束：摩擦锥 ---
                f_normal = f_e[2]  # 法向力（z方向）
                f_tangent_square = f_e[0]**2 + f_e[1]**2  # 切向力的平方
                # 法向力必须非负（只能推不能拉）
                self.opti.subject_to(in_contact * f_normal >= 0)
                # 摩擦锥约束：切向力 <= mu * 法向力
                self.opti.subject_to(in_contact * mu**2 * f_normal**2 >= in_contact * f_tangent_square)

                # --- 摆动约束：零接触力 ---
                self.opti.subject_to((1 - in_contact) * f_e == [0] * 3)

                # 第一步特殊处理：不添加速度约束（除了质心速度动力学）
                if i == 0 and type(self.dyn) != DynamicsCentroidalVel:
                    continue

                # --- 接触约束：xy方向零速度（脚不滑动） ---
                vel = self.dyn.get_frame_velocity(frame_id)(q, v)  # 脚的速度
                vel_xy = vel[:2]  # xy方向速度
                self.opti.subject_to(in_contact * vel_xy == [0] * 2)

                # --- z方向速度约束 ---
                vel_z = vel[2]  # z方向速度
                # 计算期望的z方向速度（摆动时使用样条曲线）
                vel_z_des = get_spline_vel_z(
                    swing_phase,
                    swing_period=self.swing_period,
                    h_max=self.swing_height,
                    v_liftoff=self.swing_vel_limits[0],   # 抬起时的速度
                    v_touchdown=self.swing_vel_limits[1]  # 落地时的速度
                )
                vel_diff = vel_z - vel_z_des
                # 接触时z速度为0，摆动时跟踪样条速度
                self.opti.subject_to(in_contact * vel_z + (1 - in_contact) * vel_diff == 0)

            # ========== 热启动：初始化决策变量 ==========
            # 使用步态序列中的接触脚数量来计算期望控制输入
            self.opti.set_value(self.n_contacts, self.gait_sequence.n_contacts)
            self.opti.set_initial(self.DX_opt[i], np.zeros(self.ndx_opt))
            u_warm = self.opti.value(self.u_des)[:self.nu_opt[i]]
            self.opti.set_initial(self.U_opt[i], u_warm)

            # ========== 机械臂末端执行器约束 ==========
            # 机械臂末端执行器力约束
            if self.arm_ee_frame:
                f_e = forces[3*self.n_feet:]  # 提取机械臂末端执行器的力
                self.opti.subject_to(f_e == self.arm_force_des)  # 跟踪期望力

            # 第一步特殊处理（同上）
            if i == 0 and type(self.dyn) != DynamicsCentroidalVel:
                continue

            # 机械臂末端执行器速度约束
            if self.arm_ee_frame:
                vel = self.dyn.get_frame_velocity(self.arm_ee_frame)(q, v)  # 末端执行器速度
                vel_lin = vel[:3]  # 线速度
                vel_diff = vel_lin - arm_vel_des_global  # 与期望全局速度的差
                self.opti.subject_to(vel_diff == [0] * 3)  # 跟踪期望速度

            # ========== 关节限制 ==========
            pos_min = self.robot.joint_pos_min  # 关节位置下限
            pos_max = self.robot.joint_pos_max  # 关节位置上限
            vel_min = -self.robot.joint_vel_max  # 关节速度下限
            vel_max = self.robot.joint_vel_max   # 关节速度上限
            q_j = q[7:]  # 跳过基座四元数，提取关节位置
            v_j = v[6:]  # 跳过基座角速度，提取关节速度
            # 添加关节位置和速度的边界约束
            self.opti.subject_to(self.opti.bounded(pos_min, q_j, pos_max))
            self.opti.subject_to(self.opti.bounded(vel_min, v_j, vel_max))

        # ========== 终端状态热启动 ==========
        self.opti.set_initial(self.DX_opt[self.nodes], np.zeros(self.ndx_opt))

        # ========== 存储上一次的解用于热启动 ==========
        self.DX_prev = None  # 上一次的状态增量解
        self.U_prev = None   # 上一次的控制输入解
        self.lam_g = None    # 上一次的拉格朗日乘子

    def setup_dynamics_constraints(self, i):
        pass

    def get_q(self, i):
        pass

    def get_v(self, i):
        pass

    def get_forces(self, i):
        pass

    def set_weights(self):
        """
        设置权重矩阵对角元素 Q_diag 和 R_diag
        在子类中实现具体的权重值
        """
        pass

    def set_time_params(self, dt_min, dt_max):
        """设置时间步长参数"""
        self.opti.set_value(self.dt_min, dt_min)
        self.opti.set_value(self.dt_max, dt_max)

    def set_swing_params(self, swing_height, swing_vel_limits):
        """设置摆动参数：摆动高度和速度限制"""
        self.opti.set_value(self.swing_height, swing_height)
        self.opti.set_value(self.swing_vel_limits, swing_vel_limits)

    def set_tracking_targets(self, base_vel_des, arm_vel_des=None, arm_force_des=None):
        """设置跟踪目标：基座速度、机械臂末端速度和力"""
        self.opti.set_value(self.base_vel_des, base_vel_des)
        if self.arm_ee_frame:
            self.opti.set_value(self.arm_vel_des, arm_vel_des)
            self.opti.set_value(self.arm_force_des, arm_force_des)

    def update_initial_state(self, x_init):
        """更新初始状态参数"""
        self.opti.set_value(self.x_init, x_init)

    def update_gait_sequence(self, t_current):
        """
        根据当前时间更新步态序列
        
        参数:
            t_current: 当前仿真时间
        """
        # 获取当前的时间步长序列
        dts = [self.opti.value(self.dts[i]) for i in range(self.nodes)]
        # 根据当前时间和时间步长计算接触和摆动调度
        contact_schedule, swing_schedule = self.gait_sequence.get_gait_schedule(t_current, dts, self.nodes)
        n_contacts = self.gait_sequence.n_contacts  # 当前接触脚数量
        swing_period = self.gait_sequence.swing_period  # 摆动周期
        # 更新优化问题中的参数
        self.opti.set_value(self.contact_schedule, contact_schedule)
        self.opti.set_value(self.swing_schedule, swing_schedule)
        self.opti.set_value(self.n_contacts, n_contacts)
        self.opti.set_value(self.swing_period, swing_period)

    def update_params(self, x_init, t_current):
        """
        更新状态和步态序列参数
        在每次MPC迭代时调用
        
        参数:
            x_init: 初始状态
            t_current: 当前仿真时间
        """
        self.update_initial_state(x_init)  # 更新初始状态
        self.update_gait_sequence(t_current)  # 更新步态序列
        if self.warm_start:
            self.warm_start_variables()  # 如果启用热启动，用上一次的解初始化

    def get_solver_params(self):
        """
        返回求解器参数作为堆叠向量
        用于外部求解器函数调用
        """
        params = [self.opti.value(p, self.opti.initial()) for p in self.solver_params]
        return params

    def warm_start_variables(self):
        """
        Warm start decision variables from previous solution.
        """
        pass

    def init_solver(self, solver, solver_args):
        self.solver = solver

        # Get info from self.opti and store constraint data
        x = self.opti.x
        p = self.opti.p
        f = self.opti.f
        g = self.opti.g
        lbg = self.opti.lbg
        ubg = self.opti.ubg
        self.g_data = ca.Function("g_data", [x, p], [g, lbg, ubg])

        # Initialize solver
        if self.solver == "fatrop" or self.solver == "ipopt":
            opts = solver_args["opts"]
            self.opti.solver(self.solver, opts)

            # Store solver params
            self.solver_params = [self.x_init, self.dt_min, self.dt_max, self.contact_schedule, self.swing_schedule,
                                  self.n_contacts, self.swing_period, self.swing_height, self.swing_vel_limits,
                                  self.Q_diag, self.R_diag, self.base_vel_des]
            if self.arm_ee_frame:
                self.solver_params += [self.arm_vel_des]
                self.solver_params += [self.arm_force_des]
            if self.warm_start:
                self.solver_params += [self.opti.x]

            # Store solver function (to be evaluated or compiled)
            self.solver_function = self.opti.to_function(
                "solver_function",
                self.solver_params,  # input (params)
                [self.opti.x],  # output (solution)   
            )

        elif self.solver == "osqp":
            self.sqp_iters = solver_args["iters"]
            self.osqp_opts = solver_args["opts"]

            # Store SQP data
            J_g = ca.jacobian(g, x)
            hess_f, grad_f = ca.hessian(f, x)
            self.sqp_data = ca.Function("sqp_data", [x, p], [grad_f, J_g, g, lbg, ubg])
            self.hess_data = ca.Function("hess_data", [x, p], [hess_f])
            self.f_data = ca.Function("f_data", [x, p], [f, grad_f])

            # Store initial hessian (diagonal!)
            x_val = self.opti.value(self.opti.x, self.opti.initial())
            p_val = self.opti.value(self.opti.p, self.opti.initial())
            hess_val = self.hess_data(x_val, p_val)
            self.hess_diag = np.diag(hess_val)

            # Setup OSQP with dummy data
            A_rows, A_cols = J_g.sparsity().get_triplet()  # store sparsity pattern
            A = sparse.csc_matrix((np.ones_like(A_rows), (A_rows, A_cols)), shape=J_g.shape)
            P = sparse.csc_matrix(np.diag(self.hess_diag))  # diagonal
            q = np.ones(grad_f.shape)
            l = -np.ones(g.shape)
            u = np.ones(g.shape)

            self.osqp_prob = osqp.OSQP()
            self.osqp_prob.setup(P, q, A, l, u, **self.osqp_opts)

        else:
            raise ValueError(f"Solver {self.solver} not supported")

    def compile_solver(self):
        if self.solver == "fatrop":
            # Generate C code for solver function
            self.solver_function.generate("solver_function.c")
        else:
            raise NotImplementedError(f"Solver compilation not implemented for: {self.solver}")

        self.compile_solution(num_steps=3)

    def compile_solution(self, num_steps):
        """
        Compile the first num_steps of the solution, to easily load on hardware.
        """
        pass

    def solve(self, retract_all=True):
        print(f"************** {self.solver} **************")
        if self.solver == "fatrop" or self.solver == "ipopt":
            try:
                self.sol = self.opti.solve()
            except RuntimeError as e:
                print(f"Solver failed: {e}")
                self.sol = None

            self.solve_time = self.sol.stats()["t_wall_total"]
            self.retract_opti_sol(retract_all)

            # Check constraint violation
            sol_x = self.sol.value(self.opti.x)
            params = self.opti.value(self.opti.p)
            g, lbg, ubg = self.g_data(sol_x, params)
            self.constr_viol = self.constr_viol_norm_inf(g, lbg, ubg)
            print("CV (inf norm): ", self.constr_viol)

        elif self.solver == "osqp":
            # Get current state and parameters
            current_x = self.opti.value(self.opti.x, self.opti.initial())
            current_params = self.opti.value(self.opti.p, self.opti.initial())
            start_time = time.time()

            for _ in range(self.sqp_iters):
                # Get data
                start = time.time()
                grad_f, J_g, g, lbg, ubg = self.sqp_data(current_x, current_params)
                end = time.time()
                print("Data time (ms): ", (end - start) * 1000)

                start = time.time()
                A = np.array(J_g.nonzeros())
                q = np.array(grad_f)
                l = np.array(lbg - g)
                u = np.array(ubg - g)
                self.osqp_prob.update(q=q, Ax=A, l=l, u=u)
                end = time.time()
                print("Update time (ms): ", (end - start) * 1000)

                # Solve
                start = time.time()
                sol_dx = self.osqp_prob.solve().x
                end = time.time()
                print("Solve time (ms): ", (end - start) * 1000)

                # Line search
                current_x = self._armijo_line_search(sol_dx, current_x, current_params)

            end_time = time.time()
            self.solve_time = end_time - start_time

            g, lbg, ubg = self.g_data(current_x, current_params)
            self.constr_viol = self.constr_viol_norm_inf(g, lbg, ubg)
            print("CV (inf norm): ", self.constr_viol)

            self.retract_stacked_sol(current_x, retract_all)

    def retract_opti_sol(self, retract_all=True):
        pass

    def retract_stacked_sol(self, sol_x, retract_all=True):
        pass

    def _armijo_line_search(self, dx, current_x, current_params):
        # Params
        armijo_factor = 1e-4
        a = 1.0
        a_min = 1e-4
        a_decay = 0.5
        g_max = 1e-3
        g_min = 1e-5
        gamma = 1e-5

        # Get current data
        f, grad_f = self.f_data(current_x, current_params)
        g, lbg, ubg = self.g_data(current_x, current_params)

        g_metric = self.constr_viol_norm_2(g, lbg, ubg)
        armijo_metric = grad_f.T @ dx
        accepted = False

        while not accepted and a > a_min:
            # Evaluate new solution
            new_x = current_x + a * dx
            new_f, _ = self.f_data(new_x, current_params)
            new_g, lbg, ubg = self.g_data(new_x, current_params)

            new_g_metric = self.constr_viol_norm_2(new_g, lbg, ubg)
            if new_g_metric > g_max:
                if new_g_metric < (1 - gamma) * g_metric:
                    print("Line search: g metric high, but improving")
                    accepted = True
    
            elif max(new_g_metric, g_metric) < g_min and armijo_metric < 0:
                if new_f <= f + armijo_factor * armijo_metric:
                    print("Line search: g metric low, f improving")
                    accepted = True

            elif new_f <= f - gamma * new_g_metric or new_g_metric < (1 - gamma) * g_metric:
                print("Line search: f improving or g metric improving")
                accepted = True

            a *= a_decay  # Reduce step size
            f = new_f
            g_metric = new_g_metric

        if accepted:
            # Print info (need to adjust a)
            print(f"a: {a / a_decay}, f: {f}, g metric: {g_metric}")
            return new_x

        else:
            print("Line search: Didn't converge!")
            return current_x

    def constr_viol_norm_2(self, g, lbg, ubg):
        lb_violations = np.maximum(0, lbg - g)
        ub_violations = np.maximum(0, g - ubg)
        violations = np.concatenate((lb_violations, ub_violations))

        metric = np.linalg.norm(violations)
        return metric

    def constr_viol_norm_inf(self, g, lbg, ubg):
        lb_violations = np.maximum(0, lbg - g)
        ub_violations = np.maximum(0, g - ubg)
        violations = np.concatenate((lb_violations, ub_violations))

        max_violation = np.max(np.abs(violations))
        return max_violation
