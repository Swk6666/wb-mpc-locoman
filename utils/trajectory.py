"""
轨迹生成模块
用于生成机械臂末端执行器的期望速度轨迹
支持阻抗控制（PD）来补偿位置误差
"""

import numpy as np


class TrajectoryBase:
    """轨迹基类，提供阻抗控制功能"""
    
    def __init__(self, kp: np.ndarray = None, kd: np.ndarray = None):
        """
        参数：
            kp: 位置增益 [kp_x, kp_y, kp_z]，默认为零（纯速度跟踪）
            kd: 速度增益 [kd_x, kd_y, kd_z]，默认为零
        """
        self.kp = kp if kp is not None else np.zeros(3)
        self.kd = kd if kd is not None else np.zeros(3)
        self._start_pos = None  # 轨迹起始位置（首次调用时记录）
        self._prev_vel = np.zeros(3)  # 上一次的速度（用于速度误差计算）
    
    def set_gains(self, kp: np.ndarray = None, kd: np.ndarray = None):
        """设置PD增益"""
        if kp is not None:
            self.kp = np.array(kp)
        if kd is not None:
            self.kd = np.array(kd)
    
    def reset(self, start_pos: np.ndarray = None):
        """重置轨迹，设置新的起始位置"""
        self._start_pos = start_pos
        self._prev_vel = np.zeros(3)
    
    def get_velocity(self, t: float) -> np.ndarray:
        """子类实现：计算时刻t的期望速度（不含阻抗修正）"""
        raise NotImplementedError
    
    def get_position(self, t: float, start_pos: np.ndarray = None) -> np.ndarray:
        """子类实现：计算时刻t的期望位置"""
        raise NotImplementedError
    
    def get_velocity_with_impedance(
        self, 
        t: float, 
        current_pos: np.ndarray,
        current_vel: np.ndarray = None,
    ) -> np.ndarray:
        """
        计算带阻抗控制的期望速度
        
        v_cmd = v_des + Kp * (p_des - p_cur) + Kd * (v_des - v_cur)
        
        参数：
            t: 当前时间（秒）
            current_pos: 当前末端位置 [x, y, z]
            current_vel: 当前末端速度 [vx, vy, vz]，可选
        
        返回：
            [vx, vy, vz]: 修正后的速度命令
        """
        # 首次调用时记录起始位置
        if self._start_pos is None:
            self._start_pos = current_pos.copy()
        
        # 期望速度和位置
        v_des = self.get_velocity(t)
        p_des = self.get_position(t, self._start_pos)
        
        # 位置误差修正
        pos_error = p_des - current_pos
        v_cmd = v_des + self.kp * pos_error
        
        # 速度误差修正（如果提供了当前速度）
        if current_vel is not None and np.any(self.kd != 0):
            vel_error = v_des - current_vel
            v_cmd += self.kd * vel_error
        
        return v_cmd


class CircleTrajectory(TrajectoryBase):
    """
    圆形轨迹生成器
    在指定平面内画圆，同时可以叠加其他方向的线性运动
    支持阻抗控制（PD）来补偿位置误差
    """
    def __init__(
        self,
        radius: float = 0.1,
        period: float = 3.0,
        plane: str = "yz",
        linear_vel: np.ndarray = None,
        kp: np.ndarray = None,
        kd: np.ndarray = None,
    ):
        """
        参数：
            radius: 圆的半径（米）
            period: 画一圈的周期（秒）
            plane: 画圆的平面 "xy", "xz", or "yz"
            linear_vel: 叠加的线性速度 [vx, vy, vz]（米/秒），默认为零
            kp: 位置增益 [kp_x, kp_y, kp_z]，用于阻抗控制
            kd: 速度增益 [kd_x, kd_y, kd_z]，用于阻抗控制
        """
        super().__init__(kp=kp, kd=kd)
        
        self.radius = radius
        self.period = period
        self.plane = plane
        self.linear_vel = linear_vel if linear_vel is not None else np.zeros(3)
        
        # 角速度
        self.omega = 2 * np.pi / period
        
    def get_velocity(self, t: float) -> np.ndarray:
        """
        计算时刻t的期望速度
        
        参数：
            t: 当前时间（秒）
        
        返回：
            [vx, vy, vz]: 三维速度向量
        """
        theta = self.omega * t
        v_tangent = self.radius * self.omega
        
        # 圆形轨迹的切向速度
        if self.plane == "xy":
            vx_circle = -v_tangent * np.sin(theta)
            vy_circle = v_tangent * np.cos(theta)
            vz_circle = 0
        elif self.plane == "xz":
            vx_circle = -v_tangent * np.sin(theta)
            vy_circle = 0
            vz_circle = v_tangent * np.cos(theta)
        elif self.plane == "yz":
            vx_circle = 0
            vy_circle = -v_tangent * np.sin(theta)
            vz_circle = v_tangent * np.cos(theta)
        else:
            raise ValueError(f"Unknown plane: {self.plane}")
        
        # 叠加线性速度
        vx = vx_circle + self.linear_vel[0]
        vy = vy_circle + self.linear_vel[1]
        vz = vz_circle + self.linear_vel[2]
        
        return np.array([vx, vy, vz])
    
    def get_position(self, t: float, start_pos: np.ndarray = None) -> np.ndarray:
        """
        计算时刻t的期望位置（用于参考/可视化）
        
        参数：
            t: 当前时间（秒）
            start_pos: 起始位置 [x, y, z]，默认为原点
        
        返回：
            [x, y, z]: 三维位置向量
        """
        if start_pos is None:
            start_pos = np.zeros(3)
            
        theta = self.omega * t
        
        # 圆形轨迹的位置偏移
        if self.plane == "xy":
            dx_circle = self.radius * (np.cos(theta) - 1)  # 从(r,0)开始
            dy_circle = self.radius * np.sin(theta)
            dz_circle = 0
        elif self.plane == "xz":
            dx_circle = self.radius * (np.cos(theta) - 1)
            dy_circle = 0
            dz_circle = self.radius * np.sin(theta)
        elif self.plane == "yz":
            dx_circle = 0
            dy_circle = self.radius * (np.cos(theta) - 1)
            dz_circle = self.radius * np.sin(theta)
        else:
            raise ValueError(f"Unknown plane: {self.plane}")
        
        # 叠加线性位移
        x = start_pos[0] + dx_circle + self.linear_vel[0] * t
        y = start_pos[1] + dy_circle + self.linear_vel[1] * t
        z = start_pos[2] + dz_circle + self.linear_vel[2] * t
        
        return np.array([x, y, z])


class HelixTrajectory(TrajectoryBase):
    """
    螺旋轨迹生成器
    在指定平面内画圆，同时沿垂直方向线性运动（形成螺旋）
    支持阻抗控制（PD）来补偿位置误差
    """
    def __init__(
        self,
        radius: float = 0.1,
        period: float = 3.0,
        plane: str = "yz",
        axial_vel: float = 0.1,
        kp: np.ndarray = None,
        kd: np.ndarray = None,
    ):
        """
        参数：
            radius: 圆的半径（米）
            period: 画一圈的周期（秒）
            plane: 画圆的平面 "xy", "xz", or "yz"
            axial_vel: 沿螺旋轴方向的速度（米/秒）
            kp: 位置增益 [kp_x, kp_y, kp_z]，用于阻抗控制
            kd: 速度增益 [kd_x, kd_y, kd_z]，用于阻抗控制
        """
        super().__init__(kp=kp, kd=kd)
        
        # 根据平面确定轴向
        if plane == "xy":
            linear_vel = np.array([0, 0, axial_vel])  # z方向
        elif plane == "xz":
            linear_vel = np.array([0, axial_vel, 0])  # y方向
        elif plane == "yz":
            linear_vel = np.array([axial_vel, 0, 0])  # x方向
        else:
            raise ValueError(f"Unknown plane: {plane}")
        
        # 使用CircleTrajectory实现（不带阻抗，阻抗在本类处理）
        self._circle = CircleTrajectory(
            radius=radius,
            period=period,
            plane=plane,
            linear_vel=linear_vel,
        )
        
    def get_velocity(self, t: float) -> np.ndarray:
        return self._circle.get_velocity(t)
    
    def get_position(self, t: float, start_pos: np.ndarray = None) -> np.ndarray:
        return self._circle.get_position(t, start_pos)


# 预定义的轨迹配置
TRAJECTORY_PRESETS = {
    "circle_xy": CircleTrajectory(radius=0.1, period=3.0, plane="xy"),
    "circle_xz": CircleTrajectory(radius=0.1, period=3.0, plane="xz"),
    "circle_yz": CircleTrajectory(radius=0.1, period=3.0, plane="yz"),
    "helix_x": HelixTrajectory(radius=0.1, period=3.0, plane="yz", axial_vel=0.1),
    "helix_y": HelixTrajectory(radius=0.1, period=3.0, plane="xz", axial_vel=0.1),
    "helix_z": HelixTrajectory(radius=0.1, period=3.0, plane="xy", axial_vel=0.1),
}
