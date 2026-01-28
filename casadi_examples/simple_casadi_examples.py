import casadi as ca

# 使用 SX 符号量，构建函数与梯度/海森矩阵
def example_symbolic_function():
    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    f = (x - 1) ** 2 + (y + 2) ** 2
    # 计算梯度（对 x, y 求偏导）
    grad_f = ca.jacobian(f, ca.vertcat(x, y))
    # 计算海森矩阵（对 x, y 的二阶导）
    hess_f = ca.hessian(f, ca.vertcat(x, y))[0]
    # 封装为可数值调用的 CasADi 函数
    F = ca.Function("F", [x, y], [f, grad_f, hess_f])
    val = F(3.0, -1.0)
    print("example_symbolic_function:")
    print("  f, grad, hess =", val)

# 用 Opti 构建一个小型二次规划（QP）
def example_opti_qp():
    # Opti 是 CasADi 的优化求解器，用于构建和求解优化问题
    # 先创建一个空的优化问题容器
    opti = ca.Opti()
    # 创建决策变量 x，2 维向量
    x = opti.variable(2)
    # Opti 里的决策变量是 MX，在 Opti 里一般统一用 MX，不能混用SX 和 MX
    Q = ca.diag(ca.MX([2.0, 5.0]))
    c = ca.MX([1.0, -3.0])
    # 目标函数：0.5 * x^T Q x + c^T x
    #  CasADi 中乘法用mtimes和@都可以，区别是mtimes是多项连乘，@是矩阵乘法
    cost = 0.5 * ca.mtimes([x.T, Q, x]) + c.T @ x
    # 把 cost 作为优化目标函数挂到 Opti 问题里，意思是“求解时最小化这个 cost”。
    opti.minimize(cost)
    # 等式约束的写法
    opti.subject_to(x[0] + x[1] == 1.0)
    # 不等式约束示例：x0 >= 0.5
    opti.subject_to(x[0] >= -0.2)
    # 盒约束：-1 <= x <= 1
    opti.subject_to(opti.bounded(-1.0, x, 1.0))
    # 设置求解器，这里用 ipopt，也可以用其他求解器，如 osqp, ipopt, sqpmethod, qpOASES, odrv 等
    opti.solver("ipopt", {"print_time": True}, {"print_level": 0})
    sol = opti.solve()
    print("example_opti_qp:")
    print("  x =", sol.value(x))


def example_parameters_and_function():
    # 参数 + to_function：把求解器封装成可调用函数
    opti = ca.Opti()
    x = opti.variable(2)
    p = opti.parameter(2)
    # 目标：x 尽量接近 p
    # ca.sumsqr：sum of squares，即 (x - p) 的平方和
    opti.minimize(ca.sumsqr(x - p))
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0})

    # 将求解过程封装为函数：输入参数 p，输出最优解 x
    solver_fn = opti.to_function("solver_fn", [p], [x])
    print("example_parameters_and_function:")
    print("  x(p=[1,2]) =", solver_fn(ca.DM([1.0, 2.0])))


def example_kinematics_ops():
    # 向量运算示例（对应仓库里用到的 ca.cross）
    v = ca.SX.sym("v", 3)
    w = ca.SX.sym("w", 3)
    # 叉乘：v x w
    cross_vw = ca.cross(v, w)
    V = ca.Function("V", [v, w], [cross_vw])
    print("example_kinematics_ops:")
    print("  cross([1,0,0],[0,1,0]) =", V([1, 0, 0], [0, 1, 0]))




def main():
    example_kinematics_ops()





if __name__ == "__main__":
    main()
