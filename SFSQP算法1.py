import numpy as np
from scipy.optimize import minimize

def sfsqp_algorithm(x0, mu0, max_iter=100, tol=1e-6):
    # ===================== 1. 初始化（动态适配维度） =====================
    n = len(x0)  # 变量维度（3维）
    m = len(mu0) # 乘子总数（2等式+2不等式=4）
    m_E = len(compute_eq_constraints(x0))  # 等式约束数（2）
    m_I = len(compute_ineq_constraints(x0))# 不等式约束数（2）
    
    # 算法参数
    epsilon0 = 1.0
    u_l = 1.0
    u_g = 1.0
    delta = 0.5
    xi = 0.1
    mu_max = 1e6
    alpha_bar = 0.5
    eta = 0.1
    gamma = 0.1
    r = 0.5
    tau = 0.5
    beta = 0.5
    theta = 1.0
    sigma_L0 = 1.0
    alpha_min = 1e-6  # 触发 FRP 的最小步长阈值

    # 状态变量（动态维度）
    x = x0.copy()
    mu = mu0.copy()
    mu_L = np.zeros_like(mu)  # 乘子维度和mu一致（4维）
    sigma_L = sigma_L0
    epsilon_sigma = epsilon0
    B = np.eye(n)  # 海森矩阵维度=变量数（3x3）
    k = 0
    Flag = "GLOBAL"  # 初始为全局模式

    # 滤子初始化
    F_g = [(u_g, -np.inf)]  # 全局滤子: [(p, phi)]
    F_l = [(u_l, -np.inf)]  # 局部滤子: [(h, f)]

    # 初始化上一次迭代点（用于 BFGS 更新）
    x_prev = x.copy()
    mu_prev = mu.copy()
    
    # ===================== 2. 主循环 =====================
    while not termination_condition(x, x_prev, mu, k, tol=1e-6, max_iter=100):        # -------------------- 3. 求解 QP 子问题，得到方向 (d, Dmu) --------------------
        ###Solve the subproblem (3.2) to generate the primal-dual search direction (dk , Δμk);
        d, Dmu = solve_stabilized_QP_subproblem(x, mu, B, sigma_L, mu_L)
        print(f"迭代 {k}: d = {np.linalg.norm(d):.4e}, Dmu = {np.linalg.norm(Dmu):.4e}")
        if np.linalg.norm(d) < 1e-5:
            return x, mu  # 方向过小，直接返回当前解

        # -------------------- 4. 线搜索（从 alpha=1 开始回溯） --------------------
        ###Set α = 1, xˆ = xk + αdk , μˆ = μk + αΔμk;
        alpha = 1.0
        x_hat = x + alpha * d
        mu_hat = mu + alpha * Dmu

        # 计算当前点的 p, phi, h, f
        p_hat = compute_stabilized_constraint_violation(x_hat, mu_hat, sigma_L, mu_L)
        phi_hat = compute_merit_function(x_hat, mu_hat, sigma_L, mu_L)
        h_hat = compute_constraint_violation(x_hat)
        f_hat = compute_objective(x_hat)
        print(f"迭代 {k}: alpha={alpha:.4f}, p={p_hat:.6f}, phi={phi_hat:.6f}, h={h_hat:.6f}, f={f_hat:.6f}")

        # 检查是否满足局部滤子或 KKT 误差缩减条件
        ###if (xˆ,μˆ) is acceptable to the local ﬁlter or ϑσ(xˆ,μˆ) ≤ τσ L,k 5 then
        if is_acceptable_to_local_filter(h_hat, f_hat, F_l, eta, gamma, u_l) or \
            (compute_KKT_error(x_hat, mu_hat) <= tau * sigma_L):
            # -------------------- 5. 进入局部模式，更新乘子和 sigma_L --------------------
            ###Update μL,k+1 and σ L,k+1 by (3.18), and set Flag=LOCAL and Fk+1g = {(ug, −∞)};
            mu_L = update_mu_L(mu_hat, mu_L, mu_max)
            sigma_L = update_sigma_L(sigma_L, compute_KKT_error(x_hat, mu_hat), beta, theta)
            Flag = "LOCAL"
            F_g = [(u_g, -np.inf)]  # 重置全局滤子

        else:
            # -------------------- 6. 调用内循环 G，得到可接受的 (x_hat, mu_hat) 和 alpha --------------------
            ###else：   Call Inner loop G to obtain a pair (xˆ,μˆ) and also a step size α;
            x_hat, mu_hat, alpha = inner_loop_G(x, mu, d, Dmu, F_g, alpha_min, eta, gamma, r, u_g, sigma_L, mu_L)
            p_hat = compute_stabilized_constraint_violation(x_hat, mu_hat, sigma_L, mu_L)
            phi_hat = compute_merit_function(x_hat, mu_hat, sigma_L, mu_L)

            # -------------------- 7. 检查步长是否小于 alpha_min，触发可行性恢复 --------------------
            ###if α < αkmin then
            if alpha < alpha_min:
                # FRP：可行性恢复阶段（修复维度）
                ###Set μk+1 = μL,k and go to the feasibility restoration phase for ﬁnding a new iterate xk+1 such that (xk+1,μk+1) is acceptable to the global ﬁlter Fkg ; 
                x_new, mu_new = feasibility_restoration_phase(x, mu, F_g, sigma_L, mu_L, u_g, eta, gamma)
                # 保持 mu_L, sigma_L, B 不变
                ###keep μ^L,k+1, σ^L,k+1, ε^k+1σ and Bk+1 unchanged;
                x, mu = x_new, mu_new
                # 将当前 (p, phi) 加入全局滤子
                ###set k = k + 1 and add (p(xk ,μk ), φ(xk ,μk )) to Fkg if p(xk ,μk ) > 0 and continue;
                if compute_stabilized_constraint_violation(x, mu, sigma_L, mu_L) > 0:
                    F_g.append((compute_stabilized_constraint_violation(x, mu, sigma_L, mu_L),
                                compute_merit_function(x, mu, sigma_L, mu_L)))
                k += 1
                continue  # 直接进入下一轮迭代

            # -------------------- 8. 检查近似一阶条件 (3.19)，决定是否更新参数 --------------------
            psi_val = compute_psi(x_hat, mu_hat, mu_L, sigma_L)
            ###if (3.19) is satisﬁed then
            ###Update μL,k+1 and σ L,k+1 by (3.18); Set Fk+1 g = {(ug, −∞)} and εk+1 σ = εkσ/10;
            if psi_val <= epsilon_sigma:
                mu_L = update_mu_L(mu_hat, mu_L, mu_max)
                sigma_L = update_sigma_L(sigma_L, compute_KKT_error(x_hat, mu_hat), beta, theta)
                F_g = [(u_g, -np.inf)]
                epsilon_sigma /= 10
            else:
                # 保持 mu_L, sigma_L 不变
                pass

        # -------------------- 9. 更新滤子 --------------------
        ###if (xˆ,μˆ) is acceptable to the local ﬁlter then
        ###Add (h(xˆ), f (xˆ)) to the local ﬁlter Fkl and update entries of the local ﬁlter;
        if is_acceptable_to_local_filter(h_hat, f_hat, F_l, eta, gamma, u_l):
            F_l.append((h_hat, f_hat))
        ###else if Flag=GLOBAL and p(xˆ,μˆ) > 0 then 
        ### Add (p(xˆ,μˆ), φ(xˆ,μˆ)) to the global ﬁlter Fkg and update entries of the global ﬁlter;
        elif Flag == "GLOBAL" and p_hat > 0:
            F_g.append((p_hat, phi_hat))

        # -------------------- 10. 更新迭代点和海森近似 --------------------
        ###Set xk+1 = xˆ, μk+1 = μˆ, εk+1 σ = εkσ , αk = α, k = k + 1, and Flag=GLOBAL; update Bk+1 24 by quasi-Newton formula;
        x, mu = x_hat, mu_hat
        # 拟牛顿更新 B（BFGS 公式）
        s = x - x_prev  # x_prev 是上一轮的 x
        lag_grad_cur = compute_lagrangian_gradient(x, mu)
        lag_grad_prev = compute_lagrangian_gradient(x_prev, mu_prev)
        y = lag_grad_cur - lag_grad_prev
        B = update_BFGS(B, s, y)
        x_prev, mu_prev = x.copy(), mu.copy()
        print(f'kkt_error = {compute_KKT_error(x, mu):.6e}')
        k += 1
        if k >= max_iter:
            print(f"达到最大迭代次数 {max_iter}，终止迭代")
            break

    return x, mu



# ===================== 所有维度硬编码改为动态适配 ======================
def compute_constraint_violation(x):
    """
    计算常规约束违反度 h(x)（动态适配任意约束数）
    定义：等式约束的绝对值 + 不等式约束的正部（违反部分）的和
    """
    # 1. 计算约束值（动态获取）
    c_E = compute_eq_constraints(x)  # 等式约束 [c_E1, c_E2]
    c_I = compute_ineq_constraints(x)  # 不等式约束 [c_I1, c_I2]
    
    # 2. 等式约束违反度：|c_E|（因为c_E必须=0）
    eq_viol = np.sum(np.abs(c_E))
    
    # 3. 不等式约束违反度：max(c_I, 0)（只有c_I>0时才违反）
    ineq_viol = np.sum(np.maximum(c_I, 0))
    
    # 4. 总约束违反度 h(x)
    h = eq_viol + ineq_viol
    return h

def compute_stabilized_constraint_violation(x, mu, sigma_L, mu_L):
    """
    计算稳定化约束违反度 p(x, μ)（全局滤子用，动态适配维度）
    """
    # 1. 动态拆分乘子
    m_E = len(compute_eq_constraints(x))  # 等式约束数量（2）
    m_I = len(compute_ineq_constraints(x))# 不等式约束数量（2）
    mu_E = mu[:m_E]
    mu_I = mu[m_E:]
    mu_E_L = mu_L[:m_E]
    mu_I_L = mu_L[m_E:]
    
    # 2. 计算原始约束值
    c_E = compute_eq_constraints(x)
    c_I = compute_ineq_constraints(x)
    
    # 3. 稳定化等式约束违反度（逐元素计算）
    stab_eq = c_E - sigma_L * (mu_E - mu_E_L)
    eq_viol_stab = np.sum(np.abs(stab_eq))
    
    # 4. 稳定化不等式约束违反度（只算正部）
    stab_ineq = -c_I + sigma_L * (mu_I - mu_I_L)
    ineq_viol_stab = np.sum(np.maximum(stab_ineq, 0))
    
    # 5. 总稳定化约束违反度 p(x,μ)
    p = eq_viol_stab + ineq_viol_stab
    return p

def compute_merit_function(x, mu, sigma_L, mu_L, rho=1.0):
    """
    计算辅助目标函数 φ(x, μ) = f(x) + ρ·p(x,μ)
    """
    f = compute_objective(x)
    p = compute_stabilized_constraint_violation(x, mu, sigma_L, mu_L)
    phi = f + rho * p
    return phi

def is_acceptable_to_local_filter(h, f, F_l, eta, gamma, u_l):
    """
    判断 (h, f) 是否被局部滤子接受（逻辑不变，维度自适应）
    """
    # 先检查h是否超过上限
    if h > u_l:
        return False
    
    # 遍历滤子中的所有历史点
    for (h_l, f_l) in F_l:
        cond1 = (h_l - h) >= eta * (h **2)
        cond2 = (f_l - f) >= gamma * (h** 2)
        if not (cond1 or cond2):
            return False
    return True

def is_acceptable_to_global_filter(p, phi, F_g, eta, gamma, u_g):
    """
    判断 (p, φ) 是否被全局滤子接受（逻辑不变，维度自适应）
    """
    if p > u_g:
        return False
    for (p_l, phi_l) in F_g:
        cond1 = (p_l - p) >= eta * (p **2)
        cond2 = (phi_l - phi) >= gamma * (p** 2)
        if not (cond1 or cond2):
            return False
    return True

def solve_stabilized_QP_subproblem(x, mu, B, sigma_L, mu_L):
    """
    求解稳定化QP子问题 (3.2)
    变量: d (n维), Δμ (m = m_E + m_I维)
    返回: d, Δμ
    """
    n = len(x)
    m_E = len(compute_eq_constraints(x))
    m_I = len(compute_ineq_constraints(x))
    total_vars = n + m_E + m_I

    # 当前点信息
    f_grad = compute_f_gradient(x)
    c_E = compute_eq_constraints(x)
    c_I = compute_ineq_constraints(x)
    c_E_jac = compute_eq_jacobian(x)      # shape (m_E, n)
    c_I_jac = compute_ineq_jacobian(x)    # shape (m_I, n)

    # 拆分乘子
    mu_E = mu[:m_E]
    mu_I = mu[m_E:]
    mu_E_L = mu_L[:m_E]
    mu_I_L = mu_L[m_E:]

    # 目标函数: ∇f^T d + 0.5 d^T B d + 0.5 σ_L ‖μ + Δμ‖²
    def objective(z):
        d = z[:n]
        Dmu = z[n:]
        # μ + Δμ 是整个乘子向量（等式的+不等式的）
        mu_plus_Dmu = mu + Dmu
        quad_term = 0.5 * d @ B @ d
        reg_term = 0.5 * sigma_L * np.sum(mu_plus_Dmu**2)
        return f_grad @ d + quad_term + reg_term

    # 可选：提供目标函数的梯度（可省略，SLSQP 会数值微分）
    # 为简化，这里不提供梯度，让 SLSQP 自动差分

    constraints = []

    # 等式约束: c_E[i] + ∇c_E[i]·d - σ_L (μ_E[i] + Δμ_E[i] - μ_E_L[i]) = 0
    for i in range(m_E):
        def eq_con(z, i=i):
            d = z[:n]
            Dmu_E = z[n:n+m_E]
            return (c_E[i] + (c_E_jac[i] @ d) 
                    - sigma_L * (mu_E[i] + Dmu_E[i] - mu_E_L[i]))
        constraints.append({'type': 'eq', 'fun': eq_con})

    # 不等式约束: c_I[i] + ∇c_I[i]·d - σ_L (μ_I[i] + Δμ_I[i] - μ_I_L[i]) ≤ 0
    # 转化为 >=0 形式: -c_I[i] - ∇c_I[i]·d + σ_L (μ_I[i] + Δμ_I[i] - μ_I_L[i]) ≥ 0
    for i in range(m_I):
        def ineq_con(z, i=i):
            d = z[:n]
            Dmu_I = z[n+m_E:]
            return (-c_I[i] - (c_I_jac[i] @ d) 
                    + sigma_L * (mu_I[i] + Dmu_I[i] - mu_I_L[i]))
        constraints.append({'type': 'ineq', 'fun': ineq_con})

    # 初始猜测（通常为零）
    z0 = np.zeros(total_vars)

    # 求解QP
    res = minimize(objective, z0, constraints=constraints, method='SLSQP',
                   options={'maxiter': 100, 'ftol': 1e-6})

    if res.success:
        d = res.x[:n]
        Dmu = res.x[n:]
        return d, Dmu
    else:
        # 求解失败时返回零方向，触发可行性恢复阶段
        print("QP子问题求解失败，返回零方向")
        return np.zeros(n), np.zeros(m_E + m_I)

# -------------------- 1. 目标函数 f(x) --------------------
def compute_objective(x):
    """
    目标函数 f(x) = (x₁ - 2)² + (x₂ - 1)²
    """
    x1, x2 = x
    return (x1 - 2)**2 + (x2 - 1)**2

# -------------------- 2. 目标函数梯度 ∇f(x) --------------------
def compute_f_gradient(x):
    """
    梯度 ∇f(x) = [2(x₁ - 2), 2(x₂ - 1)]
    """
    x1, x2 = x
    return np.array([2 * (x1 - 2), 2 * (x2 - 1)])
# -------------------- 3. 等式约束 c_E(x) --------------------
def compute_eq_constraints(x):
    """
    等式约束 c_E(x) = x₁ - 2x₂ + 1
    返回：shape=(1,) 的数组
    """
    x1, x2 = x
    c_E = x1 - 2 * x2 + 1
    return np.array([c_E])

# -------------------- 4. 不等式约束 c_I(x) --------------------
def compute_ineq_constraints(x):
    """
    不等式约束 c_I(x) = 0.25x₁² + x₂² - 1 （≤ 0 形式）
    返回：shape=(1,) 的数组
    """
    x1, x2 = x
    c_I = 0.25 * x1**2 + x2**2 - 1
    return np.array([c_I])

# -------------------- 5. 等式约束雅可比 ∇c_E(x) --------------------
def compute_eq_jacobian(x):
    """
    等式约束雅可比矩阵 ∇c_E(x) = [1, -2]
    形状：m_E × n = 1 × 2
    """
    return np.array([[1.0, -2.0]])

# -------------------- 6. 不等式约束雅可比 ∇c_I(x) --------------------
def compute_ineq_jacobian(x):
    """
    不等式约束雅可比矩阵 ∇c_I(x) = [0.5x₁, 2x₂]
    形状：m_I × n = 1 × 2
    """
    x1, x2 = x
    return np.array([[0.5 * x1, 2 * x2]])

# -------------------- 7. 计算KKT误差（3维+2等式+2不等式） --------------------
def compute_KKT_error(x, mu):
    f_grad = compute_f_gradient(x)
    c_E = compute_eq_constraints(x)       
    c_I = compute_ineq_constraints(x)     
    c_E_jac = compute_eq_jacobian(x)      
    c_I_jac = compute_ineq_jacobian(x)    

    # 动态拆分乘子
    m_E = len(c_E)
    mu_E = mu[:m_E]
    mu_I = mu[m_E:]

    # 1. 原始最优性误差
    lagrangian_grad = f_grad + c_E_jac.T @ mu_E + c_I_jac.T @ mu_I
    primal_optimality = np.linalg.norm(lagrangian_grad, 2)
    print("KKT1: ", primal_optimality)
    
    # 2. 等式约束可行性误差
    eq_feasibility = np.linalg.norm(c_E, 2)
    
    # 3. 不等式约束可行性误差
    ineq_feasibility = np.linalg.norm(np.maximum(c_I, 0), 2)
    
    # 4. 互补松弛误差
    complementarity = np.linalg.norm(mu_I * np.maximum(c_I, 0), 2)
    
    # 综合KKT误差
    kkt_error = np.sqrt(primal_optimality**2 + eq_feasibility**2 + ineq_feasibility**2 + complementarity**2)
    return kkt_error

# -------------------- 8. 终止条件判断（维度自适应） --------------------
def termination_condition(x, x_prev, mu, k, tol=1e-6, max_viol=1e-6, max_iter=10, min_step=1e-12):
    # 1. KKT 误差和可行性
    kkt = compute_KKT_error(x, mu)
    viol = compute_constraint_violation(x)
    if kkt < tol and viol < max_viol:
        return True
    # 2. 达到最大迭代次数
    if k >= max_iter:
        print("达到最大迭代次数")
        return True
    # 3. 变量变化极小，可能停滞
    if np.linalg.norm(x - x_prev) < min_step and k > 20:
        return True
    return False

# -------------------- 其他辅助函数（维度修复） --------------------
def update_mu_L(mu_hat, mu_L_old, mu_max):
    # 按 (3.18a) 更新 mu_L（逐元素判断）
    if np.linalg.norm(mu_hat, ord=np.inf) <= mu_max:
        return mu_hat
    else:
        return mu_L_old

def update_sigma_L(sigma_L_old, KKT_error, beta, theta):
    # 按 (3.18b) 更新 sigma_L
    return min(beta * sigma_L_old, theta * KKT_error)

def compute_psi(x, mu_hat, mu_L, sigma_L):
    """
    计算稳定化一阶最优性度量 ψ（维度自适应）
    """
    # 1. 基础计算
    f_grad = compute_f_gradient(x)
    c_E = compute_eq_constraints(x)
    c_I = compute_ineq_constraints(x)
    c_E_jac = compute_eq_jacobian(x)
    c_I_jac = compute_ineq_jacobian(x)
    
    # 2. 动态拆分乘子
    m_E = len(c_E)
    m_I = len(c_I)
    mu_E_hat = mu_hat[:m_E]
    mu_I_hat = mu_hat[m_E:]
    mu_E_L = mu_L[:m_E]
    mu_I_L = mu_L[m_E:]
    
    # 3. 构建ψ的块向量
    block1 = f_grad + c_E_jac.T @ mu_E_hat + c_I_jac.T @ mu_I_hat  # 3维
    block2 = c_E - sigma_L * (mu_E_hat - mu_E_L)  # 2维
    ineq_term = -c_I + sigma_L * (mu_I_hat - mu_I_L)  # 2维
    block3 = np.minimum(ineq_term, mu_I_hat)  # 2维
    
    # 4. 拼接所有块，计算L2范数
    psi_vector = np.concatenate([block1, block2, block3])  # 3+2+2=7维
    psi_val = np.linalg.norm(psi_vector, 2)
    
    return psi_val

def inner_loop_G(x, mu, d, Dmu, F_g, alpha_min, eta, gamma, r, u_g, sigma_L, mu_L):
    """
    内循环 G：回溯线搜索（维度自适应）
    """
    alpha = 1.0
    while alpha >= alpha_min:
        x_hat = x + alpha * d
        mu_hat = mu + alpha * Dmu
        p = compute_stabilized_constraint_violation(x_hat, mu_hat, sigma_L, mu_L)
        phi = compute_merit_function(x_hat, mu_hat, sigma_L, mu_L)
        if is_acceptable_to_global_filter(p, phi, F_g, eta, gamma, u_g) and p <= u_g:
            return x_hat, mu_hat, alpha
        alpha *= r  # 回溯因子
    return x, mu, 0.0  # 未找到，返回 0 表示触发 FRP

def feasibility_restoration_phase(x, mu, F_g, sigma_L, mu_L, u_g, eta, gamma, max_frp_iter=100):
    """
    可行性恢复阶段（FRP）：修复维度，适配3维x+4维mu
    """
    # 1. 动态获取维度
    n = len(x)  # 3
    m_E = len(compute_eq_constraints(x))  # 2
    m_I = len(compute_ineq_constraints(x))# 2
    total_vars = n + m_E + m_I  # 3+2+2=7
    
    # 2. 定义FRP子问题的目标函数
    def frp_objective(z):
        """
        z = [x1,x2,x3, mu_E1,mu_E2, mu_I1,mu_I2]（7维）
        """
        x_frp = z[:n]   
        mu_frp = z[n:]  
        
        # 核心目标：最小化稳定化约束违反度p(x,μ)
        p = compute_stabilized_constraint_violation(x_frp, mu_frp, sigma_L, mu_L)
        
        # 加小惩罚项：防止μ的绝对值过大
        mu_reg = 1e-4 * np.sum(np.square(mu_frp))
        
        return p + mu_reg

    # 3. 定义FRP子问题的约束（μ_I ≥ 0）
    frp_constraints = []
    for i in range(m_I):
        def frp_ineq_constraint(z, i=i):
            mu_frp = z[n:]
            mu_I_frp = mu_frp[m_E:]  # 不等式乘子
            return mu_I_frp[i]
        frp_constraints.append({'type': 'ineq', 'fun': frp_ineq_constraint})

    # 4. 迭代求解FRP子问题
    for frp_iter in range(max_frp_iter):
        # 初始猜测：基于当前点微调
        z0 = np.concatenate([x, mu]) + np.random.normal(0, 0.1, total_vars)
        
        # 调用SLSQP求解
        res = minimize(
            fun=frp_objective,
            x0=z0,
            constraints=frp_constraints,
            method='SLSQP',
            options={'maxiter': 100, 'ftol': 1e-6}
        )

        # 提取结果并验证
        if res.success:
            x_new = res.x[:n]
            mu_new = res.x[n:]
            
            p_new = compute_stabilized_constraint_violation(x_new, mu_new, sigma_L, mu_L)
            phi_new = compute_merit_function(x_new, mu_new, sigma_L, mu_L)

            if is_acceptable_to_global_filter(p_new, phi_new, F_g, eta, gamma, u_g) and p_new <= u_g:
                print(f"FRP成功：迭代{frp_iter+1}次，找到可接受点 p={p_new:.6f}")
                return x_new, mu_new
        
        # 降低惩罚系数重试
        frp_objective.rho = getattr(frp_objective, 'rho', 1.0) * 0.5

    # 兜底：返回当前点+小扰动
    print("FRP多次重试失败，返回当前点+小扰动")
    x_new = x + np.random.normal(0, 0.01, n)
    mu_new = np.maximum(mu + np.random.normal(0, 0.01, m_E + m_I), 0)  # 保证μ_I ≥ 0
    return x_new, mu_new

def compute_lagrangian_gradient(x, mu):
    f_grad = compute_f_gradient(x)
    c_E = compute_eq_constraints(x)
    c_I = compute_ineq_constraints(x)
    c_E_jac = compute_eq_jacobian(x)
    c_I_jac = compute_ineq_jacobian(x)
    m_E = len(c_E)
    mu_E = mu[:m_E]
    mu_I = mu[m_E:]
    return f_grad + c_E_jac.T @ mu_E + c_I_jac.T @ mu_I

def update_BFGS(B, s, y):
    # BFGS 拟牛顿更新（维度自适应）
    if np.dot(s, y) > 1e-8:  # 曲率条件（加小阈值防除0）
        
        Bs = B @ s
        B = B - np.outer(Bs, Bs) / (s @ Bs) + np.outer(y, y) / (y @ s)
    return B

# ===================== 测试代码（3维+4乘子） =====================
if __name__ == "__main__":
    # 初始点（二维）
    x0 = np.array([2.0, 2.0])
    # 初始乘子（1个等式 + 1个不等式，共2个）
    mu0 = np.array([0.0, 0.0])
    
    # 运行SFSQP算法
    x_opt, mu_opt = sfsqp_algorithm(x0, mu0, max_iter=100)
    
    # 输出结果
    print("="*50)
    print(f"最优解 x* = {np.round(x_opt, 4)}")
    print(f"最优乘子 μ* = {np.round(mu_opt, 4)}")
    print(f"目标函数值 f(x*) = {compute_objective(x_opt):.6f}")
    print(f"等式约束 c_E(x*) = {np.round(compute_eq_constraints(x_opt), 4)}")
    print(f"不等式约束 c_I(x*) = {np.round(compute_ineq_constraints(x_opt), 4)}")
    print(f"KKT误差 = {compute_KKT_error(x_opt, mu_opt):.6f}")
    print(f"约束违反度 h(x*) = {compute_constraint_violation(x_opt):.6f}")
