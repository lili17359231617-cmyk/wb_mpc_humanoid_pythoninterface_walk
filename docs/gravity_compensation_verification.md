# 重力补偿索引验证指南

## 问题背景

当前代码使用 `tau_gravity = data.qfrc_inverse[6:].copy()` 来获取重力补偿力矩。
需要验证这个索引是否正确映射到对应的关节。

## MuJoCo qfrc_inverse 结构

### 基本结构

`data.qfrc_inverse` 是 MuJoCo 逆动力学计算的结果，其结构如下：

```
qfrc_inverse = [
    f_x_base,      # [0] 基座 x 方向力
    f_y_base,      # [1] 基座 y 方向力
    f_z_base,      # [2] 基座 z 方向力
    tau_x_base,    # [3] 基座 x 方向力矩
    tau_y_base,    # [4] 基座 y 方向力矩
    tau_z_base,    # [5] 基座 z 方向力矩
    tau_joint_0,   # [6] 第0个关节力矩
    tau_joint_1,   # [7] 第1个关节力矩
    ...
    tau_joint_n    # [6+n] 第n个关节力矩
]
```

**关键点**：
- `qfrc_inverse` 的索引**直接对应** `qpos` 数组的索引
- 前6个元素（索引0-5）是基座的6DOF
- 从索引6开始是关节力矩，顺序与 `qpos` 中关节的顺序一致

### 索引映射关系

对于关节 `i`，其在 `qpos` 中的索引是 `qpos_idx = mpc_qpos_idxs[i]`，那么：
- 该关节的重力补偿力矩 = `qfrc_inverse[qpos_idx]`

## 验证方法

### 方法1: 代码逻辑验证

检查 `qfrc_inverse[6:]` 是否按 `qpos` 顺序排列：

```python
# 假设 qpos 中关节的顺序是：基座(0-6) + 关节0, 关节1, ..., 关节n
# 那么 qfrc_inverse[6] 应该对应 qpos 中第7个元素（第一个关节）

# 验证：qfrc_inverse[6:] 是否等于按 qpos 顺序提取的值
for i in range(len(mpc_qpos_idxs)):
    qpos_idx = mpc_qpos_idxs[i]  # 关节在 qpos 中的索引
    qfrc_idx = 6 + i              # 当前代码使用的索引

    # 正确的方法
    correct_value = data.qfrc_inverse[qpos_idx]

    # 当前代码的方法
    current_value = data.qfrc_inverse[6 + i]

    # 检查是否匹配
    if abs(correct_value - current_value) > 1e-6:
        print(f"不匹配！关节 {i}: qpos_idx={qpos_idx}, 正确值={correct_value}, 当前值={current_value}")
```

### 方法2: 手动检查

1. **检查 qpos 结构**：
   ```python
   print("qpos 结构:")
   print(f"  [0:3] 基座位置: {data.qpos[0:3]}")
   print(f"  [3:7] 基座姿态(四元数): {data.qpos[3:7]}")
   for i, idx in enumerate(mpc_qpos_idxs):
       print(f"  [{idx}] {MPC_JOINT_NAMES[i]}: {data.qpos[idx]}")
   ```

2. **检查 qfrc_inverse 结构**：
   ```python
   print("qfrc_inverse 结构:")
   print(f"  [0:3] 基座力: {data.qfrc_inverse[0:3]}")
   print(f"  [3:6] 基座力矩: {data.qfrc_inverse[3:6]}")
   for i, idx in enumerate(mpc_qpos_idxs):
       print(f"  [{idx}] {MPC_JOINT_NAMES[i]}: {data.qfrc_inverse[idx]}")
   ```

3. **对比验证**：
   ```python
   print("验证 qfrc_inverse[6:] 映射:")
   for i in range(len(mpc_qpos_idxs)):
       qpos_idx = mpc_qpos_idxs[i]
       qfrc_value_by_qpos = data.qfrc_inverse[qpos_idx]
       qfrc_value_by_slice = data.qfrc_inverse[6 + i]

       print(f"关节 {i} ({MPC_JOINT_NAMES[i]}):")
       print(f"  qpos索引: {qpos_idx}")
       print(f"  按qpos索引取值: {qfrc_value_by_qpos}")
       print(f"  按[6+i]取值: {qfrc_value_by_slice}")
       print(f"  匹配: {abs(qfrc_value_by_qpos - qfrc_value_by_slice) < 1e-6}")
   ```

### 方法3: 物理验证

如果索引映射错误，会出现明显的物理异常：

1. **错误的索引映射会导致**：
   - 某些关节的重力补偿力矩应用到错误的关节
   - 机器人无法保持平衡
   - 某些关节（如脚踝）抖动明显

2. **正确的索引映射应该**：
   - 机器人能够稳定站立
   - 重力补偿力矩合理（通常为几到几十 N·m）
   - 各关节力矩方向正确（与重力方向一致）

## 当前代码分析

### 当前实现

```python
# 重力补偿计算
data.qacc[:] = 0
data.xfrc_applied[:] = 0
mujoco.mj_inverse(model, data)
tau_gravity = data.qfrc_inverse[6:].copy()

# 应用控制
for i, act_id in enumerate(mpc_ctrl_idxs):
    torque = tau_gravity[i] + tau_pd[i]
    data.ctrl[act_id] = torque
```

### 潜在问题

**假设**：`qfrc_inverse[6:]` 按照 `qpos` 中关节的顺序排列。

**实际情况**：
- 如果 `qpos` 的结构是：`[基座位置(3), 基座姿态(4), 关节0, 关节1, ..., 关节n]`
- 那么 `qfrc_inverse[6]` 对应 `qpos[6]`，即第一个关节
- 但是 `mpc_qpos_idxs[0]` 可能不等于 6！

**示例**：
假设 `mpc_qpos_idxs[0] = 7`（第一个关节在 qpos 中的索引是7），那么：
- 当前代码：`tau_gravity[0] = qfrc_inverse[6]` ❌ 错误！
- 正确方法：`tau_gravity[0] = qfrc_inverse[7]` ✅ 正确！

## 正确的实现方法

### 方法1: 使用 qpos 索引（推荐）

```python
# 计算重力补偿
mujoco.mj_inverse(model, data)

# 按 qpos 索引提取重力补偿力矩
tau_gravity = np.zeros(MPC_JOINT_DIM)
for i in range(MPC_JOINT_DIM):
    qpos_idx = mpc_qpos_idxs[i]
    tau_gravity[i] = data.qfrc_inverse[qpos_idx]
```

### 方法2: 验证 [6:] 切片是否正确

```python
# 计算重力补偿
mujoco.mj_inverse(model, data)

# 验证 qfrc_inverse[6:] 是否按 qpos 顺序排列
tau_gravity_slice = data.qfrc_inverse[6:6+MPC_JOINT_DIM]
tau_gravity_correct = np.array([data.qfrc_inverse[mpc_qpos_idxs[i]] for i in range(MPC_JOINT_DIM)])

if np.allclose(tau_gravity_slice, tau_gravity_correct):
    tau_gravity = tau_gravity_slice.copy()  # 可以使用切片方法
else:
    tau_gravity = tau_gravity_correct  # 必须使用索引方法
```

## 验证脚本使用说明

由于 OpenGL 依赖问题，验证脚本可能无法在 headless 环境中运行。

### 替代验证方法

1. **在有图形界面的环境中运行**：
   ```bash
   python3 humanoid_nmpc/humanoid_wb_mpc/scripts/verify_gravity_compensation.py
   ```

2. **在 test_mpc_rl.py 中添加验证代码**：
   ```python
   # 在重力补偿计算后添加
   if frame_count == 0:  # 只在第一帧验证
       print("\n[验证] 重力补偿索引检查:")
       for i in range(len(mpc_qpos_idxs)):
           qpos_idx = mpc_qpos_idxs[i]
           correct_value = data.qfrc_inverse[qpos_idx]
           current_value = tau_gravity[i]
           match = "✓" if abs(correct_value - current_value) < 1e-6 else "✗"
           print(f"  关节 {i} ({MPC_JOINT_NAMES[i]}): qpos_idx={qpos_idx}, "
                 f"正确值={correct_value:.4f}, 当前值={current_value:.4f} {match}")
   ```

## 总结

### 验证步骤

1. ✅ **检查 qfrc_inverse 长度**：应该等于 `model.nv`（速度空间维度）
2. ✅ **检查 qfrc_inverse[6:] 长度**：应该等于关节数
3. ✅ **验证索引映射**：`qfrc_inverse[mpc_qpos_idxs[i]]` 是否等于 `qfrc_inverse[6+i]`
4. ✅ **物理验证**：观察机器人是否能够稳定站立

### 如果发现索引不匹配

使用正确的实现方法：
```python
tau_gravity = np.array([data.qfrc_inverse[mpc_qpos_idxs[i]] for i in range(MPC_JOINT_DIM)])
```

这样可以确保重力补偿力矩正确映射到对应的关节。
