# 修复编译权限问题

## 问题描述

编译时出现 `PermissionError: [Errno 13] Permission denied`，这是因为构建目录的权限不正确。

## 解决方法

### 方法一：清理并修复权限（推荐）

```bash
# 1. 清理构建目录
cd /wb_humanoid_mpc_ws
rm -rf build install log .ccache

# 2. 确保当前用户对工作空间有写权限
cd /wb_humanoid_mpc_ws
sudo chown -R $(id -u):$(id -g) .

# 3. 重新编译
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
make build-all
```

### 方法二：仅修复构建目录权限

```bash
# 修复构建目录权限
cd /wb_humanoid_mpc_ws
sudo chown -R $(id -u):$(id -g) build install log .ccache

# 确保脚本可执行
find build -name "*.sh" -exec chmod +x {} \;

# 重新编译
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
make build-all
```

### 方法三：检查并修复（如果方法一不行）

```bash
# 1. 检查当前用户ID
echo "User ID: $(id -u)"
echo "Group ID: $(id -g)"

# 2. 检查目录权限
ls -la /wb_humanoid_mpc_ws/build/catkin/devel/ 2>/dev/null || echo "Directory does not exist"

# 3. 清理并重新创建
cd /wb_humanoid_mpc_ws
rm -rf build install log .ccache
mkdir -p build install log .ccache
chmod -R u+w build install log .ccache

# 4. 重新编译
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
make build-all
```

## 预防措施

如果工作空间是从宿主机挂载的，确保在宿主机上设置正确的权限：

```bash
# 在宿主机上执行
sudo chown -R $USER:$USER /path/to/wb_humanoid_mpc_ws
```
