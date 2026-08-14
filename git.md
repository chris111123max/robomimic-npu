工作流 A：Codex 修改代码 → 服务器训练
1. Windows 开始修改前
cd E:\robosuit\robomimic-npu
git pull

然后用 Codex 打开：

E:\robosuit\robomimic-npu

让 Codex 修改代码。

2. Codex 修改完成后
cd E:\robosuit\robomimic-npu

git status
git add .
git commit -m "修改说明"
git push

例如：

git commit -m "Update BC Transformer training"

此时：

Codex
  ↓
Windows
  ↓ git push
GitHub
3. 服务器获取修改
cd /data/home/3220251075/lerobot_workspace/robomimic

git status
git pull

然后开始 NPU 训练或测试。

工作流 B：服务器修改代码 → Codex 获取修改

如果直接在 BIT 服务器上修改了代码：

1. 服务器开始修改前
cd /data/home/3220251075/lerobot_workspace/robomimic

git pull

然后修改代码。

2. 服务器修改完成后
git status
git add .
git commit -m "修改说明"
git push

例如：

git commit -m "Fix NPU training issue"

此时：

BIT服务器
   ↓ git push
GitHub
3. Windows / Codex 获取服务器修改

回到 Windows：

cd E:\robosuit\robomimic-npu

git pull

然后再用 Codex 打开项目。

此时 Codex 看到的就是服务器最新版本。