# Git 仓库设置指南 / Git Repository Setup Guide

[English](#english) | [中文](#中文)

---

## 中文

### 快速开始

```bash
# 进入项目目录
cd /Users/zhangchenhan/Github_Repo_Local/curriculum_graph_unlearning

# 初始化 Git 仓库
git init

# 添加所有文件
git add .

# 查看状态
git status

# 创建第一次提交
git commit -m "Initial commit: Graph unlearning project implementation"
```

### 详细步骤

#### 1. 初始化本地仓库

```bash
cd /Users/zhangchenhan/Github_Repo_Local/curriculum_graph_unlearning
git init
```

这会创建一个 `.git` 目录，将当前目录变成一个 Git 仓库。

#### 2. 配置用户信息（如果还没配置）

```bash
# 设置用户名
git config --global user.name "你的名字"

# 设置邮箱
git config --global user.email "your.email@example.com"

# 查看配置
git config --list
```

#### 3. 添加文件到暂存区

```bash
# 添加所有文件
git add .

# 或者选择性添加
git add README.md
git add *.py
```

#### 4. 创建第一次提交

```bash
git commit -m "Initial commit: Graph unlearning project

- Implemented Curriculum Unlearning method
- Implemented Negative Preference Optimization (NPO)
- Added baseline methods (Retrain, Gradient Ascent)
- Included 5 datasets (Cora, CiteSeer, PubMed, FB15k237, WN18RR)
- Implemented 5 GNN models (GCN, GAT, GraphSAGE, RGCN, CompGCN)
- Added comprehensive evaluation scripts
- Created conda environment setup"
```

#### 5. 连接到远程仓库（GitHub/GitLab）

**在 GitHub 上创建仓库后：**

```bash
# 添加远程仓库
git remote add origin https://github.com/你的用户名/curriculum_graph_unlearning.git

# 或使用 SSH
git remote add origin git@github.com:你的用户名/curriculum_graph_unlearning.git

# 查看远程仓库
git remote -v

# 推送到远程仓库
git branch -M main
git push -u origin main
```

### 常用 Git 命令

```bash
# 查看状态
git status

# 查看提交历史
git log
git log --oneline --graph

# 查看差异
git diff

# 创建新分支
git branch feature-name
git checkout feature-name
# 或
git checkout -b feature-name

# 切换分支
git checkout main

# 合并分支
git merge feature-name

# 拉取更新
git pull origin main

# 推送更新
git push origin main

# 撤销修改
git checkout -- filename
git restore filename

# 取消暂存
git reset HEAD filename
git restore --staged filename

# 查看远程仓库
git remote -v
```

### .gitignore 文件

项目已经包含了 `.gitignore` 文件，会自动忽略：
- Python 缓存文件 (`__pycache__`, `*.pyc`)
- 训练好的模型文件 (`*.pt`, `*.pth`)
- 数据文件
- 结果文件
- IDE 配置文件

### 推荐的工作流程

```bash
# 1. 开始新功能
git checkout -b feature/new-experiment

# 2. 进行修改和测试
# ... 编辑文件 ...

# 3. 查看修改
git status
git diff

# 4. 提交修改
git add .
git commit -m "Add new experiment configuration"

# 5. 切换回主分支
git checkout main

# 6. 合并新功能
git merge feature/new-experiment

# 7. 推送到远程
git push origin main
```

### 创建 GitHub 仓库的步骤

1. 访问 https://github.com
2. 点击右上角的 "+" -> "New repository"
3. 填写仓库信息：
   - Repository name: `curriculum_graph_unlearning`
   - Description: "Mitigating Catastrophic Collapse in Large-Scale Graph Unlearning"
   - 选择 Public 或 Private
   - **不要**勾选 "Initialize this repository with a README"（因为我们已经有了）
4. 点击 "Create repository"
5. 按照页面上的指示连接本地仓库

### 示例：完整设置流程

```bash
# 1. 初始化
cd /Users/zhangchenhan/Github_Repo_Local/curriculum_graph_unlearning
git init

# 2. 添加文件
git add .

# 3. 首次提交
git commit -m "Initial commit: Graph unlearning implementation"

# 4. 创建 main 分支
git branch -M main

# 5. 连接到 GitHub（替换为你的用户名）
git remote add origin git@github.com:你的用户名/curriculum_graph_unlearning.git

# 6. 推送到 GitHub
git push -u origin main
```

---

## English

### Quick Start

```bash
# Navigate to project directory
cd /Users/zhangchenhan/Github_Repo_Local/curriculum_graph_unlearning

# Initialize Git repository
git init

# Add all files
git add .

# Check status
git status

# Create first commit
git commit -m "Initial commit: Graph unlearning project implementation"
```

### Detailed Steps

#### 1. Initialize Local Repository

```bash
cd /Users/zhangchenhan/Github_Repo_Local/curriculum_graph_unlearning
git init
```

This creates a `.git` directory and turns the current directory into a Git repository.

#### 2. Configure User Information (if not already configured)

```bash
# Set username
git config --global user.name "Your Name"

# Set email
git config --global user.email "your.email@example.com"

# View configuration
git config --list
```

#### 3. Add Files to Staging Area

```bash
# Add all files
git add .

# Or selectively add
git add README.md
git add *.py
```

#### 4. Create First Commit

```bash
git commit -m "Initial commit: Graph unlearning project

- Implemented Curriculum Unlearning method
- Implemented Negative Preference Optimization (NPO)
- Added baseline methods (Retrain, Gradient Ascent)
- Included 5 datasets (Cora, CiteSeer, PubMed, FB15k237, WN18RR)
- Implemented 5 GNN models (GCN, GAT, GraphSAGE, RGCN, CompGCN)
- Added comprehensive evaluation scripts
- Created conda environment setup"
```

#### 5. Connect to Remote Repository (GitHub/GitLab)

**After creating repository on GitHub:**

```bash
# Add remote repository
git remote add origin https://github.com/yourusername/curriculum_graph_unlearning.git

# Or use SSH
git remote add origin git@github.com:yourusername/curriculum_graph_unlearning.git

# View remote repositories
git remote -v

# Push to remote repository
git branch -M main
git push -u origin main
```

### Common Git Commands

```bash
# View status
git status

# View commit history
git log
git log --oneline --graph

# View differences
git diff

# Create new branch
git branch feature-name
git checkout feature-name
# Or
git checkout -b feature-name

# Switch branches
git checkout main

# Merge branches
git merge feature-name

# Pull updates
git pull origin main

# Push updates
git push origin main

# Undo changes
git checkout -- filename
git restore filename

# Unstage files
git reset HEAD filename
git restore --staged filename

# View remote repositories
git remote -v
```

### .gitignore File

The project already includes a `.gitignore` file that automatically ignores:
- Python cache files (`__pycache__`, `*.pyc`)
- Trained model files (`*.pt`, `*.pth`)
- Data files
- Result files
- IDE configuration files

### Recommended Workflow

```bash
# 1. Start new feature
git checkout -b feature/new-experiment

# 2. Make changes and test
# ... edit files ...

# 3. View changes
git status
git diff

# 4. Commit changes
git add .
git commit -m "Add new experiment configuration"

# 5. Switch back to main branch
git checkout main

# 6. Merge new feature
git merge feature/new-experiment

# 7. Push to remote
git push origin main
```

### Steps to Create GitHub Repository

1. Visit https://github.com
2. Click "+" in top right -> "New repository"
3. Fill in repository information:
   - Repository name: `curriculum_graph_unlearning`
   - Description: "Mitigating Catastrophic Collapse in Large-Scale Graph Unlearning"
   - Choose Public or Private
   - **Do not** check "Initialize this repository with a README" (we already have one)
4. Click "Create repository"
5. Follow the instructions on the page to connect your local repository

### Example: Complete Setup Process

```bash
# 1. Initialize
cd /Users/zhangchenhan/Github_Repo_Local/curriculum_graph_unlearning
git init

# 2. Add files
git add .

# 3. First commit
git commit -m "Initial commit: Graph unlearning implementation"

# 4. Create main branch
git branch -M main

# 5. Connect to GitHub (replace with your username)
git remote add origin git@github.com:yourusername/curriculum_graph_unlearning.git

# 6. Push to GitHub
git push -u origin main
```

---

## Troubleshooting / 故障排除

### Problem: "Permission denied (publickey)"

**Solution / 解决方案:**
```bash
# Generate SSH key / 生成 SSH 密钥
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to ssh-agent / 添加到 ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key / 复制公钥
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings -> SSH and GPG keys -> New SSH key
# 添加到 GitHub: 设置 -> SSH 和 GPG 密钥 -> 新建 SSH 密钥
```

### Problem: Large files / 大文件问题

**Solution / 解决方案:**
```bash
# Use Git LFS for large files / 使用 Git LFS 管理大文件
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

---

## Next Steps / 下一步

After setting up Git / 设置 Git 后:

1. ✓ Repository initialized / 仓库已初始化
2. → Make regular commits / 定期提交代码
3. → Create branches for experiments / 为实验创建分支
4. → Push to GitHub for backup / 推送到 GitHub 备份
5. → Collaborate with others / 与他人协作

---

**Documentation created for curriculum_graph_unlearning project**



