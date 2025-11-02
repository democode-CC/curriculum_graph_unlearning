#!/bin/bash
# Git 仓库快速设置脚本 / Quick Git Repository Setup Script

echo "========================================"
echo "Git Repository Setup for Graph Unlearning"
echo "========================================"
echo ""

# Check if already a git repository
if [ -d ".git" ]; then
    echo "⚠️  Git repository already exists!"
    echo "已存在 Git 仓库！"
    read -p "Do you want to continue? (y/n) 是否继续？" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "Initializing Git repository..."
    echo "正在初始化 Git 仓库..."
    git init
    echo "✓ Repository initialized / 仓库已初始化"
    echo ""
fi

# Check git config
echo "Checking Git configuration..."
echo "检查 Git 配置..."
USER_NAME=$(git config --global user.name)
USER_EMAIL=$(git config --global user.email)

if [ -z "$USER_NAME" ] || [ -z "$USER_EMAIL" ]; then
    echo ""
    echo "⚠️  Git user information not configured"
    echo "Git 用户信息未配置"
    echo ""
    read -p "Enter your name / 输入你的名字: " name
    read -p "Enter your email / 输入你的邮箱: " email
    
    git config --global user.name "$name"
    git config --global user.email "$email"
    echo "✓ User information configured / 用户信息已配置"
else
    echo "✓ Git configured as: $USER_NAME <$USER_EMAIL>"
    echo "✓ Git 已配置为: $USER_NAME <$USER_EMAIL>"
fi
echo ""

# Add files
echo "Adding files to staging area..."
echo "添加文件到暂存区..."
git add .
echo "✓ Files added / 文件已添加"
echo ""

# Show status
echo "Current status / 当前状态:"
git status --short
echo ""

# Create initial commit
echo "Creating initial commit..."
echo "创建初始提交..."
git commit -m "Initial commit: Graph unlearning project implementation

- Implemented Curriculum Unlearning method
- Implemented Negative Preference Optimization (NPO)
- Added baseline methods (Retrain, Gradient Ascent)
- Included 5 datasets (Cora, CiteSeer, PubMed, FB15k237, WN18RR)
- Implemented 5 GNN models (GCN, GAT, GraphSAGE, RGCN, CompGCN)
- Added comprehensive evaluation scripts
- Created conda environment setup
- Added documentation (README, SETUP, GIT_SETUP)"

echo "✓ Initial commit created / 初始提交已创建"
echo ""

# Create main branch
git branch -M main
echo "✓ Main branch created / 主分支已创建"
echo ""

# Ask about remote repository
echo "========================================"
echo "Remote Repository Setup / 远程仓库设置"
echo "========================================"
echo ""
read -p "Do you want to connect to a remote repository? (y/n) 是否连接到远程仓库？" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Please create a repository on GitHub first if you haven't."
    echo "如果还没有，请先在 GitHub 上创建仓库。"
    echo ""
    read -p "Enter remote repository URL / 输入远程仓库 URL: " remote_url
    
    if [ ! -z "$remote_url" ]; then
        git remote add origin "$remote_url"
        echo "✓ Remote repository added / 远程仓库已添加"
        echo ""
        
        read -p "Push to remote now? (y/n) 现在推送到远程？" -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git push -u origin main
            echo "✓ Pushed to remote / 已推送到远程"
        else
            echo "You can push later with: git push -u origin main"
            echo "稍后可以使用以下命令推送: git push -u origin main"
        fi
    fi
fi

echo ""
echo "========================================"
echo "✓ Git setup complete! / Git 设置完成！"
echo "========================================"
echo ""
echo "Useful commands / 常用命令:"
echo "  git status          - Check status / 查看状态"
echo "  git log --oneline   - View history / 查看历史"
echo "  git add .           - Stage changes / 暂存修改"
echo "  git commit -m 'msg' - Commit / 提交"
echo "  git push            - Push to remote / 推送到远程"
echo ""
echo "See GIT_SETUP.md for more information."
echo "查看 GIT_SETUP.md 获取更多信息。"
echo ""



