# Hexo Blog

基于 Hexo + Butterfly 主题 + GitHub Pages 的个人博客。

## 快速开始

```bash
# 安装依赖
npm install

# 本地预览
npx hexo server

# 构建
npx hexo generate
```

## 部署

Push 到 main 分支后 GitHub Actions 自动构建部署。

确保 GitHub 仓库 Settings → Pages → Source 选择 **GitHub Actions**。

## 写新文章

```bash
npx hexo new "文章标题"
# 编辑 source/_posts/文章标题.md
# git add, commit, push
```
