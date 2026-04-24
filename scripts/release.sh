#!/usr/bin/env bash
# cnvturbo 发版脚本
#
# 用法：
#   scripts/release.sh 0.1.0              # tag 之后由 GitHub Action 自动发布
#   scripts/release.sh 0.1.0 --testpypi   # 先打 tag 并发布到 TestPyPI（手动 dispatch）
#   scripts/release.sh 0.1.0 --local-only # 只本地构建并 twine check，不打 tag
#
# 先决条件：
#   * 已配置 PyPI Trusted Publisher（见 .github/workflows/release.yaml 顶部注释）
#   * 安装 gh CLI 并已 `gh auth login`
#   * 工作目录干净，已切到 main，且 origin 是 GitHub 远程
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

err() { printf '\033[31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }
log() { printf '\033[36m[release]\033[0m %s\n' "$*"; }

# 探测 Python 解释器：优先环境变量 PY，否则 python3 → python
# 如想指定 conda env：PY=/home/dell/miniforge3/envs/myenv/bin/python scripts/release.sh ...
if [[ -n "${PY:-}" ]]; then
  command -v "$PY" >/dev/null || err "PY=$PY 不可执行"
elif command -v python3 >/dev/null; then
  PY=python3
elif command -v python >/dev/null; then
  PY=python
else
  err "未找到 python3/python，请先安装 Python 3.10+ 或设置 PY 环境变量"
fi

VERSION="${1:-}"
MODE="${2:-pypi}"   # pypi | --testpypi | --local-only

[[ -n "$VERSION" ]] || err "缺少版本号。用法: $0 <X.Y.Z> [--testpypi|--local-only]"

# 校验 SemVer（允许 X.Y.Z 与 X.Y.ZrcN / X.Y.Z.devN）
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-]?(rc|a|b|dev)[0-9]+)?$ ]]; then
  err "版本号格式不对：$VERSION（应类似 0.1.0、0.1.0rc1、0.1.0.dev1）"
fi
TAG="v${VERSION}"

log "版本: $VERSION   tag: $TAG   模式: $MODE"

# ── 1. 工作树检查 ─────────────────────────────────────────────────────────
if [[ -n "$(git status --porcelain)" ]]; then
  err "工作树不干净，先 commit 或 stash 再发版"
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "main" ]]; then
  read -rp "当前在 '$CURRENT_BRANCH' 分支，确认从此分支发版? [y/N] " ans
  [[ "$ans" =~ ^[Yy]$ ]] || err "取消"
fi

if git rev-parse "$TAG" >/dev/null 2>&1; then
  err "tag $TAG 已存在，请用更高版本号或先删除 tag"
fi

# ── 2. 拉最新 ─────────────────────────────────────────────────────────────
log "git fetch & pull..."
git fetch --tags origin
git pull --ff-only origin "$CURRENT_BRANCH"

# ── 3. 本地构建 + 校验 ────────────────────────────────────────────────────
log "清理 dist/，开始本地构建（解释器: $PY）..."
rm -rf dist build ./*.egg-info
"$PY" -m pip install --quiet --upgrade pip build twine
"$PY" -m build
"$PY" -m twine check --strict dist/*
log "本地构建通过：$(ls dist/)"

if [[ "$MODE" == "--local-only" ]]; then
  log "已完成本地构建（--local-only），未打 tag。"
  exit 0
fi

# ── 4. 打 tag 并推送 ─────────────────────────────────────────────────────
read -rp "确认打 tag $TAG 并 push 到 origin? [y/N] " ans
[[ "$ans" =~ ^[Yy]$ ]] || err "取消"

git tag -a "$TAG" -m "Release $TAG"
git push origin "$TAG"
log "已推送 tag $TAG"

# ── 5. 触发发布 ──────────────────────────────────────────────────────────
if [[ "$MODE" == "--testpypi" ]]; then
  log "触发 TestPyPI workflow（需 gh CLI）..."
  if ! command -v gh >/dev/null; then
    err "未找到 gh CLI。请安装后重试，或手动到 GitHub Actions 页面 Run workflow → testpypi"
  fi
  gh workflow run release.yaml -f target=testpypi --ref "$TAG"
  log "已触发，去 GitHub Actions 页面跟进 → https://test.pypi.org/p/cnvturbo"
else
  log "创建 GitHub Release（这会触发 PyPI 发布）..."
  if ! command -v gh >/dev/null; then
    cat <<EOF
未安装 gh CLI。手动操作：
  1. 打开 https://github.com/LogicByteCraft/cnvturbo/releases/new?tag=$TAG
  2. 填写 Release notes，点 "Publish release"
  3. .github/workflows/release.yaml 会自动发布到 PyPI（OIDC）
EOF
    exit 0
  fi
  gh release create "$TAG" \
    --title "$TAG" \
    --generate-notes \
    --verify-tag
  log "Release 已创建，PyPI 发布已触发 → https://pypi.org/p/cnvturbo"
fi
