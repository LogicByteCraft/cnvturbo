# Publishing `cnvturbo` to PyPI

本项目使用 **PyPI Trusted Publishing (OIDC)**，无需在 GitHub 仓库里存任何 `PYPI_TOKEN`。
全流程：本地打 tag → GitHub Release → Action 自动构建并发布。

---

## 一次性配置（仅首次发布前需要做一次）

### 1. 在 PyPI 添加 Trusted Publisher

未注册过 `cnvturbo`：

1. 登录 <https://pypi.org/manage/account/publishing/>。
2. 点 **Add a new pending publisher**，填：
   - PyPI Project Name: `cnvturbo`
   - Owner: `LogicByteCraft`
   - Repository name: `cnvturbo`
   - Workflow filename: `release.yaml`
   - Environment name: `pypi`
3. 提交。首次成功发布后，"pending" 自动转为正式 publisher。

> 同样到 <https://test.pypi.org/manage/account/publishing/> 配置一个 environment=`testpypi` 的 pending publisher，用于先发到 TestPyPI 验证。

### 2. 在 GitHub 创建 environment

仓库 → **Settings → Environments → New environment**：

- 创建 `pypi`（可选：要求人工 reviewer 才能跑）
- 创建 `testpypi`

### 3. 安装本地工具（可选，但推荐）

```bash
pip install build twine
brew install gh   # 或 sudo apt install gh
gh auth login     # 一次性登录 GitHub
```

---

## 日常发版流程

### 方式 A：用脚本（最省事）

```bash
cd cnvturbo
scripts/release.sh 0.1.0              # 直接发到 PyPI
scripts/release.sh 0.1.0 --testpypi   # 先发到 TestPyPI 验证
scripts/release.sh 0.1.0 --local-only # 只做本地构建 + twine check，不打 tag
```

脚本会做：

1. 校验工作树干净 + SemVer 版本号合法
2. `python -m build` + `twine check --strict dist/*`
3. 打 annotated tag `v0.1.0` 并 push
4. 通过 `gh` 创建 GitHub Release（或手动 dispatch TestPyPI workflow）
5. GitHub Action `release.yaml` 自动触发 → PyPI

### 方式 B：纯手动

```bash
# 1. 本地预览构建
python -m build
twine check --strict dist/*

# 2. 打 tag 并推送
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0

# 3. GitHub UI: Releases → Draft new release → Choose tag v0.1.0 → Publish
#    workflow 会自动跑 → 几分钟后 https://pypi.org/p/cnvturbo 出现新版本
```

---

## 验证发布

```bash
pip install --upgrade cnvturbo
python -c "import cnvturbo; print(cnvturbo.__version__)"
```

TestPyPI 验证：

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            cnvturbo
```

---

## 版本号规则

`cnvturbo` 用 `hatch-vcs` 自动从 git tag 派生版本号：

| Git tag       | `cnvturbo.__version__` |
| ------------- | ---------------------- |
| `v0.1.0`      | `0.1.0`                |
| `v0.1.0rc1`   | `0.1.0rc1`             |
| `v0.1.0.dev1` | `0.1.0.dev1`           |
| 无 tag        | 形如 `0.1.0.dev3+g…`   |

> **不要** 手动修改 `pyproject.toml` 里的 version；只通过 git tag 控制。

---

## 排错

- **PyPI 拒绝 "version already exists"**：版本号已发布过，PyPI 不允许覆盖。请新建一个更高版本号（哪怕 `0.1.0` → `0.1.0.post1`）。
- **OIDC 鉴权失败 "trusted publisher not found"**：检查 PyPI 上 publisher 的 owner/repo/workflow filename/environment 四个字段是否完全一致；workflow 文件名必须和 `.github/workflows/release.yaml` 文件名 **包含路径** 一致。
- **Action 跑过但 PyPI 没出现新版本**：通常是 `dist/` 文件名包含旧版本（构建缓存导致），脚本里已 `rm -rf dist build *.egg-info` 避免；如手动构建注意清理。
