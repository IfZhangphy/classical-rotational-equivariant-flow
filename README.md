# 旋转等变流模型

`rotational_equivariant_flow.py` 是当前目录中旋转等变版本的可执行实现，`Rotational_Equivariance.ipynb` 只负责交互式调用它。

## AutoDL 运行

上传 `autodl_rotational_equivariant_flow.tar.gz` 后，在项目目录执行：

```bash
conda create -n rotflow python=3.11 -y
conda activate rotflow
bash setup_autodl.sh
source .venv-autodl/bin/activate
```

如果镜像中 `conda` 不在 PATH，可以先执行 `which conda`；AutoDL 常见路径是 `/root/miniconda3/bin/conda`。当前 Python 3.8 无法安装 CUDA 版 JAX 0.5.3 及以上版本。

安装脚本会安装 CUDA 12 版 JAX，并检查 GPU。确认看到 `CudaDevice` 后先运行：

```bash
bash run_smoke_autodl.sh
```

开始训练：

```bash
python rotational_equivariant_flow.py \
  --epochs 2000 \
  --batch-size 128 \
  --hidden-dim 48 \
  --layers 4 \
  --output-dir rotational_equivariant_results
```

`training.png` 现在同时保存二维位置密度、径向密度 `P(r)` 和自由能曲线；其中 `P(r)` 的峰值是判断壳层结构的主要指标。

当前默认使用 `distance_floor=0.05` 作为训练初期的软核正则，避免少量粒子重合样本产生极端梯度。若要做无软核的最终物理对照，可显式加 `--distance-floor 0.001`，但建议先完成稳定训练。

训练过程中可以在另一个终端查看显存：

```bash
watch -n 1 nvidia-smi
```

如果显存不足，依次尝试 `--batch-size 64` 和 `--hidden-dim 32`。

## 本地运行

先做结构检查：

```powershell
python rotational_equivariant_flow.py --smoke-test
```

开始低显存训练：

```powershell
python rotational_equivariant_flow.py \
  --epochs 2000 --batch-size 256 --hidden-dim 48 --layers 4
```

如果显存不足，依次尝试 `--batch-size 128`、`--batch-size 64`，再把 `--hidden-dim` 调到 32。训练结果会写入 `rotational_equivariant_results/`，包括参数检查点和训练曲线。

## 模型结构

每个耦合层把粒子分成条件集和目标集。条件网络的标量特征只使用距离平方和半径平方；向量平移由条件粒子坐标的加权和生成。这样对任意二维旋转矩阵 `R` 都满足：

```text
flow(R z) = R flow(z)
```

由于缩放和平移只依赖条件粒子，目标粒子的变换仍然是解析可逆的，Jacobian 对数行列式也可以直接计算。

## 本机验证

已在当前环境完成：

- 单精度、双精度冒烟测试；
- 4 层、48 隐藏维、双消息传递、batch 256 的训练循环；
- 逆变换最大误差约 `5e-7`（单精度）；
- 旋转等变最大误差约 `5e-7`（单精度）；
- `py_compile` 和 notebook JSON 校验。

代码已经关闭 JAX 的全量显存预分配，并通过 `requirements-autodl.txt` 使用 CUDA 12 版 JAX。当前 Windows `.conda` 环境使用 CPU 版 JAX；AutoDL 应使用脚本创建的 Linux 环境。

如果 GPU 冒烟测试的等变误差异常偏大，请确认使用的是最新版脚本。脚本会固定 GPU 矩阵乘法为 float32，避免 TF32 舍入误差影响等变性检查。
