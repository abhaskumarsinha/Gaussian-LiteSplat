# Gaussian-LiteSplat
A minimal, portable Gaussian Splatting sandbox for rapid small-scale experimentation and performance insights in pure Python.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-Keras-red.svg)](https://keras.io/)
[![Backend](https://img.shields.io/badge/Backend-Torch-orange.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen.svg)](https://github.com/yourusername/gaussian-litesplat)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-yellow.svg)](https://github.com/yourusername/gaussian-litesplat/issues)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Pure%20Python-lightgrey.svg)]()

---

## 🚀 Overview
Gaussian **LiteSplat** is a **fully functional Gaussian Splatting framework** implemented entirely in **Keras** (with **Torch backend support**).  
It’s designed as a **lightweight, open-source sandbox** for researchers and developers to:
- Prototype fast, 
- Experiment freely, and  
- Understand performance behavior in a **clean, dependency-minimal** environment.


<!-- PROJECT RESULTS -->
<br />
<div align="center">

  <h2 align="center">Gaussian LiteSplat</h2>

  <p align="center">
    <em>A minimal, portable Gaussian Splatting sandbox for rapid small-scale experimentation and performance insights in pure Python.</em>
  </p>

  <p align="center">
    <strong>Result Showcase</strong><br />
    <em>2200 Gaussians · Trained 45 minutes · 15 Cameras · TempleRing dataset</em><br />
    <a href="https://vision.middlebury.edu/mview/data/">TempleRing.zip — Middlebury Multiview Dataset</a>
  </p>

  <p align="center">
    <img src="https://raw.githubusercontent.com/abhaskumarsinha/Gaussian-LiteSplat/refs/heads/main/G5Fk1gAa4AA9uhZ.jpg" alt="Gaussian LiteSplat Result" width="640" />
  </p>

  <p align="center">
    <a href="https://github.com/abhaskumarsinha/gaussian-litesplat"><strong>View on GitHub »</strong></a>
    &nbsp;·&nbsp;
    <a href="https://github.com/abhaskumarsinha/gaussian-litesplat/issues/new?labels=bug&template=bug-report.md">Report Bug</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/abhaskumarsinha/gaussian-litesplat/issues/new?labels=enhancement&template=feature-request.md">Request Feature</a>
  </p>

</div>




<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## 🧩 About Gaussian Splatting

**Gaussian Splatting** is a cutting-edge 3D representation technique that models a scene as a set of learnable 3D Gaussian primitives.  
Each Gaussian defines a point in space with position, color, opacity, and scale — allowing highly efficient, differentiable rendering of complex scenes from multiple camera views.

Unlike mesh or voxel-based methods, Gaussian Splatting provides:
- **Continuous, smooth surfaces** using anisotropic Gaussian functions  
- **Real-time rendering** capabilities with high-quality reconstruction  
- **Better scalability** for neural radiance fields (NeRF)-like tasks  

---

### 🧠 Why Gaussian LiteSplat?

**Gaussian LiteSplat** is a **minimal, portable Gaussian Splatting sandbox** implemented entirely in **Keras (Torch backend)** — no CUDA dependency required.  
It is designed for **researchers, educators, and developers** who want to understand, prototype, or test Gaussian Splatting on **CPU or lightweight GPU setups** (including Google Colab).

Key motivations:
- Make Gaussian Splatting **accessible** without high-end GPUs  
- Enable **rapid experimentation** on small-scale datasets  
- Serve as a **learning and benchmarking tool** for further research  

---

### ⚙️ Features

- 🧭 **COLMAP Importer**  
  Import scenes directly from COLMAP reconstructions (supports `SIMPLE_PINHOLE`, `PINHOLE`, and `SIMPLE_RADIAL` (without `k`)).

- 🔧 **End-to-End Training**  
  Import → Train → Export Gaussians in a simple, modular pipeline.

- 🌈 **Trainable RGB Color Support**  
  Simplified color representation — only RGB (no spherical harmonics as in the original paper).

- 🧮 **Lightweight and CPU-Compatible**  
  Runs entirely in Python and Keras (Torch backend) — no CUDA required.

- ⏱️ **Fast Experimentation**  
  Train a few thousand Gaussians and start seeing results within minutes to hours.

- ☁️ **Google Colab Friendly**  
  No complex environment setup — works with default Colab installs.

- 🧪 **Ideal For**  
  - Low-resolution or small-scale reconstruction experiments  
  - Projection and efficiency testing  
  - Teaching or demonstrating the concept  
  - Quick research prototyping  

---

### 💡 Summary

> *Gaussian LiteSplat is not a replacement for the original CUDA-accelerated implementation —  
> it’s a simplified, research-friendly version that emphasizes clarity, portability, and learning.*  
> Perfect for exploring the fundamentals of Gaussian Splatting on **any machine**, even without GPU support.



## ⚙️ Installation

Gaussian LiteSplat can be installed in two ways depending on your use case:

---

### 🪶 **Option 1 — Minimal Framework Installation**

Use this if you only need the **core Gaussian LiteSplat framework** for importing, training, and rendering:

```bash
!pip install git+https://github.com/abhaskumarsinha/Gaussian-LiteSplat.git
```
This installs only the main package and its runtime dependencies — perfect for Google Colab, light experiments, or embedding into your own projects.

### 🧰 **Option 2 — Full Development Installation**

Use this if you want the complete toolkit, including benchmarking scripts, training templates, and utilities for end-to-end experiments.
```bash
!git clone https://github.com/abhaskumarsinha/Gaussian-LiteSplat.git
%cd Gaussian-LiteSplat
!pip install -r requirements.txt
```

This setup includes:
- Benchmarking utilities
- Dataset templates
- Predefined training configurations
- Example scripts for reproducing results

✅ **Tip**:
Both installations work seamlessly on CPU and GPU (if available).
No CUDA setup is required — everything runs with Keras (Torch backend) out of the box.



## 🚀 Quick Start Example

Once installed, you can start training directly from a COLMAP reconstruction using the provided training script.  
In the **`Gaussian-LiteSplat`** directory, run:

```bash
!python ./scripts/train_colmap.py \
    --colmap_scene "[path_to_COLMAP_export_in_NeRF_format]" \
    --litesplat_scene "[path_to_save_litesplat_scene]" \
    --images_dir "[path_to_COLMAP_images_dir]" \
    --output_dir "output" \
    --torch_compile \
    --output_h 76 \
    --output_w 102 \
    --total_gaussians 2200 \
    --log_level DEBUG
```

## 🖼️ Results

Below is an example experiment comparing the **initial imported COLMAP scene** and the **trained Gaussian LiteSplat output**.

The dataset used is **TempleRing** from the [Middlebury Multiview Dataset](https://vision.middlebury.edu/mview/data/),  
featuring 15 camera views. The scene was trained with **2200 Gaussians** for **~45 minutes** on a CPU-only setup.

<br/>

<div align="center">

| Imported COLMAP Scene (Before Training) | Trained Gaussian LiteSplat Scene (After Training) |
|:--------------------------------------:|:-------------------------------------------------:|
| <img src="https://raw.githubusercontent.com/abhaskumarsinha/Gaussian-LiteSplat/refs/heads/main/G47yPFyXUAA2xUC.jpg" alt="Imported COLMAP Scene" width="360"/> | <img src="https://raw.githubusercontent.com/abhaskumarsinha/Gaussian-LiteSplat/refs/heads/main/G5Fk1gAa4AA9uhZ.jpg" alt="Trained Gaussian LiteSplat Scene" width="360"/> |

</div>

---

### 🧠 Observations

- The imported COLMAP scene provides sparse geometric information from Structure-from-Motion reconstruction.  
- Despite no CUDA or SH color support, LiteSplat efficiently learns compact Gaussian representations suitable for small-scale research.  
- Results begin to appear **within minutes**, making it ideal for testing projection, training dynamics, or dataset preprocessing.

---

> ⚡ *These results demonstrate that even a CPU-based, minimal implementation of Gaussian Splatting can produce  
> meaningful multi-view 3D reconstructions with limited hardware and simple training pipelines.*


## 📓 Notebook Examples

To help you get started quickly, **Gaussian LiteSplat** includes a set of ready-to-run **Jupyter notebooks**  
available in the [`/notebooks`](./notebooks) directory of the repository.

These notebooks demonstrate:
- 🔹 End-to-end workflow — importing a COLMAP scene, training Gaussians, and visualizing results  
- 🔹 Step-by-step explanation of LiteSplat internals (renderer, projection, and training loop)  
- 🔹 Example experiments for benchmarking and low-Gaussian-resolution studies  
- 🔹 Visualization utilities for comparing COLMAP vs trained LiteSplat scenes  

Each notebook is **Colab-compatible** — simply open it in Google Colab with no additional installations required.  
Just make sure to select the **Torch backend** for Keras if prompted.


## References
- Kerbl, Bernhard, et al. _"3D Gaussian splatting for real-time radiance field rendering."_ ACM Trans. Graph. 42.4 (2023): 139-1.

## 📚 Citation

If you find **Gaussian LiteSplat** useful in your research, teaching, or experimentation, please consider citing it:

```bibtex
@misc{Sinha2025GaussianLiteSplat,
  author       = {Abhas Kumar Sinha},
  title        = {Gaussian LiteSplat: A Minimal, Portable Gaussian Splatting Framework in Keras},
  year         = {2025},
  howpublished = {\url{https://github.com/abhaskumarsinha/Gaussian-LiteSplat}},
  note         = {A simplified, CPU-compatible Gaussian Splatting implementation for small-scale experimentation and research.}
}

