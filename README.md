# non-rigid-pytorch
This project provides non-rigid methods with pytorch optimization, for easily modification to your project.

## Env Setup
We recommand Conda as virtual environment.

```bash
conda create -n nrd python=3.12 # other versions also work
conda activate nrd

pip install -r requirements.txt
```

## Core files
* `./core/deform.py`, we implent deformation with optimization, you can add other related loss functions for your task.

* `./renderer/ulit_renderer.py`, we use nvdiffrast for ulit render, OpenCV camera, align to PyRender.

* `./util/mesh.py`, we generate ED-Graph related methods here.


## Usage example
Use like `nrd.py`, example and related picture will be updated later.

## Citation
If you find this projects useful, considering citing:
```
@incollection{sumner2007embedded,
  title={Embedded deformation for shape manipulation},
  author={Sumner, Robert W and Schmid, Johannes and Pauly, Mark},
  booktitle={ACM siggraph 2007 papers},
  pages={80--es},
  year={2007}
}
```

This project could also be a prototype of [Project EvolvingGS](https://evolvinggs.github.io), [Project DisentangedGS](https://distangledgs.github.io/) and [Project Live4D](https://live4d-vv.github.io) (Non-Rigid Deformation, PyTorch version.)

```
@incollection{zhang2025evolvinggs,
  title={EvolvingGS: Stable Volumetric Video via High-Fidelity Evolving 3D Guassian Reconstruction},
  author={Zhang, Chao and Zhou, Yifeng and Wang, Shuheng and Li, Wenfa and Wang, Degang and Xu, Yi and Jiao, Shaohui},
  booktitle={Proceedings of the SIGGRAPH Asia 2025 Technical Communications},
  pages={1--4},
  year={2025}
}

@incollection{zhou2025disentangled,
  title={Disentangled Gaussian Splatting: High-Fidelity Relightable Volumetric Video through Geometry-Appearance Decoupling},
  author={Zhou, Yifeng and Zhang, Chao and Wang, Shuheng and Li, Wenfa and Yi, Xu and Wang, Degang and Chen, Yuzhong and Jiao, Shaohui},
  booktitle={Proceedings of the SIGGRAPH Asia 2025 Technical Communications},
  pages={1--4},
  year={2025}
}

@incollection{zhou2023live4d,
  title={Live4D: a real-time capture system for streamable volumetric video},
  author={Zhou, Yifeng and Wang, Shuheng and Li, Wenfa and Zhang, Chao and Rao, Li and Cheng, Pu and Xu, Yi and Ke, Jinle and Feng, Wenduo and Zhou, Wen and others},
  booktitle={SIGGRAPH Asia 2023 technical communications},
  pages={1--4},
  year={2023}
}
```