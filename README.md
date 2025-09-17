# DMC-Net: Lightweight Dynamic Multi-scale and Multi-resolution convolution network for pancreas segmentation in CT images

Abstract
=======
Background and Objective: Convolutional neural networks (CNNs) have shown great effectiveness in medical
image segmentation. However, they may be limited in modeling large inter-subject variations in organ shapes
and sizes and exploiting global long-range contextual information. This is because CNNs typically employ
convolutions with fixed-sized local receptive fields and lack the mechanisms to utilize global information.

Methods: To address the above limitations, we developed Dynamic Multi-Resolution Convolution (DMRC) and
Dynamic Multi-Scale Convolution (DMSC) modules. Both modules enhance the representation capabilities of
single convolutions to capture varying scaled features and global contextual information. This is achieved in
the DMRC module by employing a convolutional filter on images with different resolutions and subsequently
utilizing dynamic mechanisms to model global inter-dependencies between features. In contrast, the DMSC
module extracts features at different scales by employing convolutions with different kernel sizes and
utilizing dynamic mechanisms to extract global contextual information. The utilization of convolutions with
different kernel sizes in the DMSC module may increase computational complexity. To lessen this burden, we
propose to use a lightweight design for convolution layers with a large kernel size. Thus, DMSC and DMRC
modules are designed as lightweight drop-in replacements for single convolutions, and they can be easily
integrated into general CNN architectures for end-to-end training. The segmentation network was proposed by
incorporating our DMSC and DMRC modules into a standard U-Net architecture, termed Dynamic Multi-scale
and Multi-resolution Convolution network (DMC-Net).

Results: To evaluate the effectiveness of DMSC and DMRC modules, we conducted experiments on pancreas
segmentation from abdominal computed tomography (CT) images on two commonly used benchmarks,
including NIH-Pancreas and MSD-Pancreas datasets, with 2D and 3D versions of the DMC-Net. 2D DMC-Net
achieved 85.64 and 79.82 Mean Dice Similarity Coefficient (DSC) scores in the NIH-Pancreas and MSD-Pancreas
datasets, respectively. Additionally, 3D DMC-Net achieved 87.97 and 82.92 Mean DSC scores in these two
datasets. The DMC-Net outperformed the state-of-the-art methods on pancreas segmentation in CT images.

Conclusions: Our proposed DMSC and DMRC can enhance the representation capabilities of single convolu-
tions and improve segmentation accuracy. Furthermore, their lightweight design led to lower computational
complexity while maintaining or improving segmentation performance.

Citation
=======
If you use DMC-Net in your research, please consider to cite our work.
```
@article{yang2025dmc,
  title={DMC-Net: Lightweight Dynamic Multi-scale and Multi-resolution convolution network for pancreas segmentation in CT images},
  author={Yang, Jin and Marcus, Daniel S and Sotiras, Aristeidis},
  journal={Biomedical Signal Processing and Control},
  volume={109},
  pages={107896},
  year={2025},
  publisher={Elsevier}
}
```
