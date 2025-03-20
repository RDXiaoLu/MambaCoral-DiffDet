![image](https://github.com/user-attachments/assets/3bb760f8-c981-4464-864d-5b23801f5273)
<div align="center">  
         
# A Diffusion Model and Knowledge Distillation Framework for Robust Coral Detection in Complex Underwater Environments

[![Dataset Download](https://img.shields.io/badge/Download-MambaCoral--DiffDet%20Dataset-blue)](https://drive.google.com/file/d/1XZYcADIhvO0XxR-iltXc7dliJzzwv23Y/view?usp=drive_link)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-blue?style=flat&logo=Huggingface)](https://huggingface.co/RDXiaolu/MCDD)  
![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-diffusion-model-and-knowledge-distillation/2d-object-detection-on-scoraldet-dataset-1)

</div> 

---

## 📝 Project Overview
**MambaCoral-DiffDet (MMCD)** is a robust diffusion model and knowledge distillation framework designed for coral detection in complex underwater environments. It enhances coral recognition accuracy and efficiency by incorporating state-of-the-art generative models. The code for **MambaCoral-DiffDet** will be made publicly available once our paper is accepted.

---

## 📅 Recent Updates  
- **[2025/03/20]**: We have fully open-sourced the code for the **MCDD** framework. We welcome everyone to try and apply it to their own research! 
- **[2025/03/15]**: We have added **5 coral videos** to enrich the dataset and provide additional resources for your research [Google Drive](https://drive.google.com/file/d/1gGZgNfaqIUClyeygsUnVDogAIwkJLGx9/view?usp=drive_link). 
- **[2025/03/11]**: Integrated the **DGM(Diffusion Model-Driven Data Generation Module)** ComfyUI workflow as a `.json` file along with the visual representation of the workflow for easier implementation!  
- **[2024/11/20]**: We've opened up our dataset! [[MCDD-Dataset]](https://drive.google.com/file/d/1XZYcADIhvO0XxR-iltXc7dliJzzwv23Y/view?usp=drive_link)
---

## 📂 Dataset Information

### Original Dataset
The original dataset used in this project is available at the following link:
- **[SCoralDet Dataset](https://github.com/RDXiaoLu/SCoralDet-Dataset.git)**

This dataset contains images from six coral species: Euphylliaancora, Favosites, Platygyra, Sarcophyton, Sinularia, and Wavinghand, collected from the **Coral Germplasm Conservation and Breeding Center** at **Hainan Tropical Ocean University**.

### Augmented Dataset
Using the **MambaCoral-DiffDet** model's **DGM structure**, we have created an augmented dataset. The dataset now contains **1,204 images**, representing an **86% increase** in image quantity while using only **18% of the original images**. This augmentation helps improve the robustness of coral detection models by providing a more diverse set of training images.

You can download the augmented dataset here:
- **[MambaCoral-DiffDet Augmented Dataset](https://drive.google.com/file/d/1XZYcADIhvO0XxR-iltXc7dliJzzwv23Y/view?usp=drive_link)**
- We also provide **5 coral videos** as part of the dataset. You can access and download them via the following [Google Drive](https://drive.google.com/file/d/1gGZgNfaqIUClyeygsUnVDogAIwkJLGx9/view?usp=drive_link).

---

## Dataset Preview

To showcase the diversity of generated images, here are multiple augmented versions of the same coral species generated by the DGM (Data Generation Module).

| **Original Image**  | **Generated Image 1** | **Generated Image 2** | **Generated Image 3** |
|:-------------------:|:---------------------:|:---------------------:|:---------------------:|
| ![Original Euphylliaancora](https://github.com/RDXiaoLu/SCoralDet-Dataset/blob/main/Data%20Preview/Euphylliaancora.png) | ![Augmented Euphylliaancora](https://github.com/RDXiaoLu/MambaCoral-DiffDet/blob/main/Dataset%20Preview/Generate%20Images/Euphflfiaancora_129.JPG) | ![Augmented Euphylliaancora](https://github.com/RDXiaoLu/MambaCoral-DiffDet/blob/main/Dataset%20Preview/Generate%20Images/Euphflfiaancora_175.JPG) | ![Augmented Euphylliaancora](https://github.com/RDXiaoLu/MambaCoral-DiffDet/blob/main/Dataset%20Preview/Generate%20Images/Euphflfiaancora_126.JPG) |
| ![Original Platygyra](https://github.com/RDXiaoLu/SCoralDet-Dataset/blob/main/Data%20Preview/Platygyra.png) | ![Augmented Platygyra](https://github.com/RDXiaoLu/MambaCoral-DiffDet/blob/main/Dataset%20Preview/Generate%20Images/Platygyra_126.JPG) | ![Augmented Platygyra](https://github.com/RDXiaoLu/MambaCoral-DiffDet/blob/main/Dataset%20Preview/Generate%20Images/Platygyra_109.JPG) | ![Augmented Platygyra](https://github.com/RDXiaoLu/MambaCoral-DiffDet/blob/main/Dataset%20Preview/Generate%20Images/Platygyra_206.JPG) |
| ![Original Sarcophyton](https://github.com/RDXiaoLu/SCoralDet-Dataset/blob/main/Data%20Preview/Sarcophyton.png) | ![Augmented Sarcophyton](https://github.com/RDXiaoLu/MambaCoral-DiffDet/blob/main/Dataset%20Preview/Generate%20Images/Sarcophyton_114.JPG) | ![Augmented Sarcophyton](https://github.com/RDXiaoLu/MambaCoral-DiffDet/blob/main/Dataset%20Preview/Generate%20Images/Sarcophyton_112.JPG) | ![Augmented Sarcophyton](https://github.com/RDXiaoLu/MambaCoral-DiffDet/blob/main/Dataset%20Preview/Generate%20Images/Sarcophyton_117.JPG) |
| ![Original Wavinghand](https://github.com/RDXiaoLu/SCoralDet-Dataset/blob/main/Data%20Preview/Wavinghand.png) | ![Augmented Wavinghand](https://github.com/RDXiaoLu/MambaCoral-DiffDet/blob/main/Dataset%20Preview/Generate%20Images/WavingHand_361.JPG) | ![Augmented Wavinghand](https://github.com/RDXiaoLu/MambaCoral-DiffDet/blob/main/Dataset%20Preview/Generate%20Images/WavingHand_330.JPG) | ![Augmented Wavinghand](https://github.com/RDXiaoLu/MambaCoral-DiffDet/blob/main/Dataset%20Preview/Generate%20Images/WavingHand_170.JPG) |
| ![Original Favosites](https://github.com/RDXiaoLu/SCoralDet-Dataset/blob/main/Data%20Preview/Favosites.png) | ![Augmented Favosites](https://github.com/RDXiaoLu/MambaCoral-DiffDet/blob/main/Dataset%20Preview/Generate%20Images/Favosites_153.JPG) | ![Augmented Favosites](https://github.com/RDXiaoLu/MambaCoral-DiffDet/blob/main/Dataset%20Preview/Generate%20Images/Favosites_133.JPG) | ![Augmented Favosites](https://github.com/RDXiaoLu/MambaCoral-DiffDet/blob/main/Dataset%20Preview/Generate%20Images/Favosites_108.JPG) |

---

## 🔗 Connection with SCoralDet
This work builds upon the **SCoralDet** dataset and research, extending the capabilities of coral detection models by introducing advanced techniques and enhanced architectures.

---

## 📊 Performance Comparison of MambaCoral-DiffDet (MCDD) with SOTA Models

The table below shows a comparison of **MambaCoral-DiffDet (MCDD)** against state-of-the-art (SOTA) models, demonstrating its improved performance in terms of mAP, precision, recall, and computational efficiency.

| **Model**    | **mAP50** | **mAP(50-95)** | **Precision** | **Recall** | **Parameters (M)** | **GFLOPs** |
|--------------|-----------|----------------|---------------|------------|--------------------|------------|
| MambaYOLO    | 0.801     | 0.52          | 0.848         | 0.723      | 6.0               | 13.6       |
| RT-DETR      | 0.816     | 0.546         | 0.881         | 0.770      | 42.0              | 129.6      |
| YOLOv8       | 0.790     | 0.503         | 0.782         | 0.738      | 3.0               | 8.1        |
| YOLOv9       | 0.788     | 0.521         | 0.875         | 0.681      | 2.0               | 7.6        |
| YOLOv10      | 0.797     | 0.512         | 0.800         | 0.743      | 2.3               | 6.5        |
| YOLOv11      | 0.799     | 0.518         | 0.847         | 0.735      | 2.6               | 6.3        |
| MCDD (Ours)  | **0.843** | **0.566**     | **0.876**     | **0.750**  | **6.5**           | **13.6**   |

*Table: Comparison of MambaCoral-DiffDet (MCDD) with state-of-the-art performance models.*

---

## 🛠️ How to use DGM

First make sure you can run [ComfyUI](https://github.com/comfyanonymous/ComfyUI), If not, complete the following steps：
```markdown 
git clone https://github.com/comfyanonymous/ComfyUI
```
In order to facilitate the use of academic workers and researchers, we used ComfyUI to build the DGM workflow. We provided the configuration file **DGM_cfg.json**, and its parameters and configuration are shown in the figure DGM_cfg.

![DGM_cfg](https://github.com/RDXiaoLu/MambaCoral-DiffDet/blob/main/fig/DGM_cfg.png)

You only need to import the **.json** file we provide and download the corresponding model weights to generate your own coral images.



---


## 🔍 Citation 

For more details on the original dataset, refer to the paper:

```markdown 
@ARTICLE{lu2024scoraldet,  
         author={Lu, Zhaoxuan and Liao, Lyuchao and Xie, Xingang and Yuan, Hui},  
         title={SCoralDet: Efficient real-time underwater soft coral detection with YOLO},  
         journal={Ecological Informatics},  
         year={2024},  
         artnum={102937},  
         issn={1574-9541},  
         doi={10.1016/j.ecoinf.2024.102937},  
}  
```


---

## 🎯 Applications
This dataset and the MambaCoral-DiffDet framework can be used for :

- Coral species detection and classification
- Object detection in underwater environments
- Data augmentation using diffusion models
- Knowledge distillation for marine biology applications
