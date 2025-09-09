# 🔬 FasterResFPN: Deep Learning for Automated Microplastic Detection in Aquatic Environments

FasterResFPN is an end-to-end, deep learning–based object detection system engineered to enable large-scale microplastic surveillance in natural waters. Built upon **Faster R-CNN with a ResNet-50 + Feature Pyramid Network (FPN) backbone**, it leverages multi-scale feature learning to classify and localize four major categories of microplastic particles: fibers, fragments, pellets, and films.

The system achieves state-of-the-art recall while maintaining competitive precision, making it a strong alternative to costly and time-consuming laboratory techniques such as FTIR and Raman spectroscopy.

## 📊 Example Detections

<p align="center">
  <img src="microplastic_detected.PNG" alt="Examples of detected microplastic types : fibers, fragments, pellets, and films." width="800"/>
</p>

Examples of four microplastic types detected : fibers, fragments, pellets, and films.
## 🚀 Why FasterResFPN?
Traditional microplastic detection relies on spectroscopy or manual microscopy, both of which are resource-intensive, slow, and impractical for large-scale monitoring. These methods also demand specialized equipment and expertise, limiting their accessibility in field applications. FasterResFPN, by contrast, applies two-stage deep detection to overcome real-world challenges like turbidity, lighting variation, and background clutter. By leveraging a ResNet-50 backbone with a Feature Pyramid Network (FPN), it captures both fine-grained details and high-level semantic features, making it effective across particle sizes—from large fragments and pellets to tiny, low-contrast fibers and thin films. Trained with diverse augmentations on the dataset, FasterResFPN achieves strong recall and accuracy, offering a scalable, automated alternative for environmental monitoring.

## 🗺️ Model Pipeline

1. **Backbone (ResNet-50 + FPN)**
   - Extracts **multi-scale hierarchical features** (P2–P5), allowing the model to capture both **fine-grained details** like thin films and fibers as well as **broader contextual cues** for larger particles such as fragments and pellets. This ensures balanced performance across varying microplastic sizes.

2. **Region Proposal Network (RPN)**
   - Generates **candidate bounding boxes (anchors)** across different scales and aspect ratios. These proposals are refined using **classification and regression losses**, filtering out irrelevant regions while preserving likely microplastic candidates.

3. **ROI Align**
   - Applies **bilinear interpolation** to align proposed regions precisely with feature maps. This step avoids the quantization errors of ROI Pooling and ensures that even **subtle features of translucent or irregular particles** are preserved for accurate detection.

4. **Fast R-CNN Head**
   - Processes aligned regions to perform **final classification** into 4 microplastic categories (fibers, fragments, pellets, films). Simultaneously, it applies **bounding box regression** for improved localization, ensuring tighter and more accurate predictions.

## 📈 Results

- **F1 Score**: 0.912  
- **mAP@0.5**: 0.901  
- **Recall**: 0.948 (best among tested baselines)  

**Comparison**:  
Outperforms **YOLOv7–YOLOv9** in recall, ensuring fewer missed microplastics.

**Per-class Performance**:  
- **Fibers**: Strongest performance (F1 > 0.98)  
- **Fragments & Pellets**: Balanced detection accuracy  
- **Films**: Weaker localization due to translucency and amorphous boundaries (dominant failure mode)

## 🔎 Features
- **Multi-scale detection** of different microplastic sizes via FPN.  
- **High recall** ensures fewer missed detections in noisy aquatic imagery.  
- **Robust augmentation** pipeline (flips, scaling, jitter, rotation) to handle diverse real-world conditions.  
- **Colab-compatible training notebook** (NVIDIA T4 GPU) included for reproducibility.  
- **COCO-format dataset support**, allowing easy retraining or fine-tuning on new data.  

---

## 🚀 Future Work
- Experiment with DIoU/CIoU losses, aspect-ratio clustering, and super-resolution for thin films.  
- Extend dataset with deep-sea, estuarine, and wastewater samples.  
- Apply pruning + quantization for real-time use on UAVs, AUVs, and embedded GPUs.  


