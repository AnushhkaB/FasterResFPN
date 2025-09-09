# FasteResFPN : Deep Learning for Automated Microplastic Detection in Aquatic Environments
FasterResFPN is an end-to-end, deep learning–based object detection system engineered to enable large-scale microplastic surveillance in natural waters. Built upon Faster R-CNN with a ResNet-50 + Feature Pyramid Network (FPN) backbone, it leverages multi-scale feature learning to classify and localize four major categories of microplastic particles: fibers, fragments, pellets, and films.

Trained on the Microplastic-100 dataset, the system achieves state-of-the-art recall while maintaining competitive precision, making it a strong alternative to costly and time-consuming laboratory techniques such as FTIR and Raman spectroscopy.

# Why FasterResFPN ?
Traditional microplastic detection relies on spectroscopy or manual microscopy, both of which are resource-intensive, slow, and impractical for large-scale monitoring. FasterResFPN, by contrast, applies two-stage deep detection to overcome real-world challenges like turbidity, lighting variation, and background clutter. Its use of an FPN backbone makes it effective across particle sizes, from large fragments to tiny thin films.

# Model Pipeline

1. **Backbone (ResNet-50 + FPN)**  
   - Extracts multi-scale features (P2–P5), capturing fine details (thin films) and broader context (fragments/pellets).

2. **Region Proposal Network (RPN)**  
   - Generates candidate bounding boxes using anchors, optimized with classification + regression loss.

3. **ROI Align**  
   - Preserves fine-grained details via bilinear interpolation for precise microplastic localization.

4. **Fast R-CNN Head**  
   - Performs classification into 4 classes and bounding box refinement for final predictions.
