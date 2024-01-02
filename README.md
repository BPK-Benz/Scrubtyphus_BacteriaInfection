# Scrubtyphus_BacteriaInfection

## 1. Objective 
Explore deep learning techniques to improve the speed and accuracy for Orientia tsutsugamushi bacterial infectibility by 3 techniques: 
transitioning from instance segmentation to object detection, adjusting backbone size, and reducing floating-point percision.


## 2. Image Dataset
### 2.1 What is Scrub Typhus?
Scrub typhus, caused by the bacterium Orientia tsutsugamushi, is transmitted by infected mites in vegetation-rich areas of Asia, the Pacific Islands, and the Middle East. Symptoms include fever, headache, rash, and swollen lymph nodes. Diagnosis relies on clinical and lab tests, with early antibiotic treatment being vital. No vaccine is widely available, so preventive measures, like protective clothing and insect repellent use, are key. Efforts continue to raise awareness and improve diagnostics and treatment in endemic regions. Timely medical attention is critical for recovery.

### 2.2 Image character
All organelles were fluorescently stained, with red for cell boundaries, blue for nucleus boundaries, and green for bacteria boundaries, and then captured using high-content screening.
The dataset includes numerous images containing both control genes and knockdown genes. It's worth noting that only the images with enhanced quality for the first control gene have been uploaded in "Enhanced_BioImage" directory. In contrast, the "Image_Datasets" directory has combined three fluorescence images into a single image for streamlined integration into deep learning models.

### 2.3 Data annotation
In-house software for data annotation has been generated and is available at https://github.com/Chuenchat/cellLabel
(https://github.com/BPK-Benz/Scrubtyphus_BacteriaInfection/assets/76678370/f2367995-9f98-4127-bc29-bac5ea79665e)

## 3. Cell coutning techniques comparion: 16 deep learning and 1 Image processing 
### 3.1 Image processing technique
CellProfiler is a notable tool for processing biological images, with an example illustrating its application in cell counting in the "IP " directory.

### 3.2 Deep learning by modified from MMdetection: 
- 4 main model architecture (2 Instance segmentation model: Cascade Mask R-CNN and Mask R-CNN, 2 Object detection model: Faster R-CNN and RetinaNet)
- 2 Backbone size: Resnet-101 and Resnet-50
- 2 Floating-point precision: 32 bit and 16 bit



