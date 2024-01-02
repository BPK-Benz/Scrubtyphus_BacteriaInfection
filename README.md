# Orientia tsutsugamushi Bacterial Infection in Scrub typhus 

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

## 3. Cell coutning techniques comparion: 16 deep learning and 1 Image processing 
### 3.1 Image processing technique
CellProfiler is a notable tool for processing biological images, with an example illustrating its application in cell counting in the "IP " directory.

### 3.2 Deep learning by modified from MMdetection: 
- 4 main model architecture (2 Instance segmentation model: Cascade Mask R-CNN and Mask R-CNN, 2 Object detection model: Faster R-CNN and RetinaNet)
- 2 Backbone size: Resnet-101 and Resnet-50
- 2 Floating-point precision: 32 bit and 16 bit

### How to Run Training and Evaluation using mmdetection
### Prerequisites
- Before you begin, ensure you have mmdetection installed. If not, you can follow the installation instructions from mmdetection's official repository. You can access it at https://github.com/open-mmlab/mmdetection/projects.<br>
- Pre-trained models can be downloaded from the official mmdetection website before you starting.<br>
- Several files in the mmdetection repository were modified (looking at modified_mmdetection folder) and utilized to evaluate metrics for our project.<br>

### Navigate at Modified mmdetection folder
### 3.2.1 Training
Navigate to the "CellCounting_models" directory.<br>
- Use the following command to initiate training:
python tools/train.py <path_to_config_file(.py)><br>
Replace <path_to_config_file> with the path to the desired model config file. <br>


### 3.2.2 Evaluation: Testing, Confusion matrix, Train_Time, inference_time
- Use the following command for testing:
- 1) Predicted result (Image)
python tools/test.py <path_to_config_file> <path_to_checkpoint(.pth)> --show-dir<path_to_results> --eval bbox --out <path_to_pkl_file(.pkl)> --eval-option proposal_nums="(200,300,1000)" classwise=True save_path=<path_to_save> <br>

- 2) Predicted result (Json file)
python tools/test.py <path_to_config_file> <path_to_checkpoint>  --show-dir <path_to_results> --eval bbox --out <path_to_pkl_file(.pkl)> --options jsonfile_prefix=<path_to_save> <br>

- Use the following command for confusion matrix: python tools/analysis_tools/confusion_matrix.py <path_to_config_file>   <path_to_pkl_file> <path_to_save> <br>

- Use the following command for train_time: python tools/analysis_tools/analyze_logs.py cal_train_time log.json <br>

- Use the following command for inference_time
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py <path_to_config_file> <path_to_checkpoint> --launcher pytorch --save_path <path_to_save> <br>

### 3.2.3 Example of Model results by looking at the folder
- Hyperparameter tuning
- Model prediction
- Bacterial infection assessment

## 4. Comprehensive graph and table publication 
Main figure & table and supplementary figures are presented in a graph publication folder








