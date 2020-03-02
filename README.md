# DCE_DSC_AIF
Official code for DCE_DSC_AIF

1. To run, you need two directories obtained using NordicICE (NordicNeuroLab, Norway).
 1) "raw_data": contains DCE-MRI raw dicom files, and tumor ROI nifti file (.nii) in the subdirectory for each patient.
 2) "raw_txt": contains AIF obtained from DCE-MRI, and AIF obtained from DSC-MRI as text files (e.g. imgs_x1.txt, and imgs_y1.txt) in the subdirectory for each patient.
2. Compatible with:
- Keras 2.0.8
- Tensorflow 1.10.0
