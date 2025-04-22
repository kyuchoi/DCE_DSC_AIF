# AIF GAN
Official codes for "Improving the reliability of pharmacokinetic parameters in dynamic contrast-enhanced MRI in astrocytomas: Deep learning approach" published in Radiology
(https://pubs.rsna.org/doi/full/10.1148/radiol.2020192763)
- Transforms the arterial input function (AIF) obtained from DCE-MRI into one obtained from DSC-MRI, which improves reliability of Ktrans, Ve, and Vp maps
- Basically, a pix2pix model using Wasserstein GAN with gradient penalty (WGAN-GP) loss 

1. To run, you need the following two directories obtained using NordicICE (NordicNeuroLab, Norway), a commercially available software.
2. "raw_data": contains DCE-MRI raw dicom files, and tumor ROI nifti file (.nii) in the subdirectory for each patient.
3. "raw_txt": contains AIF obtained from DCE-MRI, and AIF obtained from DSC-MRI as text files (e.g. imgs_x1.txt, and imgs_y1.txt) in the subdirectory for each patient.
4. Requirements:
- Keras 2.0.8
- Tensorflow 1.10.0
