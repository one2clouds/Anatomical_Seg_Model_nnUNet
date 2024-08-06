"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""

from nnUNet.nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

import numpy as np
# previously i got warning while importing things from nnunet and because i am using nnunetv2 so the required folder paths are specified like v2 so v1 env variablees were not defined so it kept giving me warning

from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch 
from nnUNet.nnunetv2.paths import nnUNet_results, nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import join

import SimpleITK as sitk 


# INPUT_PATH = Path("/input")
# OUTPUT_PATH = Path("/output")
# RESOURCE_PATH = Path("resources")


def change_direction(orig_image):
    new_img = sitk.DICOMOrient(orig_image,'RAS')
    return new_img


def run():
    
    # aa_img = sitk.GetArrayFromImage(sitk.ReadImage('/home/shirshak/Anatomical_Segmentation_Frac-Seg-Net/zzz_outputs/001_pred.mha'))
    # print(np.unique(aa_img, return_counts=True))
    # print(Aaaaaa)

    # TO CHANGE THE DIRECTION OF IMG 
    # <------------------------>
    new_img = change_direction(sitk.ReadImage('/home/shirshak/Anatomical_Segmentation_Frac-Seg-Net/zzz_input_data/001.mha'))
    sitk.WriteImage(new_img, '/home/shirshak/Anatomical_Segmentation_Frac-Seg-Net/zzz_input_data/001.mha')
    # <-------------------------->

    img, props = SimpleITKIO().read_images(['/home/shirshak/Anatomical_Segmentation_Frac-Seg-Net/zzz_input_data/001.mha'])

    # img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset600_CT_PelvicFrac150/imagesTr/001.mha')]) # This gives nice results 

    # img = sitk.ReadImage('/home/shirshak/Anatomical_Segmentation_Frac-Seg-Net/test_data/001.mha')
    # img = change_direction(img)
    # img_data = sitk.GetArrayFromImage(img)

    # img = img_data.copy()
    # print(img_data)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset600_CT_PelvicFrac150/nnUNetTrainer__nnUNetPlans__3d_fullres'),
        use_folds=(4,),
        checkpoint_name='checkpoint_best.pth',
    )

    predicted_segmentation, class_probabilities = predictor.predict_single_npy_array(img, props, None, None, True)
    
    # print(predicted_segmentation)

    sitk.WriteImage(sitk.GetImageFromArray(predicted_segmentation), join('/home/shirshak/Anatomical_Segmentation_Frac-Seg-Net/zzz_outputs_data', '001_pred.mha'))

    print(np.unique(predicted_segmentation, return_counts=True))
    # predicted_segmentations = predictor.predict_from_files(['./test_data/001.mha'],
    #                                                        None,
    #                                                        save_probabilities=True, overwrite=True,
    #                                                        num_processes_preprocessing=2,
    #                                                        num_processes_segmentation_export=2,
    #                                                        folder_with_segs_from_prev_stage=None, num_parts=1,
    #                                                        part_id=0)

    # print(predicted_segmentations)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
