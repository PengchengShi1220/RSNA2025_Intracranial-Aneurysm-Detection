import dicom2nifti
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

def reorient_nii(orig_nii, targ_aff="LPS"):
    """
    Reorient to the standard LPS+ DICOM coordinate system.
    
    Args:
        orig_nii: Original nibabel image object
        targ_aff: Target orientation (default: "LPS")
    
    Returns:
        Reoriented nibabel image object
    """
    current_orientation = "".join(nib.aff2axcodes(orig_nii.affine))
    if current_orientation == targ_aff:
        return orig_nii
    
    orig_ornt = nib.io_orientation(orig_nii.affine)
    targ_ornt = nib.orientations.axcodes2ornt(targ_aff)
    transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
    img_orient = orig_nii.as_reoriented(transform)
    return img_orient

def convert_dicom_to_nifti(dicom_base_path, nifti_base_path):
    """
    Convert DICOM series to NIfTI format with LPS reorientation.
    """
    # Create output directory
    nifti_base_path.mkdir(parents=True, exist_ok=True)
    
    # Get all DICOM directories
    dicom_folders = [f for f in dicom_base_path.iterdir() if f.is_dir()]
    
    for dicom_folder in tqdm(dicom_folders, desc="Converting DICOM series"):
        # Build output file path
        output_filename = f"{dicom_folder.name}.nii.gz"
        output_path = nifti_base_path / output_filename

        print(f"Processing: {dicom_folder.name}")
        
        # Convert DICOM to NIfTI (without reorientation)
        orig_nii = dicom2nifti.dicom_series_to_nifti(
            str(dicom_folder), None, reorient_nifti=False
        )['NII']
        
        # Reorient to LPS
        img_orient = reorient_nii(orig_nii, targ_aff="LPS")
        
        # Save NIfTI file
        nib.save(img_orient, str(output_path))
        print(f"✓ Success: {dicom_folder.name} -> {output_filename}")

# Main execution
if __name__ == "__main__":
    # Define paths
    dicom_base_path = Path("dicom_series")
    nifti_base_path = Path("dicom_series_to_nifti_by_dicom2nifti")
    
    # Convert DICOM to NIfTI
    convert_dicom_to_nifti(dicom_base_path, nifti_base_path)
    print("All DICOM folders processed!")