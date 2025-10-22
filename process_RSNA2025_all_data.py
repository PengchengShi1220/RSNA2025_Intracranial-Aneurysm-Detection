import SimpleITK as sitk
import nibabel as nib
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import dicom2nifti
from datetime import datetime
from multiprocessing import Pool, Manager, Lock
import pydicom
import subprocess

# --- Configuration ---
SOURCE_DICOM_DIR = "/XXX/RSNA-aneurysm-20250822/series"
BASE_OUTPUT_DIR = "/XXX/Dataset/RSNA_intracranial_aneurysm_challenge/RSNA_aneurysm_all_data_4348"
TRAIN_CSV_PATH = "train_localizers.csv"
LOG_FILE = os.path.join(BASE_OUTPUT_DIR, "log.txt")
NUM_PROCESSES = 20

# --- Supported DICOM Tags ---
SUPPORTED_DICOM_TAGS = ['Modality']

# --- Label mapping dictionary ---
LABEL_DICT = {
    "Other Posterior Circulation": 1,
    "Basilar Tip": 2,
    "Right Posterior Communicating Artery": 3,
    "Left Posterior Communicating Artery": 4,
    "Right Infraclinoid Internal Carotid Artery": 5,
    "Left Infraclinoid Internal Carotid Artery": 6,
    "Right Supraclinoid Internal Carotid Artery": 7,
    "Left Supraclinoid Internal Carotid Artery": 8,
    "Right Middle Cerebral Artery": 9,
    "Left Middle Cerebral Artery": 10,
    "Right Anterior Cerebral Artery": 11,
    "Left Anterior Cerebral Artery": 12,
    "Anterior Communicating Artery": 13
}

# Initialize thread-safe logging lock
log_lock = Lock()

def log_message(series_uid, message, is_error=True):
    """
    Log messages to file in a thread-safe manner with timestamp.
    """
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_type = "ERROR" if is_error else "SUCCESS"
    with log_lock:
        with open(LOG_FILE, 'a') as f:
            f.write(f"[{timestamp}] {log_type} - Series UID: {series_uid}, {message}\n")

def reorient_nii(orig_nii, targ_aff="LPS"):
    """
    Reorient to the standard LPS+ DICOM coord.
    """
    if "".join(nib.aff2axcodes(orig_nii.affine)) == targ_aff:
        return orig_nii
    orig_ornt = nib.io_orientation(orig_nii.affine)
    targ_ornt = nib.orientations.axcodes2ornt(targ_aff)
    transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
    img_orient = orig_nii.as_reoriented(transform)
    return img_orient

def extract_dicom_metadata(dicom_file, series_uid):
    """
    Extract only Modality from DICOM file.
    """
    try:
        ds = pydicom.dcmread(dicom_file, force=True)
        return {'modality': ds.get('Modality', 'Unknown')}
    except Exception as e:
        log_message(series_uid, f"Error extracting metadata from {dicom_file}: {str(e)}", is_error=True)
        return None

def process_single_series(series_uid, series_df, source_dicom_dir, images_output_dir, labels_single_output_dir, labels_box_output_dir, is_annotated=True):
    """
    Process a single SeriesInstanceUID: convert DICOM to NIfTI, extract metadata, and create masks.
    """
    image_filename = os.path.join(images_output_dir, f"{series_uid}_0000.nii.gz")
    full_subdir_path = os.path.join(source_dicom_dir, series_uid)

    if not os.path.isdir(full_subdir_path):
        log_message(series_uid, f"Directory for {series_uid} not found")
        return

    # Get DICOM file list
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(full_subdir_path)
    if not dicom_names:
        log_message(series_uid, f"No DICOM files found for series {series_uid}")
        return

    # Extract Modality
    metadata = extract_dicom_metadata(dicom_names[0], series_uid)
    if metadata is None:
        log_message(series_uid, f"Failed to extract metadata for series {series_uid}")
        return

    image = None
    
    # Convert DICOM to NIfTI (without reorientation)
    orig_nii = dicom2nifti.dicom_series_to_nifti(
        str(full_subdir_path), None, reorient_nifti=False
    )['NII']

    # Reorient to LPS
    img_orient = reorient_nii(orig_nii, targ_aff="LPS")
    img_orient.to_filename(image_filename)

    size = img_orient.shape         
    spacing = img_orient.header.get_zooms()  
    affine = img_orient.affine

    metadata['size'] = size
    metadata['spacing'] = spacing

    # Initialize masks
    mask_array_single = np.zeros(size[::-1], dtype=np.uint8)
    mask_array_box = np.zeros(size[::-1], dtype=np.uint8)

    if is_annotated and series_df is not None:
        # Generate random box size (in mm)
        box_size_x_mm = np.random.uniform(2.0, 5.0)
        box_size_y_mm = np.random.uniform(2.0, 5.0)
        box_size_z_mm = np.random.uniform(2.0, 5.0)
        
        # Convert to pixel units
        box_size_x = int(round(box_size_x_mm / spacing[0]))
        box_size_y = int(round(box_size_y_mm / spacing[1]))
        box_size_z = int(round(box_size_z_mm / spacing[2]))

        # Process annotations
        for _, row in series_df.iterrows():
            sop_uid = row['SOPInstanceUID']
            location_name = row['location']
            
            label_value = LABEL_DICT.get(location_name)
            if label_value is None:
                log_message(series_uid, f"Unknown location '{location_name}' for series {series_uid}")
                continue

            try:
                coords = eval(row['coordinates'])
                if not isinstance(coords, dict) or 'x' not in coords or 'y' not in coords:
                    raise ValueError("Invalid coordinates format")
                
                x_idx = int(round(float(coords['x'])))
                y_idx = int(round(float(coords['y'])))

                if is_multiframe:
                    if 'f' in coords:
                        z_idx = int(round(float(coords['f'])))
                    else:
                        log_message(series_uid, f"Multi-frame DICOM series {series_uid} missing 'f' field in coordinates, SOPInstanceUID {sop_uid}")
                        continue
                else:
                    try:
                        dicom_file_path = os.path.join(full_subdir_path, f"{sop_uid}.dcm")
                        z_idx = dicom_names.index(dicom_file_path)
                    except ValueError:
                        log_message(series_uid, f"Could not find SOPInstanceUID {sop_uid} in series {series_uid}")
                        continue
            except (ValueError, SyntaxError) as e:
                log_message(series_uid, f"Failed to parse coordinates for SOPInstanceUID {sop_uid} in series {series_uid}: {str(e)}")
                continue
            
            # Boundary check for single pixel
            if not (0 <= x_idx < size[0] and 0 <= y_idx < size[1] and 0 <= z_idx < size[2]):
                log_message(series_uid, f"Single pixel coordinates ({x_idx}, {y_idx}, {z_idx}) out of bounds for image size {size}, SOPInstanceUID: {sop_uid}")
                continue

            try:
                mask_array_single[z_idx, y_idx, x_idx] = label_value
            except IndexError as e:
                log_message(series_uid, f"IndexError for coordinates ({x_idx}, {y_idx}, {z_idx}) in series {series_uid}, Image size: {size}, Error: {str(e)}")
                continue
            
            # Calculate box coordinates
            x_start = max(0, x_idx - box_size_x // 2)
            y_start = max(0, y_idx - box_size_y // 2)
            z_start = max(0, z_idx - box_size_z // 2)
            
            x_end = min(size[0], x_idx + box_size_x // 2 + 1)
            y_end = min(size[1], y_idx + box_size_y // 2 + 1)
            z_end = min(size[2], z_idx + box_size_z // 2 + 1)
            
            if z_end > z_start and y_end > y_start and x_end > x_start:
                try:
                    mask_array_box[z_start:z_end, y_start:y_end, x_start:x_end] = label_value
                except IndexError as e:
                    log_message(series_uid, f"IndexError for box coordinates ({x_idx}, {y_idx}, {z_idx}) in series {series_uid}, Image size: {size}, Error: {str(e)}")
                    continue
            else:
                log_message(series_uid, f"Box coordinates for ({x_idx}, {y_idx}, {z_idx}) resulted in invalid slice range for series {series_uid}, Image size: {size}")

    # Save masks
    labels_filename = f"{series_uid}.nii.gz"

    affine = img_orient.affine.copy()
    header = img_orient.header.copy()
    
    mask_array_single = np.transpose(mask_array_single, (2, 1, 0))
    mask_array_box = np.transpose(mask_array_box, (2, 1, 0))

    mask_nii_single = nib.Nifti1Image(mask_array_single, affine, header)
    mask_nii_box = nib.Nifti1Image(mask_array_box, affine, header)
    
    img_orient_single = reorient_nii(mask_nii_single, targ_aff="LPS")
    img_orient_single.to_filename(os.path.join(labels_single_output_dir, labels_filename))
    img_orient_box = reorient_nii(mask_nii_box, targ_aff="LPS")
    img_orient_box.to_filename(os.path.join(labels_box_output_dir, labels_filename))

    log_message(series_uid, f"Series processed successfully {'(unannotated)' if not is_annotated else ''}, Modality: {metadata['modality']}, Spacing: {metadata['spacing']}, Size: {metadata['size']}", is_error=False)

def process_series_wrapper(args):
    """
    Wrapper function for process_single_series to handle multiprocessing.
    """
    series_uid, series_df, source_dicom_dir, images_output_dir, labels_single_output_dir, labels_box_output_dir, is_annotated = args
    process_single_series(series_uid, series_df, source_dicom_dir, images_output_dir, labels_single_output_dir, labels_box_output_dir, is_annotated)

def main():
    """
    Main function to execute all data processing steps with multiprocessing.
    """
    imagesTr_dir = os.path.join(BASE_OUTPUT_DIR, "imagesTr")
    labelsTr_single_dir = os.path.join(BASE_OUTPUT_DIR, "labelsTr_single")
    labelsTr_box_dir = os.path.join(BASE_OUTPUT_DIR, "labelsTr_3d_box")

    os.makedirs(imagesTr_dir, exist_ok=True)
    os.makedirs(labelsTr_single_dir, exist_ok=True)
    os.makedirs(labelsTr_box_dir, exist_ok=True)

    if not os.path.isfile(TRAIN_CSV_PATH):
        print(f"Error: CSV file {TRAIN_CSV_PATH} not found")
        return
    
    df = pd.read_csv(TRAIN_CSV_PATH)
    if 'SeriesInstanceUID' not in df.columns:
        print("Error: SeriesInstanceUID column missing in CSV file")
        return
    
    annotated_series_uids = set(df['SeriesInstanceUID'].unique())
    all_series_uids = set([d for d in os.listdir(SOURCE_DICOM_DIR) if os.path.isdir(os.path.join(SOURCE_DICOM_DIR, d))])
    unannotated_series_uids = all_series_uids - annotated_series_uids

    print(f"Total series in directory: {len(all_series_uids)}")
    print(f"Annotated series (from CSV): {len(annotated_series_uids)}")
    print(f"Unannotated series: {len(unannotated_series_uids)}")

    all_args = [
        (series_uid, series_df_subset, SOURCE_DICOM_DIR, imagesTr_dir, labelsTr_single_dir, labelsTr_box_dir, True)
        for series_uid, series_df_subset in df.groupby('SeriesInstanceUID')
    ]
    all_args.extend([
        (series_uid, None, SOURCE_DICOM_DIR, imagesTr_dir, labelsTr_single_dir, labelsTr_box_dir, False)
        for series_uid in unannotated_series_uids
    ])

    print(f"Total series to process: {len(all_args)}")

    with Pool(processes=NUM_PROCESSES) as pool:
        list(tqdm(pool.imap(process_series_wrapper, all_args), total=len(all_args), desc="Processing all series"))

if __name__ == "__main__":

    main()
