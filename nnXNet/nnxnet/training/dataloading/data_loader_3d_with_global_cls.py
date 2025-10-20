import os
import pandas as pd
import numpy as np
from nnxnet.training.dataloading.base_data_loader import nnXNetDataLoaderBase
from nnxnet.training.dataloading.nnxnet_dataset import nnXNetDataset

class nnXNetDataLoader3DWithGlobalCls(nnXNetDataLoaderBase):
    def __init__(self, *args, use_sampling_weight=None, **kwargs):
        """
        Parameters:
        use_sampling_weight: Option for using weights
            - None: No weights used, uniform sampling
            - 'modality': Use only modality weights
            - 'vessel': Use only vessel anatomy category weights
        """
        super().__init__(*args, **kwargs)
        
        # 定义权重文件路径
        self.case_sampling_weight_path = os.path.join("/yinghepool/shipengcheng/Dataset/nnUNet", 'train_case_sampling_weight.csv')
        self.csv_path = os.path.join("/yinghepool/shipengcheng/Dataset/nnUNet", 'train_augmented.csv')
        
        self.use_sampling_weight = use_sampling_weight
        
        self.df_full = pd.read_csv(self.csv_path)
        
        self.modality_mapping = {
            'CTA': 0, 
            'MRA': 1,
            'MRI T2': 2,
            'MRI T1post': 3
        }
        
        self.sex_mapping = {
            'Female': 0,
            'Male': 1
        }
        
        self.vessel_columns = [
            'Left Infraclinoid Internal Carotid Artery',
            'Right Infraclinoid Internal Carotid Artery', 
            'Left Supraclinoid Internal Carotid Artery',
            'Right Supraclinoid Internal Carotid Artery',
            'Left Middle Cerebral Artery',
            'Right Middle Cerebral Artery',
            'Anterior Communicating Artery',
            'Left Anterior Cerebral Artery',
            'Right Anterior Cerebral Artery',
            'Left Posterior Communicating Artery',
            'Right Posterior Communicating Artery',
            'Basilar Tip',
            'Other Posterior Circulation'
        ]
        
        self.aneurysm_column = 'Aneurysm Present'
    
        self.set_sampling_probabilities_from_csv()
    
    def set_sampling_probabilities_from_csv(self):
        """
        Set the sampling probability according to the weight setting option and the train.csv file.
        """

        # --- 1. Handle Uniform Sampling (No Weights) ---
        if self.use_sampling_weight is None:
            n_samples = len(self.indices)
            self.sampling_probabilities = np.ones(n_samples) / n_samples
            print("Using uniform sampling probability (no weights used)")
            return

        if not os.path.exists(self.case_sampling_weight_path):
            self._calculate_and_save_full_weights(self.df_full)

        if os.path.exists(self.case_sampling_weight_path):
            df_weights = pd.read_csv(self.case_sampling_weight_path)

            # Determine which weight column to use based on the user option
            if self.use_sampling_weight == 'modality':
                weight_column = 'modality_weight'
            elif self.use_sampling_weight == 'vessel':
                weight_column = 'vessel_weight'
            else:
                n_samples = len(self.indices)
                self.sampling_probabilities = np.ones(n_samples) / n_samples
                print(f"Warning: Unknown weight type '{self.use_sampling_weight}', using uniform sampling probability")
                return

            self.sampling_probabilities = self._load_weights_for_indices(df_weights, weight_column)

            print(f"Successfully loaded {self.use_sampling_weight} weights from {self.case_sampling_weight_path}, total {len(self.indices)} samples")
            print(f"Probability distribution: min={self.sampling_probabilities.min():.6f}, "
                f"max={self.sampling_probabilities.max():.6f}, "
                f"mean={self.sampling_probabilities.mean():.6f}")
        else:
            n_samples = len(self.indices)
            self.sampling_probabilities = np.ones(n_samples) / n_samples
            print("File not found after calculating weights, using uniform sampling probability")

    def _calculate_and_save_full_weights(self, df_full):
        """
        Calculates weights for all samples in the CSV and saves the two weight types.
        """
        # Count the number of positive samples for each modality
        modality_positive_counts = {}
        modality_total_counts = {}
        
        # Count the number of positive samples for each vessel category
        vessel_positive_counts = {col: 0 for col in self.vessel_columns}
        
        # First Pass: Count statistics
        for _, row in df_full.iterrows():
            modality = row['Modality']
            
            # Update modality statistics
            if modality not in modality_total_counts:
                modality_total_counts[modality] = 0
                modality_positive_counts[modality] = 0
            
            modality_total_counts[modality] += 1
            
            # Check for any vessel positivity
            # Note: If a sample is positive for multiple vessels, 
            # modality_positive_counts increases only once per sample, 
            # but vessel_positive_counts increases for each positive vessel.
            is_modality_positive = False 
            for vessel_col in self.vessel_columns:
                if row[vessel_col] == 1:
                    vessel_positive_counts[vessel_col] += 1
                    is_modality_positive = True
            
            if is_modality_positive:
                modality_positive_counts[modality] += 1

        # --- Print Statistics (for debugging/logging) ---
        print("Modality Statistics (based on full CSV):")
        for modality in modality_total_counts:
            pos_count = modality_positive_counts[modality]
            total_count = modality_total_counts[modality]
            print(f"  {modality}: {pos_count}/{total_count} positive samples")
        
        print("\nVessel Category Positive Statistics (based on full CSV):")
        for vessel_col in self.vessel_columns:
            pos_count = vessel_positive_counts[vessel_col]
            total_count = len(df_full)
            if pos_count > 0:
                print(f"  {vessel_col}: {pos_count}/{total_count} positive samples")
        
        # Second Pass: Calculate weights for all samples
        uids = []
        modality_weights = []
        vessel_weights = []

        for _, row in df_full.iterrows():
            uid = row['SeriesInstanceUID']
            modality = row['Modality']
            
            # --- Modality Weight ---
            modality_weight = 1.0
            if modality in modality_positive_counts:
                pos_count = modality_positive_counts[modality]
                total_count = modality_total_counts[modality]
                if pos_count > 0 and total_count > 0:
                    # The lower the positive rate, the higher the weight (Inverse frequency)
                    positive_rate = pos_count / total_count
                    modality_weight = 1.0 / positive_rate if positive_rate > 0 else 1.0
            
            # --- Vessel Category Weight ---
            vessel_weight = 1.0
            positive_vessel_count = 0
            vessel_weight_sum = 0.0
            
            # Accumulate weight components based on positive vessels in the current sample
            for vessel_col in self.vessel_columns:
                if row[vessel_col] == 1:
                    positive_vessel_count += 1
                    pos_count = vessel_positive_counts[vessel_col]
                    if pos_count > 0:
                        # Contribution to the sum is inversely proportional to the vessel's positive count, scaled by 300.0
                        vessel_weight_sum += 300.0 / pos_count
            
            # Final vessel weight is based on the average contribution from its positive vessels
            if positive_vessel_count > 0:
                vessel_weight = 1.0 + vessel_weight_sum / positive_vessel_count
                
            uids.append(uid)
            modality_weights.append(modality_weight)
            vessel_weights.append(vessel_weight)
            
        # Save to CSV file, containing only the two weight types
        df_weights_out = pd.DataFrame({
            'SeriesInstanceUID': uids,
            'modality_weight': modality_weights,
            'vessel_weight': vessel_weights
        })
        df_weights_out.to_csv(self.case_sampling_weight_path, index=False)
        print(f"All sample weights saved to {self.case_sampling_weight_path}")
        print(f"Modality weight range: [{min(modality_weights):.3f}, {max(modality_weights):.3f}]")
        print(f"Vessel weight range: [{min(vessel_weights):.3f}, {max(vessel_weights):.3f}]")

    def _load_weights_for_indices(self, df_weights, weight_column):
        """
        Loads the specified weight column corresponding to self.indices from the weights DataFrame.
        """
        
        # Convert self.indices into a DataFrame for merging purposes.
        # It assumes self.indices contains identifiers like 'SeriesInstanceUID'.
        df_indices = pd.DataFrame({'SeriesInstanceUID': self.indices})
        
        # Merge to get the weights corresponding to self.indices.
        # 'how='left'' ensures all indices in self.indices are kept.
        df_merged = df_indices.merge(df_weights, on='SeriesInstanceUID', how='left')
        
        # Identify samples that are in self.indices but not in the weight file (NaN after merge).
        missing_weights = df_merged[weight_column].isna()
        
        if missing_weights.any():
            # Print a warning and the count of missing samples.
            print(f"Warning: {missing_weights.sum()} samples are missing in the weight file, filling with the average weight.")
        
        # Calculate the average of existing (non-NaN) weights.
        existing_weights = df_merged[weight_column].dropna()
        if len(existing_weights) > 0:
            avg_weight = existing_weights.mean()
        else:
            # If no existing weights were found, default the average to 1.0.
            avg_weight = 1.0
            
        # Fill missing values (NaN) with the calculated average weight.
        df_merged[weight_column] = df_merged[weight_column].fillna(avg_weight)
        
        # Extract the final sample weights as a NumPy array.
        sample_weights = df_merged[weight_column].values
        
        # Normalize the weights so they sum to 1 (convert to a probability distribution).
        total_weight = np.sum(sample_weights)
        if total_weight > 0:
            return sample_weights / total_weight
        else:
            # Fallback: if total weight is 0, return a uniform distribution.
            n_samples = len(self.indices)
            return np.ones(n_samples) / n_samples

    def get_indices(self):
        return np.random.choice(self.indices, self.batch_size, replace=True, p=self.sampling_probabilities)
    
    def generate_train_batch(self):
        # 1. Select the unique keys (identifiers) for the current batch.
        selected_keys = self.get_indices()

        # 2. Preallocate memory for the main 3D data and segmentation.
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []
        
        # Preallocate memory for the 1D classification and metadata tasks.
        age_all = np.zeros(self.batch_size, dtype=np.int32)
        sex_all = np.zeros(self.batch_size, dtype=np.int32)
        modality_all = np.zeros(self.batch_size, dtype=np.int32)
        cls_task1_all = np.zeros(self.batch_size, dtype=np.int32) # Classification Task 1 (e.g., single binary label)
        # Classification Task 2 (e.g., multi-label, where columns represent different vessels)
        cls_task2_all = np.zeros((self.batch_size, len(self.vessel_columns)), dtype=np.int32)

        for j, i in enumerate(selected_keys):
            # oversampling foreground logic: determines if the patch selection should be biased
            # to include foreground pixels/labels (e.g., to sample more lesions).
            force_fg = self.get_do_oversample(j)

            # Load the 3D data, segmentation, and properties for the current case 'i'.
            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)

            # --- Extract Metadata from CSV for Classification Tasks ---
            # Retrieve the single row of metadata corresponding to the current SeriesInstanceUID 'i'.
            sample_data = self.df_full[self.df_full['SeriesInstanceUID'] == i].iloc[0]
            
            # Process Age (cast to integer)
            age_all[j] = int(sample_data['PatientAge'])
            
            # Process Sex (map string to integer code)
            sex_str = sample_data['PatientSex']
            sex_all[j] = self.sex_mapping.get(sex_str, 0)  # Default to 0 if mapping fails
            
            # Process Modality (map string to integer code)
            modality_str = sample_data['Modality']
            modality_all[j] = self.modality_mapping.get(modality_str, 0)  # Default to 0 if mapping fails
            
            # Process cls_task1 (e.g., Aneurysm Presence)
            cls_task1_all[j] = int(sample_data[self.aneurysm_column])
            
            # Process cls_task2 (e.g., Classification by Vessel Category)
            for k, vessel_col in enumerate(self.vessel_columns):
                cls_task2_all[j, k] = int(sample_data[vessel_col])
                
            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnXNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=-1)

        return {
            'data': data_all, 
            'seg': seg_all, 
            'properties': case_properties, 
            'keys': selected_keys,
            'age': age_all,
            'sex': sex_all,
            'modality': modality_all,
            'cls_task1': cls_task1_all,
            'cls_task2': cls_task2_all
        }

if __name__ == '__main__':
    folder = '/yinghepool/shipengcheng/Dataset/nnUNet/nnXNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnXNetDataset(folder, 0)  # this should not load the properties!
    dl = nnXNetDataLoader3DWithGlobalCls(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)