import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join
matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class nnXNetLogger(object):
    """
    This class is really trivial. Don't expect cool functionality here. This is my makeshift solution to problems
    arising from out-of-sync epoch numbers and numbers of logged loss values. It also simplifies the trainer class a
    little

    YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP
    """
    def __init__(self, verbose: bool = False, num_cls_task: int = 2):
        self.my_fantastic_logging = {
            'mean_fg_dice': list(),
            'mean_fg_dice_1': list(),
            'mean_fg_dice_2': list(),
            'ema_fg_dice': list(),
            'ema_fg_dice_1': list(),
            'ema_fg_dice_2': list(),
            'mean_auc': list(),  # Case-level mean AUC
            'mean_patch_auc': list(),  # Patch-level mean AUC
            'ema_auc': list(),
            'ema_patch_auc': list(),  # EMA for patch-level AUC
            'dice_per_class_or_region': list(),
            'dice_per_class_or_region_1': list(),
            'dice_per_class_or_region_2': list(),
            'total_cls_train_losses': list(),
            'total_cls_val_losses': list(),
            'train_losses': list(),
            'val_losses': list(),
            'val_patch_loss': list(),  # Total patch-level validation loss
            'val_case_loss': list(),  # Total case-level validation loss
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list()
        }
        
        for t_index in range(num_cls_task):
            self.my_fantastic_logging[f'cls_task_{t_index}_acc'] = list()  # Case-level accuracy
            self.my_fantastic_logging[f'cls_task_{t_index}_auc'] = list()  # Case-level AUC
            self.my_fantastic_logging[f'cls_task_{t_index}_loss'] = list()  # Case-level loss
            self.my_fantastic_logging[f'cls_task_{t_index}_patch_acc'] = list()  # Patch-level accuracy
            self.my_fantastic_logging[f'cls_task_{t_index}_patch_auc'] = list()  # Patch-level AUC
            self.my_fantastic_logging[f'cls_task_{t_index}_patch_loss'] = list()  # Patch-level loss

        self.my_fantastic_logging[f'cls_modality_acc'] = list()
        self.my_fantastic_logging[f'cls_modality_auc'] = list()
        self.my_fantastic_logging[f'cls_modality_loss'] = list() 

        self.verbose = verbose
        # shut up, this logging is great

    def log(self, key, value, epoch: int):
        """
        Sometimes shit gets messed up. We try to catch that here
        """
        assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
            f'This function is only intended to log stuff to lists and to have one entry per epoch. Unknown key: {key}'

        if self.verbose:
            print(f'logging {key}: {value} for epoch {epoch}')

        if len(self.my_fantastic_logging[key]) < (epoch + 1):
            self.my_fantastic_logging[key].append(value)
        else:
            assert len(self.my_fantastic_logging[key]) == (epoch + 1), \
                f'something went horribly wrong. My logging lists length is off by more than 1 for key {key}'
            print(f'maybe some logging issue!? logging {key} and {value}')
            self.my_fantastic_logging[key][epoch] = value

        # Handle EMA for mean_auc
        if key == 'mean_auc':
            new_ema_auc = self.my_fantastic_logging['ema_auc'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['ema_auc']) > 0 else value
            self.log('ema_auc', new_ema_auc, epoch)

        # Handle EMA for mean_patch_auc
        if key == 'mean_patch_auc':
            new_ema_patch_auc = self.my_fantastic_logging['ema_patch_auc'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['ema_patch_auc']) > 0 else value
            self.log('ema_patch_auc', new_ema_patch_auc, epoch)

        # Handle EMA for mean_fg_dice
        if key == 'mean_fg_dice':
            new_ema_pseudo_dice = self.my_fantastic_logging['ema_fg_dice'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['ema_fg_dice']) > 0 else value
            self.log('ema_fg_dice', new_ema_pseudo_dice, epoch)
        
        if key == 'mean_fg_dice_1':
            new_ema_pseudo_dice = self.my_fantastic_logging['ema_fg_dice_1'][-1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['ema_fg_dice_1']) > 0 else value
            self.log('ema_fg_dice_1', new_ema_pseudo_dice, epoch)
        
        if key == 'mean_fg_dice_2':
            new_ema_pseudo_dice = self.my_fantastic_logging['ema_fg_dice_2'][-1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['ema_fg_dice_2']) > 0 else value
            self.log('ema_fg_dice_2', new_ema_pseudo_dice, epoch)

    def plot_progress_png(self, output_folder):
        # We infer the epoch from our internal logging
        epoch = min([len(i) for i in self.my_fantastic_logging.values() if len(i) > 0]) - 1  # Lists of epoch 0 have len 1
        
        # Check if we have at least one epoch of data
        if epoch < 0:
            print("No data available for plotting")
            return
        
        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(4, 1, figsize=(30, 72))

        # Plot 1: Losses (train, val, patch, case)
        ax = ax_all[0]
        x_values = list(range(epoch + 1))
        
        # Track if we have any data to plot for legend
        has_legend_data = False
        
        # Check and plot each loss metric only if it has sufficient data
        if len(self.my_fantastic_logging['train_losses']) >= epoch + 1:
            ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', ls='-', label="train_loss", linewidth=4)
            has_legend_data = True
        
        if len(self.my_fantastic_logging['val_losses']) >= epoch + 1:
            ax.plot(x_values, self.my_fantastic_logging['val_losses'][:epoch + 1], color='r', ls='-', label="val_loss", linewidth=4)
            has_legend_data = True
        
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        if has_legend_data:
            ax.legend(loc=(0, 1))

        # Plot 2: AUC (case and patch)
        ax = ax_all[1]
        ax2 = ax.twinx()
        
        has_legend_data_ax = False
        has_legend_data_ax2 = False
        
        if 'mean_auc' in self.my_fantastic_logging and 'mean_fg_dice' in self.my_fantastic_logging:
            # print("len(self.my_fantastic_logging['mean_auc']): ", len(self.my_fantastic_logging['mean_auc']))
            # print("epoch + 1: ", epoch + 1)
            if len(self.my_fantastic_logging['mean_auc']) >= epoch + 1:
                ax.plot(x_values, self.my_fantastic_logging['mean_auc'][:epoch + 1], color='g', ls='dotted', label="case_auc", linewidth=3)
                has_legend_data_ax = True
            
            if len(self.my_fantastic_logging['ema_auc']) >= epoch + 1:
                ax.plot(x_values, self.my_fantastic_logging['ema_auc'][:epoch + 1], color='g', ls='-', label="case_auc (mov. avg.)", linewidth=4)
                has_legend_data_ax = True
            
            if len(self.my_fantastic_logging['mean_patch_auc']) >= epoch + 1:
                ax.plot(x_values, self.my_fantastic_logging['mean_patch_auc'][:epoch + 1], color='y', ls='dotted', label="patch_auc", linewidth=3)
                has_legend_data_ax = True
            
            if len(self.my_fantastic_logging['ema_patch_auc']) >= epoch + 1:
                ax.plot(x_values, self.my_fantastic_logging['ema_patch_auc'][:epoch + 1], color='y', ls='-', label="patch_auc (mov. avg.)", linewidth=4)
                has_legend_data_ax = True
            
            ax.set_ylabel("AUC")
            
            if has_legend_data_ax:
                ax.legend(loc=(0, 1))
            
            if len(self.my_fantastic_logging['mean_fg_dice']) >= epoch + 1:
                ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='dotted', label="pseudo_dice", linewidth=3)
                has_legend_data_ax2 = True
            
            if len(self.my_fantastic_logging['ema_fg_dice']) >= epoch + 1:
                ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1], color='g', ls='-', label="pseudo_dice (mov. avg.)", linewidth=4)
                has_legend_data_ax2 = True
            
            ax2.set_ylabel("pseudo dice")
            if has_legend_data_ax2:
                ax2.legend(loc=(0.2, 1))

        elif 'mean_auc' in self.my_fantastic_logging:
            # print("len(self.my_fantastic_logging['mean_auc']): ", len(self.my_fantastic_logging['mean_auc']))
            # print("epoch + 1: ", epoch + 1)
            if len(self.my_fantastic_logging['mean_auc']) >= epoch + 1:
                ax.plot(x_values, self.my_fantastic_logging['mean_auc'][:epoch + 1], color='g', ls='dotted', label="case_auc", linewidth=3)
                has_legend_data_ax = True
            
            if len(self.my_fantastic_logging['ema_auc']) >= epoch + 1:
                ax.plot(x_values, self.my_fantastic_logging['ema_auc'][:epoch + 1], color='g', ls='-', label="case_auc (mov. avg.)", linewidth=4)
                has_legend_data_ax = True
            
            if len(self.my_fantastic_logging['mean_patch_auc']) >= epoch + 1:
                ax.plot(x_values, self.my_fantastic_logging['mean_patch_auc'][:epoch + 1], color='y', ls='dotted', label="patch_auc", linewidth=3)
                has_legend_data_ax = True
            
            if len(self.my_fantastic_logging['ema_patch_auc']) >= epoch + 1:
                ax.plot(x_values, self.my_fantastic_logging['ema_patch_auc'][:epoch + 1], color='y', ls='-', label="patch_auc (mov. avg.)", linewidth=4)
                has_legend_data_ax = True
            
            ax.set_ylabel("AUC")
            if has_legend_data_ax:
                ax.legend(loc=(0, 1))
        else:
            if len(self.my_fantastic_logging['mean_fg_dice']) >= epoch + 1:
                ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='dotted', label="pseudo_dice", linewidth=3)
                has_legend_data_ax2 = True
            
            if len(self.my_fantastic_logging['ema_fg_dice']) >= epoch + 1:
                ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1], color='g', ls='-', label="pseudo_dice (mov. avg.)", linewidth=4)
                has_legend_data_ax2 = True
            
            ax2.set_ylabel("pseudo dice")
            if has_legend_data_ax2:
                ax2.legend(loc=(0.2, 1))
        
        ax.set_xlabel("epoch")

        # Plot 3: Epoch duration
        ax = ax_all[2]
        has_legend_data = False
        
        if (len(self.my_fantastic_logging['epoch_end_timestamps']) >= epoch + 1 and 
            len(self.my_fantastic_logging['epoch_start_timestamps']) >= epoch + 1):
            
            durations = [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                                            self.my_fantastic_logging['epoch_start_timestamps'][:epoch + 1])]
            ax.plot(x_values, durations, color='b', ls='-', label="epoch_duration", linewidth=4)
            ylim = [0] + [ax.get_ylim()[1]]
            ax.set(ylim=ylim)
            has_legend_data = True
        
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        if has_legend_data:
            ax.legend(loc=(0, 1))

        # Plot 4: Learning rate
        ax = ax_all[3]
        has_legend_data = False
        
        if len(self.my_fantastic_logging['lrs']) >= epoch + 1:
            ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label="learning_rate", linewidth=4)
            has_legend_data = True
        
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        if has_legend_data:
            ax.legend(loc=(0, 1))

        plt.tight_layout()
        fig.savefig(join(output_folder, "progress.png"))
        plt.close()

    def get_checkpoint(self):
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        self.my_fantastic_logging = checkpoint