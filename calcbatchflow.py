from optflowmods.flowgui import FlowGUI
from optflowmods.flowprocessor import FlowProcessor
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import os

def main():
    gui = FlowGUI()
    gui.mainloop()

    folder_path = gui.folder_path
    window_size = gui.window_size
    polyN_size = gui.polyN_size
    polyS_size = gui.polyS_size
    frames_to_skip = gui.frame_skip_num
    vectors_to_skip = gui.vector_skip_num
    gauss_sigma = gui.gauss_sigma

    if len(folder_path) < 1 :
        print('Please enter a directory to process')
        sys.exit()

    #make dictionary of parameters for log file use
    log_params = {  'folder_path': folder_path,
                    'window_size': window_size,
                    'polyN_size': polyN_size,
                    'polyS_size': polyS_size,
                    'frames_to_skip': frames_to_skip,
                    'vectors_to_skip': vectors_to_skip,
                    'gauss_sigma': gauss_sigma,
                    'files_processed' : [],
                    "Files Not Processed" : []
                } 

    ims = [im_name for im_name in os.listdir(folder_path) if im_name.endswith('.tif') and not im_name.startswith('.')]

    # shared save path
    save_path = os.path.join(folder_path, 'opt_flow_summary')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # lists to fill with summary values
    summary_list = []
    col_headers = []

    with tqdm(total = len(ims)) as pbar:
        pbar.set_description("Processing Images")
        for im_name in ims:
            print(f'Processing {im_name}')
            log_params['files_processed'].append(im_name)
            fp = FlowProcessor( image_path = os.path.join(folder_path, im_name),
                                win_size = window_size,
                                polyN = polyN_size,
                                polyS = polyS_size,
                                frame_skip = frames_to_skip,
                                vect_skip = vectors_to_skip,
                                gauss_sigma = gauss_sigma)

            print('flow processor created')

            # log error and skip image if frames < 2 
            if fp.num_frames < 2:
                print(f"****** ERROR ******",
                    f"\n{im_name} has less than 2 frames",
                    "\n****** ERROR ******")
                log_params['Files Not Processed'].append(f'{im_name} has less than 2 frames')
                pbar.update(1)
                continue

            if window_size*2 > fp.image.shape[-1] or window_size*2 > fp.image.shape[-2]:
                print(f"****** ERROR ******",
                    "Your window is too large for the image",
                    "\n****** ERROR ******")
                log_params['Files Not Processed'].append(f'{im_name} Window too large for image size')
                pbar.update(1)
                continue

            print('sanity checks complete')

            # calculate flow and summary
            print('calculating flow')
            fp.calc_mean_flow()
            mags_array, masked_mags_array = fp.calc_regional_flow()
            with tqdm(total = 10, desc = 'cleaning up') as pbar2:
                for i in range(10):
                    pbar2.update(1)
            print('plotting summary')
            plots = fp.plot_summary()

            # save summary to file
            print('creating folders')
            file_save_path = os.path.join(save_path, im_name.rsplit(".",1)[0])
            if not os.path.exists(file_save_path):
                os.makedirs(file_save_path)
            
            # append summary stats to summary list
            file_summary = {}
            file_summary['file_name'] = im_name
            
            for j in range(fp.num_channels):
                print('saving channel', j+1)
                print('calculating total flow statistics')
                total_flow = mags_array[:,j,:,:].ravel()
                total_mean_flow = np.mean(total_flow)
                total_std_flow = np.std(total_flow)
                print('calculating regional flow statistics')
                total_flow_masked = np.concatenate(masked_mags_array[f'Ch{j + 1}'])
                total_mean_flow_masked = np.mean(total_flow_masked)
                total_std_flow_masked = np.std(total_flow_masked)
                ch_summary = plots[f'Ch{j + 1}']

                # save text and summary files
                print('saving channel summary')
                ch_summary.savefig(os.path.join(file_save_path, f'ch{j + 1}_summary.png'))

                # save summary stats for current channel
                file_summary[f'Ch{j+1} mean total flow (px/frame)'] = total_mean_flow
                file_summary[f'Ch{j+1} std total flow (px/frame)'] = total_std_flow
                file_summary[f'Ch{j+1} mean masked total flow (px/frame)'] = total_mean_flow_masked
                file_summary[f'Ch{j+1} std masked total flow (px/frame)'] = total_std_flow_masked

                for key in file_summary.keys():
                    if key not in col_headers:
                        col_headers.append(key)

            # append summary stats to summary list
            summary_list.append(file_summary)
            
            pbar.update(1)
    
    # save log file
    print('saving log file')
    with open(os.path.join(save_path, 'log.txt'), 'w') as f:
        for key, value in log_params.items():
            f.write(f'{key}: {value}\n')
    
    # generate a summary dataframe
    print('generating summary dataframe')
    summary_df = pd.DataFrame(summary_list, columns = col_headers)
    summary_df.to_csv(os.path.join(save_path, 'summary.csv'), index = False)

if __name__ == '__main__':
    main()