import numpy as np
import json
import time


def init_log(subset, init_n_samples, n_samples, th, c, rp_th, log_file):

    with open(log_file, 'w') as lf:
        lf.write(f'Execution initiated on {time.ctime()} \n\n')
        lf.write('Parameter configuration for this execution:\n\n')
        lf.write(f'subset: {subset}\n'
                 f'init_n_samples: {init_n_samples}\n'
                 f'n_samples: {n_samples}\n'
                 f'th: {th}\n'
                 f'c: {c}\n'
                 f'rp th: {rp_th}')
        lf.write(f'\n\nAnalysed videos:\n\n')
        lf.write(f'Video idx'.ljust(15))
        lf.write(f'Video name'.ljust(15))
        lf.write(f'Number of proposals'.ljust(20) + '\n')

    return


def frame_to_time(proposals, fps, dataset):
    """Since C3D feature vectors contain 16 frames with a window of 8 frames,
       the number of frames analysed until each vector will be  8*n + 8.
       Therefore the time passed until that frame will be (8*n + 8)/fps.
    """
    temporal_proposals = np.empty(proposals.shape)
    for i in range(0, proposals.shape[0]):
        for j in range(0, proposals.shape[1]):
            if dataset == 'ActivityNet':
                temporal_proposals[i, j] = (8 * proposals[i, j] + 8) / fps
            else:
                temporal_proposals[i, j] = (16 * proposals[i, j]) / fps

    return temporal_proposals


def fill_log_file(iter, video_name, proposals, finished, log_file):

    with open(log_file, 'a') as lf:
        if finished == 0:
            lf.write('\n')
            lf.write(str(iter).ljust(15))
            lf.write(video_name.ljust(15))
            lf.write(str(proposals).ljust(20))
        elif finished == 1:
            lf.write('\n\n' + 'Total videos analysed: ' + str(iter))

    return


def write_res(proposal_data, json_results):

    with open(json_results, 'w') as fobj:
        json.dump(proposal_data, fobj)

    return
