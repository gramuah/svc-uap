#!/usr/bin/python
import argparse
import os
import time
from svc_uap import svc_rp
from proposals_utils import init_log, frame_to_time, fill_log_file, write_res
from data_utils import get_videos, get_vid_info
from evaluation_utils import run_evaluation, plot_metric


def parse_input_arguments():

    description = 'SVC-UAP. Unsupervised Temporal Action Proposals.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('-data', choices=['ActivityNet', 'Thumos14'],
                   default='ActivityNet', help='Dataset')
    p.add_argument('-gt', default='../gt/activity_net.v1-3.min.json',
                   help='Ground truth file in .json format.')
    p.add_argument('-h5', default='../h5/c3d-activitynet.hdf5',
                   help='Features in .hdf5 file.')
    p.add_argument('-set', default='validation',
                   choices=['training', 'validation', 'testing', 'Test'],
                   help='Dataset subset.')
    p.add_argument('-init_n', type=int, default=256,
                   help='Initial number of samples to take when starting the '
                        'algorithm or a new proposal.')
    p.add_argument('-n', type=int, default=256,
                   help='Number of new samples to take when analysing the same'
                        'proposal.')
    p.add_argument('-th', type=float, default=0.1,
                   help='Classification error rate threshold.')
    p.add_argument('-c', type=float, default= 1.1787e-5,
                   help='C parameter of the of the Linear SVM.')
    p.add_argument('-rpth', type=float, default=1,
                   help='Rank-pooling threshold.')
    p.add_argument('-res', default='../res/svc-uap-res.json',
                   help='Proposals result.')
    p.add_argument('-eval', action='store_false',
                   help='Use this variable if only evaluation is needed.')
    p.add_argument('-log', default='../log/svc-uap.log',
                   help='Log file with execution information.')
    p.add_argument('-fig', default='../res/ar-an.png',
                   help='Figure with AR-AN metric.')

    return p.parse_args()


def svc_uap(data, vid_names, gt, h5, init_n, n, th, c, log, rp_th):

    ppsals_data = {'results': {},
                   'version': 'VERSION 1.3',
                   'external_data': {},
                   }

    for idx, vid in enumerate(vid_names):
        print(f'Processing video {idx} / {len(vid_names)} ...')

        d, fps = get_vid_info(vid, gt, h5, data)

        ppsals, score = svc_rp(d, init_n, n, c, th, rp_th)

        temp_ppsals = frame_to_time(ppsals, fps, data)

        this_vid_ppsals = []
        for i in range(0, temp_ppsals.shape[0]):
            ppsal = {'score': score[i],
                     'segment': [temp_ppsals[i, 0], temp_ppsals[i, 1]],
                     }
            this_vid_ppsals += [ppsal]

        ppsals_data['results'][vid] = this_vid_ppsals
        if idx == len(vid_names) - 1:
            fill_log_file(idx+1, vid, temp_ppsals.shape[0], 1, log)
        else:
            fill_log_file(idx, vid, temp_ppsals.shape[0], 0, log)

    return ppsals_data


def main(data, gt, h5, set, init_n, n, th, c, rpth, res, eval, log, fig):

    if eval:
        if not os.path.exists('../log'):
            os.makedirs('../log')
        init_log(set, init_n, n, th, c, rpth, log)
        vid_names = get_videos(gt, set)

        print('Running svm clustering.')
        start_time = time.time()
        ppsals = svc_uap(data, vid_names, gt, h5, init_n, n, th, c, log, rpth)
        elapsed_time = time.time() - start_time
        print('Execution time in seconds = ' + str(elapsed_time))

        if not os.path.exists('../res'):
            os.makedirs('../res')
        write_res(ppsals, res)

    print('Running evaluation.')
    if not os.path.isfile(res):
        print(f'{res}: No such file or directory.')
        return
    else:
        average_n_proposals, average_recall, recall = run_evaluation(gt, res,
                                                                 subset=set)
        plot_metric(average_n_proposals, average_recall, recall, fig_file=fig)

    return


if __name__ == '__main__':
    args = parse_input_arguments()
    main(**vars(args))
