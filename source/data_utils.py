import json
import h5py


def save_video_names(name):

    video_names = []
    video_names.append(name)
    print(video_names)

    return video_names


def get_videos(gt, set):

    jf = open(gt, 'r')
    parsedjf = json.load(jf)
    video_names = list()
    for key in parsedjf['database']:
        if parsedjf['database'][key]['subset'] == set:
            video_names.append(key)
        elif parsedjf['database'][key]['subset'] == set:
            video_names.append(key)
        elif parsedjf['database'][key]['subset'] == set:
            video_names.append(key)

    return video_names


def get_vid_info(video, gt, h5, dataset='ActivityNet'):

    jf = open(gt, 'r')
    parsedjf = json.load(jf)
    h5f = h5py.File(h5, "r")

    if dataset == 'ActivityNet':
        d = h5f['/v_' + video + '/c3d_features'][()]
        duration = parsedjf['database'][video]['duration']
        fps = (8 * d.shape[0] + 8) / duration
    else:
        d = h5f[video + '/c3d_features'].value
        fps = parsedjf['database'][video]['fps']

    return d, fps
