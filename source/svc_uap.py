import numpy as np
from sklearn import svm, preprocessing
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVR
from numpy import linalg as LA


def compute_cer(X, y, c=1.0, n_samples=4):
    """Train the classifier with the given data and compute the classification
       error rate (CER) for the given data with the trained SVM.
       It can also plot the results of the classification.
    """
    scaler = preprocessing.MinMaxScaler()
    X_n = scaler.fit_transform(X)

    clsf = svm.SVC(C=c, kernel='linear', class_weight='balanced')

    clsf.fit(X_n, y)
    y_p = clsf.predict(X_n)
    err1 = float(np.sum(np.absolute(y[0:(len(y) - n_samples)]
                                    - y_p[0:(len(y_p) - n_samples)]))) / \
           float((len(y) - n_samples))
    err2 = float(np.sum(np.absolute(y[-n_samples:] - y_p[-n_samples:]))) / \
           float(n_samples)
    cer = (err1 + err2) / 2

    return cer


def svc_rp(data, init_n_samples=4, n_samples=4, c=1.0, th=0.5, rp_th=10.0):
    """Classifies the given data and splits or groups the data depending on how
       tight the classification is. The score associated to each proposal
       follows an exponential function of the classification error.
       We implement rank_pooling filtering to discard bg proposals using the
       threshold rp_th
    """
    state = 0
    idx = 0
    proposals = np.zeros((1, 2), dtype=int)
    scores = np.zeros(1)
    p = 0

    if data.shape[0] < (init_n_samples + n_samples):  # just one proposal
        proposals[0, 0] = 0
        proposals[0, 1] = data.shape[0]
        scores[0] = 0.  # automatic score of 0

        if rank_pooling_filter(data, rp_th):  # all the data must be filtered
            proposals = np.delete(proposals, 0, 0)
            scores = np.delete(scores, 0, 0)
    else:
        while idx < data.shape[0]:
            if state == 0:  # initial state: initialize the proposal
                proposals[p, 0] = idx
                idx = init_n_samples + n_samples
                X = data[0:idx]
                y = np.concatenate((np.zeros(init_n_samples, dtype=np.int),
                                    np.ones(n_samples, dtype=np.int)))
            elif state == 1:  # proposal grows
                X = np.concatenate((X, data[idx:idx + n_samples]))
                y = np.concatenate((np.zeros(X[0:-n_samples].shape[0],
                                             dtype=np.int),
                                    np.ones(n_samples, dtype=np.int)))
                idx = idx + n_samples
            elif state == 2:  # generate proposal

                proposals[p, 1] = idx - 1  # save proposal last index
                scores[p] = np.exp(-cer)  # assing an score

                # filter proposal
                if rank_pooling_filter(data[proposals[p,0]:proposals[p,1]],
                                       rp_th):
                    proposals = np.delete(proposals, p, 0)
                    scores = np.delete(scores, p, 0)
                else:
                    p = p + 1  # increment proposal index

                proposals = np.concatenate((proposals, np.zeros((1, 2),
                                                                dtype=int)))
                # initialize next proposal
                proposals[p, 0] = idx
                # prepare for the next score to be filled
                scores = np.concatenate((scores, np.zeros(1)))

                X = np.concatenate((X[-n_samples:],
                                    data[idx:(idx + init_n_samples)]))
                y = np.concatenate((np.zeros(X[0:-n_samples].shape[0],
                                             dtype=np.int),
                                    np.ones(n_samples, dtype=np.int)))
                idx = idx + init_n_samples

            cer = compute_cer(X, y, c, n_samples)
            if cer >= th:  # compare with the threshold
                state = 1  # grow current proposal
            else:
                state = 2  # create proposal

        #Generate the last proposal for the video
        proposals[p, 1] = data.shape[0] - 1  # End of video
        scores[p] = 0.
        # filter proposal
        if rank_pooling_filter(data[proposals[p,0]:proposals[p,1]], rp_th):
            proposals = np.delete(proposals, p, 0)
            scores = np.delete(scores, p, 0)

    return proposals, scores


def rank_pooling_filter(X, rp_th):

    filter_status = True  #initialization, by default: no action detected

    #Is it an empty proposal of just one frame?
    if X.size == 0:
        return filter_status

    #create an array as a random version of the original vectors
    X_rand = np.random.permutation(X)

    # Obtaining dynamics with Rank Pooling
    c = 1

    D_X = rank_pooling(X, c, 'non-lin','L2')
    D_X_rand = rank_pooling(X_rand, c, 'non-lin','L2')

    d = LA.norm(D_X - D_X_rand)

    if d > rp_th:
        filter_status = False  #action detected

    return filter_status


def get_non_linearity(x):

        y = np.multiply(np.sign(x), np.sqrt(np.absolute(x)))

        return y


def rank_pooling(data, cval, rank_type, norm):

        # Apply smooth
        t_index = np.transpose(np.tile(np.arange(1, data.shape[0] + 1), (data.shape[1], 1)))
        v_smooth = np.divide(np.cumsum(data, axis=0, dtype=float), t_index)

        # Apply non linearity to data if required
        if rank_type == 'non-lin':
            v_smooth = get_non_linearity(v_smooth)

        # Apply normalization to data if required
        if norm == 'L2':
            v_smooth_norm = normalize(v_smooth, norm='l2', axis=1)
        else:
            v_smooth_norm = v_smooth

        # Train a linear svm with liblinear and obtain the u vectors (rank pooling paper)
        labels = np.arange(0, v_smooth.shape[0])
        regrss = LinearSVR(C=cval)
        regrss.fit(v_smooth_norm, labels)
        u = regrss.coef_   #dynamics

        return u
