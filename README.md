# SVC for Unsupervised Temporal Action Proposals
This repository contains the code of the SVC model for unsupervised Temporal Action Proposals. The pipeline was presented [here](https://www.mdpi.com/1424-8220/20/10/2953/htm).

<p align="center">
  <img src="./png/svc-uap.png" alt="Unsupervised Temporal Action Proposals" title="Unsupervised Temporal Action Proposals with SVC" width="652" zoom="343" align="center" />
</p>

### Citation

If you find anything of this repository useful for your projects, please consider citing this work:

```bibtex
@article{Baptista2020svc,
	author  = {M. {Baptista R\'ios} and R. J. {L\'opez-Sastre} and F. J. {Acevedo-Rodr\'iguez} and P. {Mart\'in-Mart\'in} and S. {Maldonado-Basc\'on}},
	journal = {Sensors},
	title   = {Unsupervised Action Proposals Using Support Vector Classifiers for Online Video Processing},
	year	= {2020},
	volume  = {20},
	doi     = {10.3390/s20102953},
}
```

#### What can you find?

The project we release here contains the code with the SVC-UAP model and the best parameter configuration  ActivityNet1.3 and Thumos14 datasets.

#### Preparing the data

Before you start, please be sure to prepare the data as described below.

######  ActivityNet1.3

- Ground Truth:

  The code accepts ground truth files with `.json` format following ActivityNet style.

  For ActivityNet1.3, its annotations can be downloaded from [here](http://activity-net.org/download.html). Once you have them, remember to put the `.json` file in the directory: `.gt/` 

  Additionally,  we provide the following command to directly download and put them in the right directory.

```bash
wget -O gt/activity_net.v1-3.min.json http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json
```

- Features:

  For ActivityNet1.3, we have extracted C3D features pretrained on Sports1M dataset. Concretely, we used feature vectors from fc6 layer, with dimension 4096, corresponding each of them to a 16-frame volume. In each volume, 8 frames are overlapped. Afterwards, feature vectors are reduced to 500d with PCA.

  Features can be downloaded from `?`.

  Once you have them, store them as: `.h5/c3d-activitynet.hdf5` 

###### Thumos'14

- Ground Truth:

  For Thumos'14, we have converted its original annotations to the ActivityNet format in a `.json` file. We share itwith this repo at `gt/gt-thumos14.json`.

- Features:

  We have extracted features with a C3D pretrained on Sports1M dataset. Concretely, we used feature vectors from fc6 layer, with dimension 4096, corresponding each of them to a non-overlapped 16-frame volume.

  Features can be downloaded from `?`.

  Once you have them, store them as: `.h5/c3d-thumos14.hdf5` 

#### Usage

*(The code has been prepared to be run with Python 3.7. Therefore, we recommend to use the same version. Check also `requirements.txt` file to see the list of needed packages.)*

Once you have prepared the data, change the current directory to `source/`. Running `main.py` will start the whole svc-uap. However, remember to configure the following options:

- `-d`: name of the dataset.
- `-gt`: path to the ground truth file.
- `-h5`: path to the features file.
- `-set`: dataset subset.
- `-init_n`: initial number of samples to take when starting the algorithm or a new proposal.
- `-n`: number of new samples to take when analysing the same proposal.
- `-th`: classification error rate.
- `-c`: C parameter of the Linear SVM.
- `-rpth`: rank pooling threshold.
- `-res`: `.json` file with results.
- `-eval`: to set only evaluation mode.
- `-log`: log file with execution information.
- `-fig`: figure with th AR-AN metric.

If you just run the following command:

```bash
python main.py
```

you will get the results for the best parameter configuration for ActivityNet dataset.

On the other hand, if you want to reproduce the results of the best configuration for the Thumos'14 dataset, you should do as follows:

```bash
python main.py -data Thumos14 -gt ../gt/gt-thumos14.json -h5 ../h5/c3d-thumos14.hdf5 -set Test -init_n 8 -n 8 -th 0.09 -c 0.019306 -rpth 0.1
```

#### Results

The code is prepared to provide at the end of execution the performance of the SVC-UAP method in terms of the AR-AN metric (Average Recall - Average Number of proposals). To this end, the folder directory `./res` (in the project directory) and you will find:

- `ar-an.png`: figure with the AR-AN metric.
- `svc-uap-res.json`: result file with proposals.

In case you wish to evaluate again the results, you can run again the project but setting the `-eval` option to True. The code will just take the previously obtained `.json` result file with the proposals and evaluate it.