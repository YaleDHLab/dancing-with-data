# Note: This repository has been archived
This project was developed under a previous phase of the Yale Digital Humanities Lab. Now a part of Yale Library’s Computational Methods and Data department, the Lab no longer includes this project in its scope of work. As such, it will receive no further updates.


# Dancing with Data

This repository contains source code for generating and visualizing 3D dance sequences with neural networks.

# Requirements

The python dependencies for this codebase can be installed with:

```bash
pip install -r requirements.txt
```

If you are working with a CUDA-enabled GPU, you will need to install the GPU version of tensorflow:

```bash
pip install tensorflow-gpu==1.11.0
```

# Data Preparation

This codebase intends to parse 3D data in a particular format. Specifically, the input data should be a numpy array with the following shape:

```bash
X[time_index][body_part_index][coordinate_index]
```

In other words, `X.shape` will return three values. The first indicates the number of time intervals captured in the data. The second indicates the number of body parts tracked. The third contains three values for the x, y, and z positional coordinates of the `ith` body part, in that order.

### Preparing Data from Shogun Live

The original data files captured for this analysis were produced by Shogun Live, software that tracks the positions of markers in a motion capture studio. Shogun Live outputs a file with .fbx format. To convert that .fbx format to a .npy array with the appropriate shape, we used the following Rube Goldberg setup:

.fbx to .glb to .json to .npy

#### `.fbx -> .glb`

First we convert from .fbx to .glb. To do so, one can install Facebook's [FBX2glTF](https://github.com/facebookincubator/FBX2glTF) binary, and run it on an .fbx file:

```bash
FBX2glTF -i input_name.fbx -o output_name -b
```

A sample .fbx file may be found in `./data/fbx/dance.fbx`.

#### `.glb -> .json`

Next we convert from .glb to .json. To do so, one can start a local web server in `./data`:

```bash
cd dancing-with-robots/data

# python 3
python -m http.server 7091

# python 2
python -m http.server 7091

# open the web page
open "http://localhost:7091"
```

Once your web browser is on `localhost:7091`, you should see a little web page with some dancing figures. If you don't see dancing figures, look at `./data/index.html` and make sure the .glb filename and path match the path to your local .glb file. If you do see dancing figures, you should see a download button at the top left of the screen. Clicking that will download the .json file.

#### `.json -> .npy`

Finally we convert from .json to .npy. To do so, one can run:

```bash
cd dancing-with-robots/data

python format_inputs.py
```

This will save a file `./data/npy/dance.npy` that can be fed into the `./model.ipynb`

# Running the Model

Once the data is ready (samples are included in this repository), one can run the model by starting the Jupyter notebook:

```bash
jupyter notebook model.ipynb
```

Once the notebook starts, click `Cell -> Run All` and the model will try to learn to dance.
