

![Neural3DMM architecture](images/architecture_figure1.png "Neural3DMM architecture")

# Project Abstract 
*Generative models for 3D geometric data arise in many important applications in 3D computer vision and graphics. In this paper, we focus on 3D deformable shapes that share a common topological structure, such as human faces and bodies. Morphable Models were among the first attempts to create compact representations for such shapes; despite their effectiveness and simplicity, such models have limited representation power due to their linear formulation. Recently, non-linear learnable methods have been proposed, although most of them resort to intermediate representations, such as 3D grids of voxels or 2D views. In this paper, we introduce a convolutional mesh autoencoder and a GAN architecture based on the spiral convolutional operator, acting directly on the mesh and leveraging its underlying geometric structure. We provide an analysis of our convolution operator and demonstrate state-of-the-art results on 3D shape datasets compared to the linear Morphable Model and the recently proposed COMA model.* 

[Paper Arxiv Link](https://arxiv.org/abs/1905.02876)


# Repository Requirements

This code was written using Pytorch 0.4.1, however runs with Pytorch 1.1 as well. We use tensorboardX for the training metrics. We recommend setting up a virtual environment using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and specifically with Python2 (we need Python 2 not for training but for the downsampling and upsampling, see below). To install Pytorch in a conda environment, simply run 

```
$ conda install pytorch torchvision -c pytorch
```

Then the rest of the requirements can be installed with 

```
$ pip install -r requirments.txt
```


### Downsampling & Upsampling
For the downsampling code we use a function from the [COMA repository](https://github.com/anuragranj/coma), specifically the files **mesh_sampling.py** and **shape_data.py** (previously **facemesh.py**) were taken from the COMA repo and adapted to our needs. These in turn use the [MPI-Mesh](https://github.com/MPI-IS/mesh) package (a mesh library similar to Trimesh or Open3D).  This package requires Python 2, which is why we recommend doing everything with Python 2. However once you have cached the generated downsampling and upsampling matrices, it is possible to run the training code with Python 3 as well if necessary. In order to install MPI-Mesh do the following (or read their installation instructions if something is unclear):

```
$ git clone https://github.com/MPI-IS/mesh.git
$ cd mesh
$ apt-get install libboost-all-dev
$ make
$ make install
```



# Data Organization

The following is the organization of the dataset directories expected by the code:

* data **root_dir**/
  * **dataset** name/ (eg DFAUST)
    * preprocessed/
      * train.npy (number_meshes, number_vertices, 3) (no Faces because they all share topology)
      * test.npy 
      * template.obj (all of the spiraling and downsampling code is run on the template only once)
      * templates/ (if **not** using COMA downsampling, eg we used Meshlab downsamplings)
        * downsample_method/
          * template_d0.obj (same as template.obj)
          * template_d1.obj
          * template_d2.obj
          * template_d3.obj
          * template_d4.obj
      * points_train/ (created by data_generation.py)
      * points_val/ (created by data_generation.py)
      * points_test/ (created by data_generation.py)
      * paths_train.npy (created by data_generation.py)
      * paths_val.npy (created by data_generation.py)
      * paths_test.npy (created by data_generation.py)

    * results/ (created by the code)
      * spiral_autoencoder/
          * latent_size/
            * checkpoints/ (has all of the pytorch models saved as well as optimizer state and epoch to continue training)
            * samples/ (has samples of reconstructions saved throughout training)
            * predictions/ (reconstructions on test set)
            * summaries/ (has all of the tensorboard files)

In order to display all of the Tensorboards for all of the models you have run, simply run from **root_dir**

```
$ tensorboard --logdir=results/
```

# Running the Code

#### First Data preprocessing 

```
$ python data_generation.py --root_dir=/path/to/data_root_dir --dataset=DFAUST --num_valid=100
```

#### Running training

The first time you run the code it will check if you have the downsamples cached (calculating the downsampling and upsampling matrices takes a few minutes), and then does the spiraling code on the template, which is in **spiral_utils.py**, afterwards beginning the training. The training is done in an ipython notebook, which you can run with 
```
$ jupyter notebook neural3dmm.ipynb
```

Where you can see the arguments for the training in a dictionary called **args** in the 2nd cell of the notebook. The first cell has metadata arguments that you need to fill in such as the data **root_dir** and the **dataset** name, whether you want to use the GPU, etc. 

Some important notes:
* The reference points parameter needs exactly one vertex index per disconnected component of the mesh. So for DFAUST you only need one, but for COMA which has the eyes as diconnected components, you need a reference point on the head as well as one on each eye
* The **spiral_utils.py** spiraling code works by walking along the triangulation exploiting the fact that the triangles are all listed in a consistent ordering (either clockwise or counter-clockwise) in order to get the spiral scan ordering for each neighborhood. These are saved as lists (their length depending on number of hops and number of neighbors), these are then truncated to be the length of the mean spiral lenght + 2 standard deviations of the spiral lengths. Afterwards the shorter spirals are padded with -1s, resulting in spiraling indices of equivalent lengths for all the vertices. These are used in the spiral_conv function from **models.py**.




