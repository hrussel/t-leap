# T-LEAP: Occlusion-robust pose estimation of walking cows using temporal information

This is the official repo of the paper *T-LEAP: Occlusion-robust pose estimation of walking cows using temporal 
information (Russello et al., 2022)*. If using, please cite our [paper](https://doi.org/10.1016/j.compag.2021.106559):

    @article{russello2022t,
    title={T-LEAP: Occlusion-robust pose estimation of walking cows using temporal information},
    author={Russello, Helena and van der Tol, Rik and Kootstra, Gert},
    journal={Computers and Electronics in Agriculture},
    volume={192},
    pages={106559},
    year={2022},
    publisher={Elsevier}
    }

For questions, contact me at: `helena [dot] russello [at] wur [dot] nl`.

## License

Copyright 2022 Helena Russello

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Requirements

To install the python environment, I recommend to use the `environment.yml` file with the command:
    
    conda env create -f environment.yml 


## Usage

If you already have other pose estimation models working in Pytorch, I'd advise to use the `tleap.py` 
file in the `models` subfolder and run it with your own code.

### Configuration

In the sub-folder `cfg`, you can find an example of a default config file.
  * `data_folder` is the path to your images.
  * `dataset_csv` is the path to the csv file of your training dataset. The expected format of this file is described in the next subsection.
  * `dataset_test` is the path to the csv file of your test dataset.
  * `device` pytorch argument for the device.
  * `frequent` logging frequency.
  * `save_checkpoint` folder where to save the checkpoints.
  * `load_checkpoint` whether to load the model from a checkpoint. (i.e., No, or the path to the checkpoint).
  * `wandb` Whether to log training with [*Weights and Biases*](https://wandb.ai/).
  * `group` If you use logging with *Weights and Biases*, you can give a group name to keep your runs organised.
  * `file_format` file format for the logged images.
  * `keypoints` list of keypoints
  * `lr` learning rate
  * `epochs` number of training epochs.
  * `batch_size` 
  * `lr_decay` whether to use a LR scheduler
  * `seq_length` sequence lenght. Use 1 for static LEAP, 2 to 4 for T-LEAP
  * `optimizer` optimizer to use. Choose between `sgd`, `adam` or `amsgrad`
  * `depth` depth of the NN. Use 3 for the initial depth of LEAP, 4 for T-LEAP and deeper LEAP 
### Data format
*Note: the data doesn't follow the most straightforward structure, but (if there is demand for it) I plan to add in the future some boilerplates later on to make this easier.*


1) Extract all the frames of your videos as images. You need one folder with the name of the video, 
in which each frame is named as `frame_number.png`. For instance, if the name of the video is `video1.mp4`, 
there would be a folder named `video1` and frames `0.png` to say `100,png`. Note that the frame number doesn't need 
to start at 0 (for instance if you want to drop empty frames at the start or end of the video)
2) Have a subfolder named `labels_csv` that contains the keypoint annotation per video. Each video should have its own
csv file and named `video_name.csv`. Here is how the csv file should look like. Note that the column of the likelihood 
is not used, but still needed for parsing. Put any value you like.

        video,  frame, Keypoint1_x, Keypoint1_y, Keypoint1_likelihood, Keypoint2_x, Keypoint2_y, Keypoint2_likelihood, ....
        --------------------------------------------------------------------------------------------------------------------
        video1, 0,     92,          136,         1,                    90,          130,         1, ...

3) In summary, your data folder should have a structure like in the example bellow. 
  
        ./data
            |__ video1/ 
                     |__ 0.png
                         1.png
                         ...
            |__ video2/ 
                     |__ 35.png
                         36.png
                         ...                 
            |__ labels_csv/ 
                     |__ video1.csv
                         video2.csv
                         ...   
    
4) In addition to that, you should have a separate csv file for training and testing. The location or name of this file doesn't matter, simply write the path to them in the config file.
The csv file contains the list of video frames sequences that belong to the train or test set. The format is as follows:

        filename,      start_frame,    end_frame
        ------------------------------------------------
        video1.mp4,    102,            105
        video26.mp4,   10,             13
        video1.mp4,    1,              4   
        ....

5) The images can be cropped in advance (faster training), or when they are loaded. Note that the resize operation 
takes a long time and might drastically increase the training time. 
For cropping the images in training and testing, add the following transforms:

       transform = [
               SequentialPoseDataset.CropBody(margin=100, random=False, square=True, keep_orig=True),
               SequentialPoseDataset.Rescale(200)
           ]

### Training and testing the model

Most of the parameters and different paths should be specified in your `.yml` configuration file (as specified in the configuration section above).
Run the script:

    train_sep.py --config path/to/your/configfile


Some parameters can be overriden from the `.yml` config file, and provided as arguments to the 
python script: `frequent`, `lr`, `epochs`, `batch_size`, `seq_length`, `seed`, `optimizer`, `group`, `depth`.

Once the training is complete, the trained model is evaluated on the test set.

## Benchmarks

Because the dataset from the T-LEAP paper is not open-source, you cannot train T-LEAP on that data.
Instead, we *plan to (expected mid-2022)* provide benchmark results on the 
[Horse-10 dataset](http://www.mackenziemathislab.org/horse10) from Mathis et al. 2020. 

### Horse-10

*Coming soon...*