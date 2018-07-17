# Micrograph oriented sequential micrograph stitching: VFSMS
In material researches, it is often highly desirable to observe images of whole microscopic sections with high resolution. So that micrograph stitching is an important technology to produce a panorama or larger image by combining multiple images with overlapping areas, while retaining microscopic resolution. However, due to high complexity and variety of microstructure, most traditional methods could not balance the speed and accuracy of stitching strategy. To overcome this problem, we develop a very fast sequential micrograph stitching method, called VFSMS, which employ incremental searching strategy and GPU acceleration to guarantee the accuracy and speed of the stitching results. Experimental results demonstrate that VFSMS achieve state-of-art performance on six types' microscopic datasets on both accuracy and speed aspects. Besides, it significantly outperform the most famous and commonly used software, such as ImageJ, Photoshop and Autostitch.

## Requirements
Python 3 need to be installed before running this scripts.
To run this algorithm, you need to install the python packages as follows:

    opencv-contrib(we have tested oepncv3.3.1)

## Examples
There are some examples of VFSMS are shown behind.

Six types’ local and global micrographs and their shooting path. The red translucent region represent one shot from microscope. The red dotted line refer to shooting path. (a) Iron crystal in scanning electron microscopy (SEM) with its detail imaging. (b) Pairwise shooting path of (a) with 2 local images. (c) Dendritic crystal in SEM with its detail imaging. (d) Grid shooting path of (c) with 90 local images. (e) Zircon in SEM with its detail imaging. (f) Zircon in transmission electron microscope (TEM) with its detail imaging. (g) Zircon in black scattered electron imaging (BSE) with its detail imaging. (h) Zircon in cathodoluminescence spectroscopy (CL) with its detail imaging. (i) Shooting path for (e)(f)(g)(h), the numbers of local images depends on the length of sample.
<p align = "center">
<img src="https://github.com/MATony/ImageStitch/blob/master/demoImages/examplesOfImageStitch.png">
</p>


## Citation
If you use it successfully for your research please be so kind to cite our work:

Ma B, Ban X, Huang H, Ya S, Liu W, Wu D. Micrograph oriented sequential micrograph stitching: VFSMS, Still in Submission...

## Acknowledgements
The authors acknowledge the financial support from the National Key Research and Development Program of China (No. 2016YFB0700500)
