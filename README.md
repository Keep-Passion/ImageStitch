# VFSMS£:Very fast sequential micrograph stitching
In material researches, it is often highly desirable to observe images of whole microscopic sections with high resolution. So that micrograph stitching is an important technology to produce a panorama or larger image by combining multiple images with overlapping areas, while retaining microscopic resolution. However, due to high complexity and variety of microstructure, most traditional methods could not balance the speed and accuracy of stitching strategy. To overcome this problem, we develop a very fast sequential micrograph stitching method, called VFSMS, which employ incremental searching strategy and GPU acceleration to guarantee the accuracy and speed of the stitching results. Experimental results demonstrate that VFSMS achieve state-of-art performance on six types¡¯ microscopic datasets on both accuracy and speed aspects. Besides, it significantly outperform the most famous and commonly used software, such as ImageJ, Photoshop and Autostitch.

## Requirements
Python 3 need to be installed before running this scripts.
To run this algorithm, you need to install the python packages as follows:

    opencv-contrib(we have tested oepncv3.3.1)

## Examples
<p align = "center">
<img src="https://raw.githubusercontent.com/clovermini/MarkdownPhotos/master/005.png">
</p>


## Citation
This is an implementation of VFSMS algorithm in Python 3.
If you use it successfully for your research please be so kind to cite our work:
Ma B, Ban X, Hai H, Su Y, Wan B. VFSMS:Very fast sequential micrograph stitching, Still in Submission...
