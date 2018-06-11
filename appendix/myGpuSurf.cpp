#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "conversion.h"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include <boost/python.hpp>
#include <iostream>

using namespace std;
using namespace cv::cuda;
using namespace cv;
namespace py = boost::python;

typedef unsigned char uchar_t;

vector<KeyPoint> h_keypointsA, h_keypointsB;
Mat h_descriptorsA, h_descriptorsB;
vector<DMatch> h_good_matches;

Mat tranVectorFloatToMat(vector<float> descriptor)
{
	// 将从GPU上下载得到的描述子换成CPU的mat矩阵
	int keypointNum = descriptor.size() / 128;
	Mat descriptorCPU(keypointNum, 128, CV_32F);
	for (int i = 0; i < keypointNum; i++)
	{
		float* data = descriptorCPU.ptr<float>(i);
		for (int j = 0; j < 128; j++)
		{
			data[j] = descriptor[i * 128 + j];
		}
	}
	return descriptorCPU;
}

Mat tranVectorKeyPointsToMat(vector<KeyPoint> Vkeypoints)
{
	int num = Vkeypoints.size();
	Mat keypointsMat = Mat(num, 2, CV_32F);
	for (int i = 0; i < num; i++)
	{
		float* data = keypointsMat.ptr<float>(i);
		data[0] = Vkeypoints[i].pt.x;
		data[1] = Vkeypoints[i].pt.y;
	}
	return keypointsMat;
}

Mat tranVectorDMatchToMat(vector<DMatch> VDMatch)
{
	int num = VDMatch.size();
	Mat dMatchMat = Mat(num, 2, CV_32S);
	for (int i = 0; i < num; i++)
	{
		int* data = dMatchMat.ptr<int>(i);
		data[0] = VDMatch[i].trainIdx;
		data[1] = VDMatch[i].queryIdx;
	}
	return dMatchMat;

}

PyObject* getImageAKeyPoints()
{
	NDArrayConverter cvt;
	Mat temp = tranVectorKeyPointsToMat(h_keypointsA);
	return cvt.toNDArray(temp);
}

PyObject* getImageBKeyPoints()
{
	NDArrayConverter cvt;
	Mat temp = tranVectorKeyPointsToMat(h_keypointsB);
	return cvt.toNDArray(temp);
}

PyObject* getImageBDescriptors()
{
	NDArrayConverter cvt;
	return cvt.toNDArray(h_descriptorsB);
}

PyObject* getImageADescriptors()
{
	NDArrayConverter cvt;
	return cvt.toNDArray(h_descriptorsA);
}

PyObject* getGoodMatches()
{
	NDArrayConverter cvt;
	Mat temp = tranVectorDMatchToMat(h_good_matches);
	return cvt.toNDArray(temp);
}

void matchFeaturesBySurf(PyObject *h_imageAPtr, PyObject *h_imageBPtr, float h_keypointsRatio, float searchRatio)
{
	// Release Vector before use
	h_good_matches.clear();
	h_keypointsA.clear();
	h_keypointsB.clear();

	NDArrayConverter cvt;

	GpuMat d_imageA, d_imageB;
	GpuMat d_keypointsA, d_keypointsB;
	GpuMat d_descriptorsA, d_descriptorsB;

	// 接收DLL传递的图像并转成MAT,并加载到GPU上
	d_imageA.upload(cvt.toMat(h_imageAPtr));
	CV_Assert(!d_imageA.empty());
	d_imageB.upload(cvt.toMat(h_imageBPtr));
	CV_Assert(!d_imageB.empty());


	SURF_CUDA surf;
	surf.keypointsRatio = h_keypointsRatio;
	// 压入向量
	surf(d_imageA, GpuMat(), d_keypointsA, d_descriptorsA);
	surf(d_imageB, GpuMat(), d_keypointsB, d_descriptorsB);

	//cout << "		FOUND " << d_keypointsA.cols << " keypoints on first image" << endl;
	//cout << "		FOUND " << d_keypointsB.cols << " keypoints on second image" << endl;

	//GPU: matching descriptors
	Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
	vector<vector<DMatch>> d_matches;
	matcher->knnMatch(d_descriptorsA, d_descriptorsB, d_matches, 2);

	// downloading results
	vector<float> h_descriptorsA_Array, h_descriptorsB_Array;
	surf.downloadKeypoints(d_keypointsA, h_keypointsA);				surf.downloadKeypoints(d_keypointsB, h_keypointsB);
	surf.downloadDescriptors(d_descriptorsA, h_descriptorsA_Array);	surf.downloadDescriptors(d_descriptorsB, h_descriptorsB_Array);
	h_descriptorsA = tranVectorFloatToMat(h_descriptorsA_Array);
	h_descriptorsB = tranVectorFloatToMat(h_descriptorsB_Array);


	for (int i = 0; i < d_matches.size(); i++)
	{
		if (d_matches[i][0].distance < searchRatio * d_matches[i][1].distance)
		{
			h_good_matches.push_back(d_matches[i][0]);
		}
	}
	//cout << "		The number of matches is " << h_good_matches.size() << endl;

	//// Show matching, only useful in testing
	//Mat img_matches;
	//drawMatches(Mat(d_imageA), h_keypointsA, Mat(d_imageB), h_keypointsB, h_good_matches, img_matches);

	//namedWindow("matches", 0);
	//imshow("matches", img_matches);
	//waitKey(0);

	// Release GPU Memory
	d_imageA.release();			d_imageB.release();
	d_keypointsA.release();		d_keypointsB.release();
	d_descriptorsA.release();	d_descriptorsB.release();
	surf.releaseMemory();
}

static void init()
{
	Py_Initialize();
	import_array();
}

BOOST_PYTHON_MODULE(myGpuSurf)
{
	init();
	py::def("matchFeaturesBySurf", matchFeaturesBySurf);
	py::def("getImageADescriptors", getImageADescriptors);
	py::def("getImageBDescriptors", getImageBDescriptors);
	py::def("getImageAKeyPoints", getImageAKeyPoints);
	py::def("getImageBKeyPoints", getImageBKeyPoints);
	py::def("getGoodMatches", getGoodMatches);
}