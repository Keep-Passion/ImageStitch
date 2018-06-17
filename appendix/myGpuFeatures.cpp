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


Mat tranVectorKeyPointsDescriptorsToMat(vector<KeyPoint> Vkeypoints, Mat Mdescriptors)
{
	int keyPointsNum = Vkeypoints.size();
	// int descriptorsDim = Vdescriptors.size() / keyPointsNum;
	int descriptorsDim = Mdescriptors.cols;

	int sz[3] = { keyPointsNum, descriptorsDim, 2 };
	Mat keyPointsDescriptors(3, sz, CV_32F, Scalar(0.00));
	
	for (int i = 0; i < keyPointsNum; i++)
	{
		float* data = keyPointsDescriptors.ptr<float>(i);
		data[0] = Vkeypoints[i].pt.x;
		data[2] = Vkeypoints[i].pt.y;
	}

	//for (int i = 0; i < keyPointsNum; i++)
	//{
	//	float* data = keyPointsDescriptors.ptr<float>(i);
	//	for (int j = 0; j < descriptorsDim; j++)
	//	{
	//		data[j*2 + 1] = Vdescriptors[j];
	//	}
	//}

	for (int i = 0; i < keyPointsNum; i++)
	{
		float* dataAccept = keyPointsDescriptors.ptr<float>(i);
		float* dataSend = Mdescriptors.ptr<float>(i);
		for (int j = 0; j < descriptorsDim; j++)
		{
			dataAccept[j*2 + 1] = dataSend[j];
		}
	}
	return keyPointsDescriptors;
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

PyObject* detectAndDescribeBySurf(PyObject *h_imageAPtr,  float hessianThreshold, int nOctaves, int nOctaveLayers, bool isExtended, float keypointsRatio, bool isUpright)
{
	// 建立GPU上存储空间
	GpuMat d_image, d_keypoints, d_descriptors;

	// 接收DLL传递的图像并转成MAT,并加载到GPU上
	NDArrayConverter cvt;
	d_image.upload(cvt.toMat(h_imageAPtr));
	CV_Assert(!d_image.empty());

	SURF_CUDA surf = SURF_CUDA(hessianThreshold, nOctaves, nOctaveLayers, isExtended, keypointsRatio, isUpright);
	surf(d_image, GpuMat(), d_keypoints, d_descriptors);
	// downloading results from GPU to CPU
	vector<KeyPoint> h_keypoints;
	Mat h_descriptors;

	surf.downloadKeypoints(d_keypoints, h_keypoints);
	d_descriptors.download(h_descriptors);

	// print only for testing
	//cout << "		FOUND " << d_keypoints.cols << " keypoints on image" << endl;
	//for (int i = 0; i < 64; i++)
	//{
	//	cout << h_descriptors.at<float>(0, i) << ",";
	//}
	//cout << endl;
	//cout << h_keypoints[0].pt.x << "," << h_keypoints[0].pt.y << endl;;
	//cout << endl;

	// releasing
	d_image.release();
	d_keypoints.release();
	d_descriptors.release();
	surf.releaseMemory();

	// 通过DLL发送ndarry回去
	return cvt.toNDArray(tranVectorKeyPointsDescriptorsToMat(h_keypoints, h_descriptors));
}

PyObject* detectAndDescribeByOrb(PyObject *h_imageAPtr, int nFeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel, int WTA_K, int scoreType, int patchSize, int fastThreshold, bool blurForDescriptor)
{
	// 建立GPU上存储空间
	GpuMat d_image, d_keypoints, d_descriptors, d_descriptors_32F;

	// 接收DLL传递的图像并转成MAT,并加载到GPU上
	NDArrayConverter cvt;
	d_image.upload(cvt.toMat(h_imageAPtr));
	CV_Assert(!d_image.empty());
	Ptr<cuda::ORB> d_orb = cuda::ORB::create(nFeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, 0, patchSize, fastThreshold, blurForDescriptor);

	d_orb->detectAndComputeAsync(d_image, GpuMat(), d_keypoints, d_descriptors);
	d_descriptors.convertTo(d_descriptors_32F, CV_32F);
	
	vector<KeyPoint> h_keypoints;
	Mat h_descriptors;

	d_orb->convert(d_keypoints, h_keypoints);
	d_descriptors_32F.download(h_descriptors);

	// print only for testing
	//cout << "		FOUND " << d_keypoints.cols << " keypoints on image" << endl;
	//cout << "		Des shape" << h_descriptors.rows << " " << h_descriptors.cols << endl;
	//for (int i = 0; i < h_descriptors.cols; i++)
	//{
	//	cout << h_descriptors.at<float>(0, i)<< ",";
	//}
	//cout << endl;	
	//cout << h_keypoints[0].pt.x << "," << h_keypoints[0].pt.y << endl;;
	//cout << endl;


	// releasing
	d_image.release();
	d_keypoints.release();
	d_descriptors.release();
	d_descriptors_32F.release();

	// 通过DLL发送ndarry回去
	return cvt.toNDArray(tranVectorKeyPointsDescriptorsToMat(h_keypoints, h_descriptors));
}

PyObject* matchDescriptors(PyObject *h_descriptorsAPtr, PyObject *h_descriptorsBPtr, int featureType, float param)
{
	GpuMat d_descriptorsA, d_descriptorsB;

	NDArrayConverter cvt;
	
	d_descriptorsA.upload(cvt.toMat(h_descriptorsAPtr));
	d_descriptorsB.upload(cvt.toMat(h_descriptorsBPtr));

	Ptr<cv::cuda::DescriptorMatcher> matcher;
	vector<DMatch> h_good_matches;

	if(featureType == 1 || featureType == 2)
	{
		// sift 和 surf 模式下用欧氏距离，并判断最近邻和次近邻距离
		matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
		vector<vector<DMatch>> d_matches;
		matcher->knnMatch(d_descriptorsA, d_descriptorsB, d_matches, 2);
		for (int i = 0; i < d_matches.size(); i++)
		{
			if (d_matches[i][0].distance < param * d_matches[i][1].distance)
			{
				h_good_matches.push_back(d_matches[i][0]);
			}
		}
	}
	else if (featureType == 3)
	{
		// orb 运用 汉明距离，并判断距离是否小于某个参数
		vector<DMatch> d_matches;
		matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
		matcher->match(d_descriptorsA, d_descriptorsB, d_matches, GpuMat());
		for (int i = 0; i < d_matches.size(); i++)
		{
			if (d_matches[i].distance < param)
			{
				h_good_matches.push_back(d_matches[i]);
			}
		}
	}

	// releasing
	d_descriptorsA.release(); d_descriptorsB.release();
	matcher.release();

	// 通过DLL发送ndarry回去
	return cvt.toNDArray(tranVectorDMatchToMat(h_good_matches));
}

static void init()
{
	Py_Initialize();
	import_array();
}

BOOST_PYTHON_MODULE(myGpuFeatures)
{
	init();
	py::def("detectAndDescribeBySurf", detectAndDescribeBySurf);
	py::def("detectAndDescribeByOrb", detectAndDescribeByOrb);
	py::def("matchDescriptors", matchDescriptors);
}