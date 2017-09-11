/*  Copyright 2011 AIT Austrian Institute of Technology
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/

/*
* IntegralImage.h
*
*  Created on: Nov 16, 2011
*      Author: Georg Nebehay
*/

#ifndef INTEGRALIMAGE_H_
#define INTEGRALIMAGE_H_
#include "switch.h"
#include <opencv/cv.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ocl/ocl.hpp"
#include "opencv2/nonfree/ocl.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;
using namespace cv::ocl;

namespace tld
{

	template <class T>
	class IntegralImage
	{
	public:
		T *data;  /* Array containg the entries for the integral image in row-first manner. Of size width*height. Allocated by tldAllocIntImg */
				  /* width, height: Dimensions of integral image.*/
		int width;
		int height;

		IntegralImage(cv::Size size)
		{
			data = new T[(size.width) * (size.height)];
		}

		virtual ~IntegralImage()
		{
			delete[] data;
		}

		void oclcalcIntImg(const cv::Mat &img, bool squared = false)
		{
			// const unsigned char *input = (const unsigned char *)(img.data);
			// T *output = data;
			int *input;
			int *output = data;
			DevicesInfo devices;
			getOpenCLDevices(devices);
			setDevice(devices[SELECTED_DEVICE_ID]);
			oclMat img1, img2, MatSqure;
			cv::Mat tempMat;
			img1 = img;


			cv::ocl::integral(img1, img2);
			tempMat = img2;

			input = (int*)tempMat.data;
			//double tic = cvGetTickCount();

			for (int i = 0; i <img.cols; i++)
			{
				for (int j = 0; j < img.rows; j++)
				{
					output[img.cols * j + i] = input[tempMat.cols * (j + 1) + i + 1];
				}
			}

			//double toc = (cvGetTickCount() - tic) / cvGetTickFrequency();
			//toc = toc / 1000000;
			//printf("            the copy data  used is %f************\n", toc);//float fps = 1 / toc;


		}

		void oclcalcIntImgSquare(const cv::Mat &img, bool squared = false)
		{
			// const unsigned char *input = (const unsigned char *)(img.data);
			// T *output = data;
			float  *input;
			float *output = data;
			oclMat img1, img2, MatSqure;
			cv::Mat tempMat;
			img1 = img;
			cv::ocl::integral(img1, img2, MatSqure);
			tempMat = MatSqure;

			input = (float*)tempMat.data;
			for (int i = 0; i <img.cols; i++)
			{
				for (int j = 0; j < img.rows; j++)
				{
					output[img.cols * j + i] = input[tempMat.cols * (j + 1) + i + 1];
				}
			}
		}

		void calcIntImg(const cv::Mat &img, bool squared = false)
		{
			const unsigned char *input = (const unsigned char *)(img.data);
			T *output = data;

			for (int i = 0; i < img.cols; i++)
			{
				for (int j = 0; j < img.rows; j++)
				{
					T A = (i > 0) ? output[img.cols * j + i - 1] : 0;
					T B = (j > 0) ? output[img.cols * (j - 1) + i] : 0;
					T C = (j > 0 && i > 0) ? output[img.cols * (j - 1) + i - 1] : 0;
					T value = input[img.step * j + i];

					if (squared)
					{
						value = value * value;
					}

					output[img.cols * j + i] = A + B - C + value;
					//printf("index, i=%d,j=%d,output value is %d\n", i,j,output[img.cols*j + i]);
				}
			}

		}


	};


} /* namespace tld */
#endif /* INTEGRALIMAGE_H_ */
