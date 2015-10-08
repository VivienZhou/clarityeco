#ifndef __SERIALIZE_H_
#define __SERIALIZE_H_

#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

string serialize(Mat input, string filename);

Mat deserialize(stringstream& input, string &filename);

#endif
