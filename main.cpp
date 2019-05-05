#include<opencv2/dnn.hpp>
#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;
using namespace cv::dnn;
string model_file = "C:/Users/Tang_Y/Downloads/bvlc_googlenet.caffemodel";
string model_txt_file = "C:/Users/Tang_Y/Downloads/bvlc_googlenet.prototxt";
string labels_txt_file = "C:/Users/Tang_Y/Downloads/synset_words.txt";
vector<string> readlabels();
int main()
{
	Mat image = imread("C:/Users/Tang_Y/Desktop/cat.jpg");
	if (image.empty())
	{
		printf("could not load pictures...");
		return -1;
	}
	namedWindow("input_image", CV_WINDOW_AUTOSIZE);
	imshow("input_image", image);
	vector<string> labels = readlabels();
	Net net = readNetFromCaffe(model_txt_file, model_file);
	if (net.empty())
	{
		printf("read caffe model failure..");
		return -1;
	}
	Mat inputBlob = blobFromImage(image, 1.0, Size(224, 224), Scalar(104, 117, 123));
	Mat prob;
	for (int i = 0; i < 9; i++)
	{
		net.setInput(inputBlob, "data");
		prob = net.forward("prob");
	}
	Mat probMat = prob.reshape(1, 1);
	Point classNumber;
	double classprob;
	minMaxLoc(probMat, NULL, &classprob, NULL, &classNumber);
	int index = classNumber.x;
	printf("\n current image classification:%s,possible:%.2f", labels.at(index).c_str(), classprob);
	putText(image, labels.at(index), Point(20, 20), FONT_HERSHEY_COMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
	imshow("²âÊÔ½á¹û", image);
	waitKey(0);
	return 0;

}
vector<string> readlabels()
{
	vector<string> classNames;
	ifstream fp(labels_txt_file);
	if (!fp.is_open())
	{
		printf("couldn't open the file");
		exit(-1);
	}
	string name;
	while (!fp.eof())
	{
		getline(fp, name);
		if (name.length())
		{
			classNames.push_back(name.substr(name.find(' ') + 1));
		}
	}
	fp.close();
	return classNames;

}