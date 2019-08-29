#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void load_cascade(CascadeClassifier& cascade, string fname)
{
	String  path = "C:/opencv/sources/data/haarcascades/";	// ����� ����
	String  full_name = path + fname;

	CV_Assert(cascade.load(full_name));					// �з��� �ε� �� ����ó��
}

Mat preprocessing(Mat image)
{
	Mat gray;
	cvtColor(image, gray, CV_BGR2GRAY);	// ��ϵ� ����ȯ
	equalizeHist(gray, gray);					// ������׷� ��Ȱȭ
	return gray;
}