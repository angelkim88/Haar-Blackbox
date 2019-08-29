#include <opencv2/opencv.hpp>
#include "preprocess.hpp"

using namespace cv;
using namespace std;

// ���ڿ� ��� �Լ� - �׸��� ȿ�� 
void put_string(Mat &frame, string text, Point pt, int value)
{
	text += to_string(value);
	Point shade = pt + Point(2, 2);
	int font = FONT_HERSHEY_SIMPLEX;
	putText(frame, text, shade, font, 0.7, Scalar(0, 0, 0), 2);		// �׸��� ȿ�� 	
	putText(frame, text, pt, font, 0.7, Scalar(120, 200, 90), 2);// �ۼ� ����
}


Point2d calc_center(Rect obj)							// �簢�� �߽� ���	
{
	Point2d c = (Point2d)obj.size() / 2.0;
	Point2d center = (Point2d)obj.tl() + c;
	return center;
}

int main()
{
	VideoCapture  capture;

	capture.open("../car_image/normal.avi");
	if (!capture.isOpened())
	{
		cout << "ī�޶� ������� �ʾҽ��ϴ�." << endl;
		exit(1);
	}



	CascadeClassifier face_cascade, eyes_cascade;
	load_cascade(face_cascade, "haarcascade_frontalface_alt2.xml");	// ���� �� �����
	load_cascade(eyes_cascade, "haarcascade_eye.xml");			// �� �����	

	vector<Rect> faces, eyes;
	vector<Point2d> eyes_center;

	

	for (;;) {
		Mat frame;
		capture.read(frame);

		put_string(frame, "EXPOS: ", Point(10, 40), capture.get(CAP_PROP_EXPOSURE));

		//////////////////////////////////

		Mat  gray = preprocessing(frame);								// ��ó��
		face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0, Size(100, 100));	// �� ����
		if (faces.size() > 0)					// �� �簢�� ����Ǹ�
		{	// �� ���� ����
			eyes_cascade.detectMultiScale(gray(faces[0]), eyes, 1.15, 7, 0, Size(25, 20));

			if (eyes.size() == 2) {							// �� �簢���� ����Ǹ�
				for (size_t i = 0; i < eyes.size(); i++)
				{
					Point2d center = calc_center(eyes[i] + faces[0].tl());	// �߽��� ���
					circle(frame, center, 5, Scalar(0, 255, 0), 2);	// �� �߽ɿ� �� �׸���
				}
			}

			rectangle(frame, faces[0], Scalar(255, 0, 0), 2);	// �� ���� �簢�� �׸���
			imshow("image", frame);
		}
		
		////////////////////////////////

		
		if (waitKey(30) >= 0) break;
	}
	return 0;
}








