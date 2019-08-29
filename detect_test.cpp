#include <opencv2/opencv.hpp>
#include "preprocess.hpp"

using namespace cv;
using namespace std;

// 문자열 출력 함수 - 그림자 효과 
void put_string(Mat &frame, string text, Point pt, int value)
{
	text += to_string(value);
	Point shade = pt + Point(2, 2);
	int font = FONT_HERSHEY_SIMPLEX;
	putText(frame, text, shade, font, 0.7, Scalar(0, 0, 0), 2);		// 그림자 효과 	
	putText(frame, text, pt, font, 0.7, Scalar(120, 200, 90), 2);// 작성 문자
}


Point2d calc_center(Rect obj)							// 사각형 중심 계산	
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
		cout << "카메라가 연결되지 않았습니다." << endl;
		exit(1);
	}



	CascadeClassifier face_cascade, eyes_cascade;
	load_cascade(face_cascade, "haarcascade_frontalface_alt2.xml");	// 정면 얼굴 검출기
	load_cascade(eyes_cascade, "haarcascade_eye.xml");			// 눈 검출기	

	vector<Rect> faces, eyes;
	vector<Point2d> eyes_center;

	

	for (;;) {
		Mat frame;
		capture.read(frame);

		put_string(frame, "EXPOS: ", Point(10, 40), capture.get(CAP_PROP_EXPOSURE));

		//////////////////////////////////

		Mat  gray = preprocessing(frame);								// 전처리
		face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0, Size(100, 100));	// 얼굴 검출
		if (faces.size() > 0)					// 얼굴 사각형 검출되면
		{	// 눈 검출 수행
			eyes_cascade.detectMultiScale(gray(faces[0]), eyes, 1.15, 7, 0, Size(25, 20));

			if (eyes.size() == 2) {							// 눈 사각형이 검출되면
				for (size_t i = 0; i < eyes.size(); i++)
				{
					Point2d center = calc_center(eyes[i] + faces[0].tl());	// 중심점 계산
					circle(frame, center, 5, Scalar(0, 255, 0), 2);	// 눈 중심에 원 그리기
				}
			}

			rectangle(frame, faces[0], Scalar(255, 0, 0), 2);	// 얼굴 검출 사각형 그리기
			imshow("image", frame);
		}
		
		////////////////////////////////

		
		if (waitKey(30) >= 0) break;
	}
	return 0;
}








