#include <iostream>
#include <CGAL/Point_set_3.h>
#include <CGAL/IO/read_ply_points.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <Tools.h>
#include <Distance_Transform.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/create_offset_polygons_2.h>
/*
���д�����ʹ��CGAL����������߼򻯵Ĺؼ�����,��������:

CGAL::Polyline_simplification_2<K> simplifier:
�ⶨ����һ��Polyline_simplification_2����simplifier,���ڽ��������򻯡�
K�Ƕ������ֵ��������,ͨ��ΪCGAL::Exact_predicates_inexact_constructions_kernel��
simplifier.simplify:
����simplify�������������򻯡�
������ԭʼ������ĵ�����begin��end��
����Ǽ򻯺������������simplified��
std::back_inserter:
���ڹ���һ��back_insert_iterator,�Ա㽫simplify�����push_back��simplified�����С�
ԭ��:
Polyline_simplification_2�ڲ�ʹ��Douglas-Peucker�㷨�ƽ�ԭʼ������
���ϵݹ�ָ�,������Զ��,ɾ���м�����㡣
ͬʱʹ���ںϼ����ϲ�ϸС�Ρ�
ɾ���ظ��㡣
�ۺ�����,������Ч������,ɾ������㡣
�������д�������CGAL��װ�õ��㷨,�ǳ����������������㼯�ļ򻯴���,����Ҫ�Լ�ʵ��Douglas-Peucker���㷨��
*/
//#include <CGAL/Polyline_simplification_2.h>
#include <limits>
#include <math.h>
#include <iomanip>
using namespace std;
typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
//��������
typedef CGAL::Polygon_2<Kernel> Polygon2f;
typedef float FT;
float INF = 10e7;
#include <iostream>
#include <vector>
#include <cmath>
double resolution;
double top_left_x;
double top_left_y;
int width = 500;
double getDistance(Point_2f& p1, Point_2f& p2) {
	double dx = p2.x - p1.x;
	double dy = p2.y - p1.y;
	return std::sqrt(dx * dx + dy * dy);
}

double getPerpendicularDistance(Point_2f& p, Point_2f& start, Point_2f& end) {
	double area = std::abs(0.5 * (start.x * end.y + end.x * p.y + p.x * start.y - end.x * start.y - p.x * end.y - start.x * p.y));//���/2
	double length = getDistance(start, end);
	return (2.0 * area) / length;
}

void douglasPeuckerSimplify(std::vector<Point_2f> polyline, std::vector<Point_2f>& simplified, double epsilon, double epsilonL, vector<pair<double, double>>& lengthAngles) {
	int end = polyline.size() - 1;
	//�����ټ򻯣�ֱ�Ӹ�ֵ
	if (end < 2) {
		simplified = polyline;
		return;
	}
	int index = 0;
	double maxDistance = 0.0;

	for (int i = 1; i < end; i++) {
		double distance = getPerpendicularDistance(polyline[i], polyline[0], polyline[end]);
		if (distance > maxDistance) {
			index = i;
			maxDistance = distance;
		}
	}
	//if (maxDistance > epsilon || (lengthAngles[index].first > epsilonL) || (lengthAngles[index].second - 90.0) < 15.0)
	if (maxDistance > epsilon || (lengthAngles[index].second - 90.0) < 15.0)
		//	if (maxDistance > epsilon || (lengthAngles[index].first > epsilonL))
		//if (maxDistance > epsilon) 
	{
		std::vector<Point_2f> simplified1, simplified2;
		std::vector<Point_2f> subPolyline1(polyline.begin(), polyline.begin() + index + 1);
		std::vector<Point_2f> subPolyline2(polyline.begin() + index, polyline.end());
		std::vector<pair<double, double>> sublengthAngles1(lengthAngles.begin(), lengthAngles.begin() + index + 1);
		std::vector<pair<double, double>> sublengthAngles2(lengthAngles.begin() + index, lengthAngles.end());
		douglasPeuckerSimplify(subPolyline1, simplified1, epsilon, epsilonL, sublengthAngles1);
		douglasPeuckerSimplify(subPolyline2, simplified2, epsilon, epsilonL, sublengthAngles2);
		simplified.insert(simplified.end(), simplified1.begin(), simplified1.end() - 1);
		simplified.insert(simplified.end(), simplified2.begin(), simplified2.end());
	}
	else {
		simplified.push_back(polyline[0]);
		simplified.push_back(polyline[end]);
	}
}

vector<Point_2f> Tools::polygon_ISO(const std::vector<Point_2f>& polygon, double ISOValue)
{
	//��read_polygon���Եõ�����

	//��ȡ����ε����꣬���߽��л���ͼ��  ��ͼ��ת��Ϊ����Ļ�����Ϊ�����ˣ����Զ���εĶ�����Ŀ�����࣬���±߽���Բ��
	cv::Mat  image1 = Tools::read_polygon(polygon, resolution, abs(ISOValue), top_left_x, top_left_y, width);
	//��ȡ����λ��
	int width_x = image1.cols;
	int width_y = image1.rows;
	vector<std::vector<float>> grid_df_1;
	//��ͼ���л�ȡ��Ҫ�õ��ľ���
	Image_File_IO::convert_image_2_matrix(image1, grid_df_1, width_x, width_y);
	cv::Mat canny_output;
	//ͼ���ֵ�� ���ߵĵط�Ϊ255
	cv::threshold(image1, canny_output, 100, 255, cv::THRESH_BINARY);

	vector<vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;
	//�汾���⣬д���ı�  CV_RETR_LIST��CV_CHAIN_APPROX_SIMPLE
	//���ͼ���������п��ܵ����� 
	/*
ͼ���д��ڶ����ͨ������findContours() �����Ὣͼ���е�ÿ����ͨ������Ϊһ��������������
��ʹ���ͼ����ֻ��һ������Σ�����ö���ε��ڲ����ڿն���������ͨ������Щ����Ҳ���ܱ���Ϊ������������
���ڶ���κ�������
*/
	cv::findContours(canny_output, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	//�԰�Χ�����ľ��ν��������Ա㰴�վ��ε�����Ӵ�С��˳���������������
	cout << "����������" << contours.size() << endl;
	for (int i = 0; i < contours.size(); i++)
	{
		//����ÿ�������İ�Χ����
		cv::Rect reci = cv::boundingRect(contours[i]);
		for (size_t j = i + 1; j < contours.size(); j++)
		{//�����Χ����
			cv::Rect recj = cv::boundingRect(contours[j]);
			//�Ա�������Ӧ�ľ�������
			//���н�������
			if (reci.area() < recj.area())
			{
				vector<cv::Point> tmp = contours[i];
				contours[i] = contours[j];
				contours[j] = tmp;
			}
		}
	}
	// ����һ�����ڴ洢�ƽ������ε�����
	std::vector<std::vector<Point_2f>> approxContours(contours.size());
	/*
	curve����Ҫ�ƽ������ߣ�ͨ������һϵ�е���ɵ�������

	epsilon���ƽ����Ȳ���������ƽ��߶���ԭ����֮��������롣
	��С��epsilonֵ��ʹ�ƽ��Ķ���θ��ӽ�ԭʼ���ߡ�

	closed��һ������ֵ��ָ�������Ƿ��Ǳպϵġ�
	*/
	// ��ÿ���������ж���αƽ�  �ǽ����߽��Ʊ�ʾΪһϵ�е㣬�����ٵ��������һ���㷨
	double arlength1 = cv::arcLength(contours[0], true);
	double arlength2 = cv::arcLength(contours[1], true);
	cout << "�ܳ�Ϊ��" << arlength1 << endl;
	cout << "�ܳ�Ϊ��" << arlength2 << endl;
	double arlength3 = cv::arcLength(polygon, true);
	cout << "�ܳ�Ϊ��" << arlength3 << endl;
	/*******************************************************************************************/
	//�����ÿ������������������ĳ��ȺͽǶ�ֵ�������жϵ�ǰ�����Ƿ�Ӧ�ü������� ����douglasPeuckerSimplify�㷨
	vector<pair<double, double>> lengthAngles;
	for (int i = 0; i < polygon.size(); i++) {
		if (i == 0) {
			Point_2f v1 = polygon[i] - polygon[(i + 1) % polygon.size()];
			Point_2f v2 = polygon[i] - polygon[polygon.size() - 1];
			double maxlength = max(cv::norm(v1), cv::norm(v2));
			double angle = acosf(v1.dot(v2) / (cv::norm(v1) * cv::norm(v2))) * 180.0 / CV_PI;
			pair<double, double> p(maxlength, angle);
			lengthAngles.push_back(p);

		}
		else {
			Point_2f v1 = polygon[i] - polygon[(i + 1) % polygon.size()];
			Point_2f v2 = polygon[i] - polygon[i - 1];
			double maxlength = max(cv::norm(v1), cv::norm(v2));
			double angle = acosf(v1.dot(v2) / (cv::norm(v1) * cv::norm(v2))) * 180.0 / CV_PI;
			pair<double, double> p(maxlength, angle);
			lengthAngles.push_back(p);
		}
	}
	//��ȡ����֮��������̫�� �����Ż������������￼��ֱ��ʹ��ԭ���Ķ����������
	for (size_t i = 0; i < contours.size(); i++)
	{
		// ָ���ƽ����Ȳ��� epsilon����С����ֵ��ʹ�ñƽ�����θ��ӽ�ԭʼ����,��Ϊ�Ǽ����������Ƿ�С����ֵ
		// cv::arcLength(contours[i], true)��������������ܳ��򻡳�
		//double epsilon = 0.0018 * cv::arcLength(contours[i], true);
		double epsilon = 0.0056 * cv::arcLength(contours[i], true);
		double epsilonL = 0.6 * cv::arcLength(contours[i], true);
		cout << "contours�Ĵ�С" << contours[0].size() << endl;
		// ���ж���αƽ�
		//cv::approxPolyDP(polygon, approxContours[i], epsilon, true);
		douglasPeuckerSimplify(polygon, approxContours[i], epsilon, epsilonL, lengthAngles);

		cout << "epsilon��ֵ: " << epsilon << endl;
		cout << "epsilonL��ֵ: " << epsilonL << endl;
	}
	//�������İ�Χ������Եõ���������
	int max_index = 0;
	if (ISOValue < 0)
	{
		max_index = 1;
	}
	//�õ���չ���ҽ��ж��������˵�����
	vector<Point_2f> polyp;
	cout << endl;
	//ʹ�ö������� �õ��ļ򻯶���εĽ��ƽ�� �������������Ķ���εļ�
	for (size_t j = 0; j < approxContours[0].size(); j++)
	{
		//	FT x = approxContours[0][j].x * resolution + top_left_x;
		//	FT y = approxContours[0][j].y * resolution + top_left_y;
		FT x = approxContours[0][j].x;
		FT y = approxContours[0][j].y;
		Point_2f pp(x, y);
		//	cout << fixed << std::setprecision(11) << x << " " << y << " " << endl;
		polyp.push_back(pp);
	}
	cout << endl;

	return polyp;
};

cv::Mat Tools::read_polygon(const std::vector<Point_2f>& polygon, double& resolution,
	double ISOValue, double& top_left_x, double& top_left_y, int& width)
{
	//���ɶ���εı߿�  �������ɽ����չʾ����
	cv::Rect rec = cv::boundingRect(polygon);
	//Bbox2f rec = Global::bbox_of(polygon);
	//����
	resolution = 0.8;//����resolution
	width = 1.0 * (rec.width) / resolution;//����width
	//������ʮ 20+20 = 40
	float dis_add = ISOValue + 20;
	//����Ҫ���ŵ����Ͻǵ�����λ��
	top_left_x = rec.x - ISOValue - dis_add * resolution;
	top_left_y = rec.y - ISOValue - dis_add * resolution;
	//����Ҫ���ŵ����½ǵ�����λ��
	double buttom_right_x = rec.x + (rec.width) + ISOValue + dis_add * resolution;
	double buttom_right_y = rec.y + (rec.height) + ISOValue + dis_add * resolution;

	//���ڴ�С
	int width_x = (buttom_right_x - top_left_x) / resolution;
	int width_y = (buttom_right_y - top_left_y) / resolution;

	cv::Mat drawing = cv::Mat::zeros(cv::Size(width_x, width_y), CV_8UC1);
	//����ԭʼ�����
	for (int j = 0; j < polygon.size(); j++)
	{

		//���ƶ���Σ������֮�������
		Point_2f p;
		//�õ���㣬��Ϊԭʼ����̫���ˣ��ڴ�����ʾ���������൱�ڽ���һ������
		int begin_x = (polygon[j].x - top_left_x) / resolution;
		int begin_y = (polygon[j].y - top_left_y) / resolution;
		int end_x = (polygon[(j + 1) % polygon.size()].x - top_left_x) / resolution;
		int end_y = (polygon[(j + 1) % polygon.size()].y - top_left_y) / resolution;
		cv::Scalar color = cv::Scalar(255);
		line(drawing, Point_2f(begin_x, begin_y), Point_2f(end_x, end_y), color, 1);
	}
	//���ض���ε�ͼ�� cv::Mat drawing
	return drawing;
};

cv::Mat Tools::reads_polygonss(const std::vector<vector<Point_2f>>& polygons)
{

	cv::Mat drawing = cv::Mat::zeros(cv::Size(2000, 2000), CV_8UC1);
	//����ԭʼ�����
	std::vector<std::vector<cv::Point>> intPolygons;
	for (const auto& polygon : polygons) {
		std::vector<cv::Point> intPolygon;
		// ����������е�ÿ����  
		for (const auto& point : polygon) {
			cv::Point intPoint(cv::int32_t(point.x), cv::int32_t(point.y));
			// ��ת����ĵ���ӵ� intPolygon ��  
			intPolygon.push_back(intPoint);
		}
		// ��ת����Ķ������ӵ� intPolygons ��  
		intPolygons.push_back(intPolygon);
	}
	// ��ͼ���ϻ��ƶ���
	for (const auto& p : polygons)
	{
		/*
		int a = 1;
		for (const auto& v : p) {
			cv::circle(drawing, v, 1, cv::Scalar(255), -1);  // ���ƺ�ɫԲ��

				Point_2f textPosition(v.x, v.y);  // �ı���λ��
				std::string text = std::to_string(a);  // ������ת��Ϊ�ַ���
				a++;
				cv::Scalar textColor(255);  // �ı�����ɫ����ɫ��
				int fontFace = cv::FONT_HERSHEY_SIMPLEX;  // ��������
				double fontScale = 0.3;  // �������ű���
				int thickness = 1;  // �ı��߿�
	cv::putText(drawing, text, textPosition, fontFace, fontScale, textColor, thickness);

		}
		*/
		int a = 1;
		for (const auto& v : p) {
			cv::circle(drawing, v, 1, cv::Scalar(255), -1);  // ���ƺ�ɫԲ��											 

			Point_2f textPosition(v.x, v.y);  // �ı���λ��
			std::string text = std::to_string(a);  // ������ת��Ϊ�ַ���
			a++;
			cv::Scalar textColor(255);  // �ı�����ɫ����ɫ��
			int fontFace = cv::FONT_HERSHEY_SIMPLEX;  // ��������
			double fontScale = 0.3;  // �������ű���
			int thickness = 1;  // �ı��߿�
			cv::putText(drawing, text, textPosition, fontFace, fontScale, textColor, thickness);

		}
		for (int i = 0; i < intPolygons.size(); i++) {

			cv::polylines(drawing, intPolygons[i], true, cv::Scalar(255, 255, 255), 1);
		}
	}
	/*

	for (int i = 0; i < polygons.size(); i++) {
		vector<Point_2f>polygon = polygons[i];
		for (int j = 0; j < polygon.size(); j++)
		{

			//���ƶ���Σ������֮�������
			int begin_x = polygon[j].x;
			int begin_y = polygon[j].y;
			int end_x = polygon[(j + 1) % polygon.size()].x;
			int end_y = polygon[(j + 1) % polygon.size()].y;
			cv::Scalar color = cv::Scalar(255);
			line(drawing, Point_2f(begin_x, begin_y), Point_2f(end_x, end_y), color, 1);
		}
	}
	*/


	//���ض���ε�ͼ�� cv::Mat drawing
	return drawing;

};

void Tools::convert_matrix(const std::vector<std::vector<float>>& initial_grid, cv::Mat& mat, const int& width_x, const int& width_y, float Value_thres)
{
	mat = cv::Mat::zeros(cv::Size(width_x, width_y), CV_8UC1);

	for (unsigned int x = 0; x < width_x; x++)
	{
		for (unsigned int y = 0; y < width_y; y++)
		{
			//�ڲ�С��Value_thres����������Ϊ��ɫ���ⲿΪ��ɫ
			//�൱������չ����֮�ڵ��ǰ�ɫ��֮���Ϊ��ɫ
			if (initial_grid[x][y] < Value_thres)
			{
				mat.ptr(y)[x] = 255;
			}
			else {
				mat.ptr(y)[x] = 0;
			}

		}
	}
};

void Image_File_IO::convert_image_2_matrix(const cv::Mat& image, std::vector<std::vector<float>>& initial_grid, int& width_x, int& width_y)
{
	width_x = image.cols;
	width_y = image.rows;
	initial_grid.resize(width_x);
	for (unsigned int i = 0; i < initial_grid.size(); i++)
	{
		initial_grid[i].resize(width_y);
	}

	for (unsigned int x = 0; x < width_x; x++)
	{
		for (unsigned int y = 0; y < width_y; y++)
		{
			uchar  row_gray = image.ptr(y)[x];
			//��¼��������Ӵ�������,�Լ�����ֵ����ΪINF
			//�൱�ڰ��ߴ�Ϊ�㣬Ȼ������ֵΪ��ĵط�����Ϊ����
			if (row_gray > 100)
			{
				//����������Ϊ���ʾ�Ǳ߽�
				initial_grid[x][y] = 0;
			}
			else
			{
				//�����صĵط�����Ϊ����
				initial_grid[x][y] = INF;
			}
		}
	}
};

int main() {

	vector<vector<Point_2f>> poly;
	vector<vector<Point_2f>> poly1;
	vector<vector<Point_2f>> p;
	std::string filename = "polygons.txt";
	std::ifstream input(filename);
	float a, b;
	if (input.is_open()) {
		std::string line;
		while (std::getline(input, line)) {
			std::vector<Point_2f> vec;
			std::istringstream iss(line);

			while (iss >> a >> b) {
				vec.push_back(Point_2f(a, b));
			}
			poly.push_back(vec);
			poly1.push_back(vec);
		}
		input.close();
	}
	else {
		std::cout << "�޷����ļ�" << std::endl;
	}
	vector<int> l;
	for (size_t i = 0; i < poly.size(); i++)
	{
		l.push_back(poly[i].size());
	}
	cout << "��������" << l.size() << endl;
	int sum = 0;
	for (size_t i = 0; i < l.size(); i++)
	{
		cout << "ԭʼ�Ķ���εĵ����ͱ����� " << l[i] << endl;
		sum += l[i];
	}
	cout << "ԭʼ�Ķ���εĵ����ͱ����� " << sum << endl;
	
	for (size_t i = 0; i < poly.size(); i++)
	{
		vector<Point_2f> c = Tools::polygon_ISO(poly[i], 20.0);


		p.push_back(c);
	}
	std::vector<vector<Point_2f>> polygonss;
	//�޳���һ��ֱ������Ķ����
	for (auto sub_p : p) {
		std::vector<Point_2f>pp;
		for (int i = 0; i < sub_p.size(); i++) {
			if (i == 0) {
				Point_2f p1 = sub_p[sub_p.size() - 1];
				Point_2f p2 = sub_p[(i + 1) % sub_p.size()];
				Point_2f p0 = sub_p[i];
				double k1 = (p2.y - p0.y) / (p2.x - p0.x);
				double k2 = (p0.y - p1.y) / (p0.x - p1.x);
				Point_2f v1 = sub_p[i] - sub_p[(i + 1) % sub_p.size()];
				Point_2f v2 = sub_p[i] - sub_p[sub_p.size() - 1];
				double angle = acosf(v1.dot(v2) / (cv::norm(v1) * cv::norm(v2))) * 180.0 / CV_PI;
				if (abs(k1 - k2) > (0.1)) {
					pp.push_back(sub_p[i]);
				}
				else {
					cout << "ɸѡ��ĵ�" << sub_p[i].x << " " << sub_p[i].y << " " << endl;
				}
			}
			else {
				Point_2f p1 = sub_p[i - 1];
				Point_2f p2 = sub_p[(i + 1) % sub_p.size()];
				Point_2f p0 = sub_p[i];
				double k1 = (p2.y - p0.y) / (p2.x - p0.x);
				double k2 = (p0.y - p1.y) / (p0.x - p1.x);
				Point_2f v1 = sub_p[i] - sub_p[(i + 1) % sub_p.size()];
				Point_2f v2 = sub_p[i] - sub_p[i - 1];
				double angle = acosf(v1.dot(v2) / (cv::norm(v1) * cv::norm(v2))) * 180.0 / CV_PI;
				if (abs(k1 - k2) > (0.1)) {
					pp.push_back(sub_p[i]);
				}
				else {
					cout << "ɸѡ��ĵ�" << sub_p[i].x << " " << sub_p[i].y << " " << endl;
					cout << "ɸѡ��ĵ�Ƕ�ֵ" << angle << " " << endl;
				}
			}
		}
		polygonss.push_back(pp);
	}
		std::ofstream file("Simplify_polygons_test.txt");
		if (file.is_open()) {
			for (const std::vector<Point_2f>& vec : polygonss) {
				for (auto num : vec) {
					file << std::setprecision(11)<< num.x << " " << std::setprecision(11) <<num.y << " ";
				}
				file << std::endl;
				//�����޸ĺ��dp�Ķ���εĵ����ͱ���
				cout << "�޸ĺ��dp�Ķ���εĵ����ͱ����� "<<vec.size() << endl;

			}
			file.close();
			std::cout << "�����ѳɹ�д���ļ�" << std::endl;
		}
		else {
			std::cout << "�޷����ļ�" << std::endl;
		}
		 sum = 0;
		for (int i = 0; i < l.size(); i++) {
			cout <<"��" << i + 1 <<" �����" << "��֮����ٵĵ������:" << " " << l[i] - p[i].size() << endl;
			sum += l[i] - polygonss[i].size();
		}
		cout << "�򻯵������������" << sum << endl;
		cv::Mat image = Tools::reads_polygonss(p);
		cv::bitwise_not(image, image);
		cv::namedWindow("�򻯺�Ķ����", cv::WINDOW_AUTOSIZE);
		cv::imshow("�򻯺�Ķ����", image);


		cv::Mat image1 = Tools::reads_polygonss(polygonss);
		cv::bitwise_not(image1, image1);
		cv::namedWindow("��б�ʺ͵�����", cv::WINDOW_AUTOSIZE);
		cv::imshow("��б�ʺ͵�Ķ����", image1);
		for (auto ppo : polygonss) {
			std::cout <<"��б�ʺ͵�Ķ���ε������" << ppo.size() << std::endl;
		}
		cv::Mat originalimage = Tools::reads_polygonss(poly1);
		cv::bitwise_not(originalimage, originalimage);
		cv::namedWindow("ԭʼ�Ķ����", cv::WINDOW_AUTOSIZE);
		cv::imshow("ԭʼ�Ķ����", originalimage);
		cv::imwrite("original.png", originalimage);
		cv::imwrite("simplified_delete.png", image1);
		cv::waitKey(0);
		


}