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
这行代码是使用CGAL库进行轮廓线简化的关键部分,解析如下:

CGAL::Polyline_simplification_2<K> simplifier:
这定义了一个Polyline_simplification_2对象simplifier,用于进行轮廓简化。
K是定义的数值精度类型,通常为CGAL::Exact_predicates_inexact_constructions_kernel。
simplifier.simplify:
调用simplify方法进行轮廓简化。
输入是原始轮廓点的迭代器begin和end。
输出是简化后的轮廓点容器simplified。
std::back_inserter:
用于构造一个back_insert_iterator,以便将simplify的输出push_back到simplified容器中。
原理:
Polyline_simplification_2内部使用Douglas-Peucker算法逼近原始轮廓。
不断递归分割,保留最远点,删除中间冗余点。
同时使用融合技术合并细小段。
删除重复点。
综合起来,可以有效简化轮廓,删除冗余点。
所以这行代码利用CGAL封装好的算法,非常方便地完成了轮廓点集的简化处理,不需要自己实现Douglas-Peucker等算法。
*/
//#include <CGAL/Polyline_simplification_2.h>
#include <limits>
#include <math.h>
#include <iomanip>
using namespace std;
typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
//定义多边形
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



double getDistance( Point_2f& p1,  Point_2f& p2) {
	double dx = p2.x - p1.x;
	double dy = p2.y - p1.y;
	return std::sqrt(dx * dx + dy * dy);
}

double getPerpendicularDistance( Point_2f& p,  Point_2f& start,  Point_2f& end) {
	double area = std::abs(0.5 * (start.x * end.y + end.x * p.y + p.x * start.y - end.x * start.y - p.x * end.y - start.x * p.y));//叉乘/2
	double length = getDistance(start, end);
	return (2.0 * area) / length;
}

void douglasPeuckerSimplify( std::vector<Point_2f> polyline,  std::vector<Point_2f>& simplified, double epsilon, double epsilonL, vector<pair<double, double>>& lengthAngles) {
	int end = polyline.size() - 1;
	//不可再简化，直接赋值
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
		if (maxDistance > epsilon  || (lengthAngles[index].second - 90.0) < 15.0)
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
	//在read_polygon可以得到定义

	//获取多边形的坐标，连线进行绘制图像  将图像转化为矩阵的化，因为连线了，所以多边形的顶点数目会增多，导致边界会变圆滑
	cv::Mat  image1 = Tools::read_polygon(polygon, resolution, abs(ISOValue), top_left_x, top_left_y, width);
	cv::namedWindow("hh", cv::WINDOW_AUTOSIZE);
	cv::imshow("hh", image1);
	//获取绘制位置
	int width_x = image1.cols;
	int width_y = image1.rows;
	vector<std::vector<float>> grid_df_1;
	//从图像中获取你要得到的矩阵
	Image_File_IO::convert_image_2_matrix(image1, grid_df_1, width_x, width_y);
	cv::namedWindow("image1", cv::WINDOW_AUTOSIZE);
	cv::imshow("image1", image1);
	cv::Mat canny_output;
	//图像二值化 有线的地方为255
	cv::threshold(image1, canny_output, 100, 255, cv::THRESH_BINARY);
	cv::namedWindow("ISO", cv::WINDOW_AUTOSIZE);
	cv::imshow("ISO", canny_output);
	cv::Mat ii = image1;
	cv::bitwise_not(ii, ii);
	cv::namedWindow("原始多边形1", cv::WINDOW_AUTOSIZE);
	cv::imshow("原始多边形1", ii);
	vector<vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;
	//版本问题，写法改变  CV_RETR_LIST和CV_CHAIN_APPROX_SIMPLE
	//获得图像矩阵的所有可能的轮廓 
	/*
图像中存在多个连通的区域：findContours() 函数会将图像中的每个连通区域都视为一个独立的轮廓。
即使你的图像中只有一个多边形，如果该多边形的内部存在空洞或其他连通区域，这些区域也可能被视为独立的轮廓。
分内多边形和外多边形
*/
	cv::findContours(canny_output, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	//对包围轮廓的矩形进行排序，以便按照矩形的面积从大到小的顺序对轮廓进行排序
	cout <<"轮廓数量：" << contours.size() << endl;
	for (int i = 0; i < contours.size(); i++)
	{
		//计算每个轮廓的包围矩阵
		cv::Rect reci = cv::boundingRect(contours[i]);
		for (size_t j = i + 1; j < contours.size(); j++)
		{//计算包围矩阵
			cv::Rect recj = cv::boundingRect(contours[j]);
			//对比轮廓对应的矩阵的面积
			//进行交换排序
			if (reci.area() < recj.area())
			{
				vector<cv::Point> tmp = contours[i];
				contours[i] = contours[j];
				contours[j] = tmp;
			}
		}
	}
	// 定义一个用于存储逼近后多边形的容器
	std::vector<std::vector<Point_2f>> approxContours(contours.size());
	/*
	curve：需要逼近的曲线，通常是由一系列点组成的轮廓。

	epsilon：逼近精度参数，代表逼近线段与原曲线之间的最大距离。
	较小的epsilon值会使逼近的多边形更接近原始曲线。

	closed：一个布尔值，指定曲线是否是闭合的。
	*/
	// 对每个轮廓进行多边形逼近  是将曲线近似表示为一系列点，并减少点的数量的一种算法
	double arlength1 = cv::arcLength(contours[0], true);
	double arlength2 = cv::arcLength(contours[1], true);
	cout << "周长为：" << arlength1 << endl;
	cout << "周长为：" << arlength2 << endl;
	double arlength3 = cv::arcLength(polygon, true);
	cout << "周长为：" << arlength3 << endl;
	/*******************************************************************************************/
	//计算出每个点与相邻两个顶点的长度和角度值，用于判断当前顶点是否应该继续存在 加上douglasPeuckerSimplify算法
	vector<pair<double,double>> lengthAngles;
	for (int i = 0; i <polygon.size(); i++) {
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
	//提取轮廓之后点的数量太多 所以优化不完整，这里考虑直接使用原来的多边形做输入
	for (size_t i = 0; i < contours.size(); i++)
	{
		// 指定逼近精度参数 epsilon，较小的阈值会使得逼近多边形更接近原始轮廓,因为是计算最大距离是否小于阈值
		// cv::arcLength(contours[i], true)计算给定轮廓的周长或弧长
		//double epsilon = 0.0018 * cv::arcLength(contours[i], true);
		double epsilon = 0.0056 * cv::arcLength(contours[i], true);
		double epsilonL = 0.6 * cv::arcLength(contours[i], true);
		cout << "contours的大小" << contours[0].size() << endl;
		// 进行多边形逼近
		//cv::approxPolyDP(polygon, approxContours[i], epsilon, true);
		douglasPeuckerSimplify(polygon, approxContours[i], epsilon, epsilonL, lengthAngles);

		cout << "epsilon的值: " << epsilon << endl;
		cout << "epsilonL的值: " << epsilonL << endl;
	}
	//根据最大的包围矩阵可以得到最大的轮廓
	int max_index = 0;
	if (ISOValue < 0)
	{
		max_index = 1;
	}
	//得到扩展后并且进行多边形拟合了的坐标
	vector<Point_2f> polyp;
	vector<Point_2f> polyc;
	vector<Point_2f> polyautodef;
//	cout << approxContours[0].size() << endl;
	cout << endl;
	//使用多边形拟合 得到的简化多边形的近似结果 ，用最大的轮廓的多边形的简化
	for (size_t j = 0; j < approxContours[0].size(); j++)
	{
	//	FT x = approxContours[0][j].x * resolution + top_left_x;
	//	FT y = approxContours[0][j].y * resolution + top_left_y;
	FT x = approxContours[0][j].x ;
	FT y = approxContours[0][j].y ;
		Point_2f pp(x, y);
	//	cout << fixed << std::setprecision(11) << x << " " << y << " " << endl;
		polyp.push_back(pp);
	}
	cout << endl;

	/*******************************************************************************************/
	/*
	//使用点之间的距离
	for (size_t j = 0; j < contours[0].size() - 1; j++)
	{
		FT x = contours[0][j].x * resolution + top_left_x;
		FT y = contours[0][j].y * resolution + top_left_y;
		FT x1 = contours[0][(j + 1) % contours[0].size()].x * resolution + top_left_x;
		FT y1 = contours[0][(j + 1) % contours[0].size()].y * resolution + top_left_y;
		Point_2f pp(x, y);
		Point_2f cc(x1, y1);
		double distancec = sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y));
		//直接剔除
		//通过点之间的距离来剔除不是很影响拓扑的点
		if (distancec > 2)
		{
	//		cout << fixed << std::setprecision(11) << x << " " << y << " " << endl;
			polyc.push_back(pp);
		}

	}


	/*************************************************************************************************/
	//使用斜率判断点是否在一条直线上
	
	
	/*
	vector<pair<int, int>> ks;
	FT x = contours[0][0].x * resolution + top_left_x;
	FT y = contours[0][0].y * resolution + top_left_y;
	FT x1 = contours[0][1].x * resolution + top_left_x;
	FT y1 = contours[0][1].y * resolution + top_left_y;
	FT kp = (y1 - y) / (x1 - x);
	for (size_t j = 1; j < contours[0].size(); j++)
	{
		FT x = contours[0][j].x * resolution + top_left_x;
		FT y = contours[0][j].y * resolution + top_left_y;
		FT x1 = contours[0][(j + 1) % contours[0].size()].x * resolution + top_left_x;
		FT y1 = contours[0][(j + 1) % contours[0].size()].y * resolution + top_left_y;
		Point_2f pp(x, y);
		Point_2f cc(x1, y1);
		FT k = (y1 - y) / (x1 - x);
		pair<int, int>p(k, j);
		if (k != kp)
		{
			ks.push_back(p);
			kp = k;
		}

	}
	int flag = 1;
	FT kpre = ks[0].first;
	for (size_t j = 1; j < ks.size(); j++)
	{
		FT x = contours[0][ks[j - 1].second % contours[0].size()].x * resolution + top_left_x;
		FT y = contours[0][ks[j - 1].second % contours[0].size()].y * resolution + top_left_y;
		FT x1 = contours[0][(ks[j].second) % contours[0].size()].x * resolution + top_left_x;
		FT y1 = contours[0][(ks[j].second) % contours[0].size()].y * resolution + top_left_y;
		FT x2 = contours[0][(ks[(j)].second + 1) % contours[0].size()].x * resolution + top_left_x;
		FT y2 = contours[0][(ks[(j)].second + 1) % contours[0].size()].y * resolution + top_left_y;
		FT k1 = ks[j].first;
		if (fabs(k1 - kpre) == 0.0 && flag == 1)
		{
			flag = 0;
			Point_2f t(x, y);
	//		cout << fixed << std::setprecision(11) << x << " " << y << " " << endl;
			polyautodef.push_back(t);
		}
		else if (fabs(k1 - kpre) > 0.4 && fabs(sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))) > sqrt(3))
		{
			flag = 1;
			kpre = k1;
			Point_2f t1(x1, y1);
			Point_2f t2(x2, y2);
			if (t1 == t2)
			{
				polyautodef.push_back(t1);
			}
			else {
				polyautodef.push_back(t1);
				polyautodef.push_back(t2);
			}
		//	cout << fixed << std::setprecision(11) << x1 << " " << y1 << " " << endl;
		//	cout << fixed << std::setprecision(11) << x2 << " " << y2 << " " << endl;


		}

	}
	*/

//	cout << endl;
//	cout << "通过斜率剔除的点数量 ：" << polyautodef.size() << endl;
	/*
Point_2f mainxy(0, 0);
for (size_t j = 0; j < contours[max_index].size() - 1; j++)
{
	FT x = contours[max_index][j].x * resolution + top_left_x;
	FT y = contours[max_index][j].y * resolution + top_left_y;
	FT x1 = contours[max_index][j + 1].x * resolution + top_left_x;
	FT y1 = contours[max_index][j + 1].y * resolution + top_left_y;
	Point_2f pp(x, y);
	Point_2f cc(x1, y1);
	mainxy = pp;
	double distancec = sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y));
	//对 通过点之间的距离来剔除不是很影响拓扑的点，
	//但是  不是直接剔除还将作为一个均值的参数影响最终点的位置
	if (distancec < 2)
	{
		mainxy.x = (x + x1) / 2;
		mainxy.y = (y + y1) / 2;
	}
	else {
		cout << fixed << std::setprecision(3) << x << " " << y << " " << endl;
		polyc.push_back(mainxy);
	}
}
	*/

	/*
	* //	cv::Mat image22 = Tools::read_polygon(polyc, resolution, abs(ISOValue), top_left_x, top_left_y, width);

//	cv::Mat imagek = Tools::read_polygon(polyautodef, resolution, abs(ISOValue), top_left_x, top_left_y, width);
	// 二值反转
	cv::bitwise_not(image, image);
	cv::namedWindow("扩展后的多边形", cv::WINDOW_AUTOSIZE);
	cv::imshow("扩展后的多边形", image);

	cv::Mat image = Tools::read_polygon(polyp, resolution, abs(ISOValue), top_left_x, top_left_y, width);
	cv::bitwise_not(image, image);
	cv::namedWindow("扩展后的多边形", cv::WINDOW_AUTOSIZE);
	cv::imshow("扩展后的多边形", image);
	std::cout << polyp.size() << std::endl;
	cv::waitKey(0);
	cv::waitKey(0);
	//cv::namedWindow("exendedimage1", cv::WINDOW_AUTOSIZE);
	//cv::imshow("exendedimage1", image22);
	//cv::namedWindow("exendedimage2", cv::WINDOW_AUTOSIZE);
	//cv::imshow("exendedimage2", imagek);
	cv::waitKey(0);
	*/
	
	return polyp;
};

cv::Mat Tools::read_polygon(const std::vector<Point_2f>& polygon, double& resolution,
	double ISOValue, double& top_left_x, double& top_left_y, int& width)
{
	//生成多边形的边框  用于生成界面的展示窗口
	cv::Rect rec = cv::boundingRect(polygon);
	//Bbox2f rec = Global::bbox_of(polygon);
	//引用
	resolution = 0.8;//定义resolution
	width = 1.0 * (rec.width) / resolution;//定义width
	//加了四十 20+20 = 40
	float dis_add = ISOValue + 20;
	//定义要扩张的左上角的坐标位置
	top_left_x = rec.x - ISOValue - dis_add * resolution;
	top_left_y = rec.y - ISOValue - dis_add * resolution;
	//定义要扩张的右下角的坐标位置
	double buttom_right_x = rec.x + (rec.width) + ISOValue + dis_add * resolution;
	double buttom_right_y = rec.y + (rec.height) + ISOValue + dis_add * resolution;

	//窗口大小
	int width_x = (buttom_right_x - top_left_x) / resolution;
	int width_y = (buttom_right_y - top_left_y) / resolution;

	cv::Mat drawing = cv::Mat::zeros(cv::Size(width_x, width_y), CV_8UC1);
	//绘制原始多边形
	for (int j = 0; j < polygon.size(); j++)
	{

		//绘制多边形，点与点之间的连线
		Point_2f p;
		//得到起点，因为原始数据太大了，在窗口显示不下来，相当于进行一个缩放
		int begin_x = (polygon[j].x - top_left_x) / resolution;
		int begin_y = (polygon[j].y - top_left_y) / resolution;
		int end_x = (polygon[(j + 1) % polygon.size()].x - top_left_x) / resolution;
		int end_y = (polygon[(j + 1) % polygon.size()].y - top_left_y) / resolution;
		cv::Scalar color = cv::Scalar(255);
		line(drawing, Point_2f(begin_x, begin_y), Point_2f(end_x, end_y), color, 1);
	}
	//返回多边形的图像 cv::Mat drawing
	return drawing;
};

cv::Mat Tools::reads_polygonss(const std::vector<vector<Point_2f>>& polygons)
{
	
	cv::Mat drawing = cv::Mat::zeros(cv::Size(2000, 2000), CV_8UC1);
	//绘制原始多边形
	std::vector<std::vector<cv::Point>> intPolygons;
	for (const auto& polygon : polygons) {
		std::vector<cv::Point> intPolygon;
		// 遍历多边形中的每个点  
		for (const auto& point : polygon) {
			cv::Point intPoint(cv::int32_t(point.x), cv::int32_t(point.y));
			// 将转换后的点添加到 intPolygon 中  
			intPolygon.push_back(intPoint);
		}
		// 将转换后的多边形添加到 intPolygons 中  
		intPolygons.push_back(intPolygon);
	}
	// 在图像上绘制顶点
	for (const auto& p : polygons)
	{
		/*
		int a = 1;
		for (const auto& v : p) {
			cv::circle(drawing, v, 1, cv::Scalar(255), -1);  // 绘制红色圆点											 
			    
				Point_2f textPosition(v.x, v.y);  // 文本的位置
				std::string text = std::to_string(a);  // 将数字转换为字符串
				a++;
				cv::Scalar textColor(255);  // 文本的颜色（绿色）
				int fontFace = cv::FONT_HERSHEY_SIMPLEX;  // 字体类型
				double fontScale = 0.3;  // 字体缩放比例
				int thickness = 1;  // 文本线宽
	cv::putText(drawing, text, textPosition, fontFace, fontScale, textColor, thickness);
			
		}	
		*/
		int a = 1;
		for (const auto& v : p) {
			cv::circle(drawing, v, 1, cv::Scalar(255), -1);  // 绘制红色圆点											 

			Point_2f textPosition(v.x, v.y);  // 文本的位置
			std::string text = std::to_string(a);  // 将数字转换为字符串
			a++;
			cv::Scalar textColor(255);  // 文本的颜色（绿色）
			int fontFace = cv::FONT_HERSHEY_SIMPLEX;  // 字体类型
			double fontScale = 0.3;  // 字体缩放比例
			int thickness = 1;  // 文本线宽
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

				//绘制多边形，点与点之间的连线
				int begin_x = polygon[j].x;
				int begin_y = polygon[j].y;
				int end_x = polygon[(j + 1) % polygon.size()].x;
				int end_y = polygon[(j + 1) % polygon.size()].y;
				cv::Scalar color = cv::Scalar(255);
				line(drawing, Point_2f(begin_x, begin_y), Point_2f(end_x, end_y), color, 1);
			}
		}
		*/


		//返回多边形的图像 cv::Mat drawing
		return drawing;
	
};

void Tools::convert_matrix(const std::vector<std::vector<float>>& initial_grid, cv::Mat& mat, const int& width_x, const int& width_y, float Value_thres)
{
	mat = cv::Mat::zeros(cv::Size(width_x, width_y), CV_8UC1);

	for (unsigned int x = 0; x < width_x; x++)
	{
		for (unsigned int y = 0; y < width_y; y++)
		{
			//内部小于Value_thres的区域设置为白色，外部为黑色
			//相当于在扩展距离之内的是白色，之外的为黑色
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
			//记录多边形连接处的坐标,以及将其值设置为INF
			//相当于白线处为零，然后像素值为零的地方设置为无穷
			if (row_gray > 100)
			{
				//有像素设置为零表示是边界
				initial_grid[x][y] = 0;
			}
			else
			{
				//无像素的地方设置为无穷
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

			while (iss >> a>>b) {
				vec.push_back(Point_2f(a,b));
			}
			poly.push_back(vec);
			poly1.push_back(vec);
		}
		input.close();
	}
	else {
		std::cout << "无法打开文件" << std::endl;
	}
	vector<int> l;
	for (size_t i = 0; i < poly.size(); i++)
	{
		l.push_back(poly[i].size());
	}
	cout << "建筑个数" <<l.size() << endl;
	int sum = 0;
	for (size_t i = 0; i < l.size(); i++)
	{
		cout << "原始的多边形的点数和边数： " << l[i] << endl;
		sum += l[i];
	}
	cout << "原始的多边形的点数和边数： " << sum << endl;
	
	for (size_t i = 0; i < poly.size(); i++)
	{
		vector<Point_2f> c = Tools::polygon_ISO(poly[i], 20.0);


		p.push_back(c);
	}
	std::vector<vector<Point_2f>> polygonss;
	


	
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
					cout << "筛选后的点" << sub_p[i].x << " " << sub_p[i].y << " " << endl;
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
					cout << "筛选后的点" << sub_p[i].x << " " << sub_p[i].y << " " << endl;
					cout << "筛选后的点角度值" << angle << " " << endl;
				}
			}
		}
		polygonss.push_back(pp);
	}
	
	
	

//文科楼 artsci	后处理
/*
	文科楼 artsci
	for (int i = 0; i < polygonss.size();i++) {
		for (int j = 0; j < polygonss[i].size(); j++) {





	if (i == 0) {

		cv::Point2f direction = polygonss[i][32] - polygonss[i][33];  // 移动的方向向
		polygonss[i][33] += 0.006 * direction;
		polygonss[i][34] += 0.006 * direction;
	}
	if (i == 1) {
		cv::Point2f direction = polygonss[i][34] - polygonss[i][0];  // 移动的方向向
		polygonss[i][0] += 0.002 * direction;
	}
}
	}
*/
	/*
	for (int i = 0; i < polygonss.size(); i++) {
	

		if (i == 0)
		{
			// 删除索引为 2 的元素
			for (int j = 0; j < 2; j++)
			polygonss[i].erase(polygonss[i].begin() + 7);

			}
		if (i == 1)
		{
			// 删除索引为 2 的元素
			for (int j = 0; j < 10; j++)
				polygonss[i].erase(polygonss[i].begin() + 15);

		}
	
		
	}
	for (int i = 0; i < polygonss.size(); i++) {


		if (i == 0)
		{
			// 删除索引为 2 的元素
			for (int j = 0; j < 4; j++)
				polygonss[i].erase(polygonss[i].begin() + 16);

		}
		if (i == 1)
		{
			// 删除索引为 2 的元素
			for (int j = 0; j < 2; j++)
				polygonss[i].erase(polygonss[i].begin() + 16);

		}


	}
	
	*/
	

//洞洞楼yh2场景的 移动方向
	/*
	yh2场景的 移动方向
	cv::Point2f direction = polygonss[3][0] - polygonss[3][1];  // 移动的方向向


				for (int i = 0; i < polygonss.size(); i++) {
					for (int j = 0; j < polygonss[i].size(); j++) {





						if (i == 2) {


							polygonss[i][j] += 0.02 * direction;
						}
						if (i == 3) {

							polygonss[i][j] += 0.02 * direction;
						}
					}
				}
	*/
	
	
	std::ofstream file("Simplify_polygon_result_Hitech.txt");
	

	if (file.is_open()) {
		for (const std::vector<Point_2f>& vec : polygonss) {
			for (auto num : vec) {
				file << std::setprecision(11)<< num.x << " " << std::setprecision(11) <<num.y << " ";
			}
			file << std::endl;
			//经过修改后的dp的多边形的点数和边数
			cout << "修改后的dp的多边形的点数和边数： "<<vec.size() << endl;

		}
		file.close();
		std::cout << "数据已成功写入文件" << std::endl;
	}
	else {
		std::cout << "无法打开文件" << std::endl;
	}
	sum = 0;
	for (int i = 0; i < l.size(); i++) {
		cout <<"第" << i + 1 <<" 多边形" << "简化之后减少的点的数量:" << " " << l[i] - p[i].size() << endl;
		sum += l[i] - polygonss[i].size();
	}
	cout << "简化点的数量的总数" << sum << endl;
	cv::Mat image = Tools::reads_polygonss(p);
	cv::bitwise_not(image, image);
	cv::namedWindow("简化后的多边形", cv::WINDOW_AUTOSIZE);
	cv::imshow("简化后的多边形", image);

	
	cv::Mat image1 = Tools::reads_polygonss(polygonss);
	cv::bitwise_not(image1, image1);
	cv::namedWindow("简化斜率和点多边形", cv::WINDOW_AUTOSIZE);
	cv::imshow("简化斜率和点的多边形", image1);
	for (auto ppo : polygonss) {
		std::cout <<"简化斜率和点的多边形点的数量" << ppo.size() << std::endl;
	}

	cv::Mat originalimage = Tools::reads_polygonss(poly1);
	cv::bitwise_not(originalimage, originalimage);
	cv::namedWindow("原始的多边形", cv::WINDOW_AUTOSIZE);
	cv::imshow("原始的多边形", originalimage);

	cv::imwrite("original.png", originalimage);
	cv::imwrite("simplified_delete.png", image1);
	cv::waitKey(0);
	
	
	
}