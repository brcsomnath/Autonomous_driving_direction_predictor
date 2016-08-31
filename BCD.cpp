
#include <cv.h>
#include <highgui.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <strings.h>
#include <map>
#include <utility>
#include <cmath>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
using namespace std;

#define pdd pair<double,double>
#define pff pair<float,float>

pdd rgb2maxwell(int r, int g, int b){
	pdd a;
	if(r==0)r=1;
	if(g==0)g=1;
	if(b==0)b=1;
	double x1 = log2(255.0/r);
	double x2 = log2(255.0/g);
	double x3 = log2(255.0/b);
	double mag = sqrt(x1*x1 + x2*x2 + x3*x3);
	if(mag<=0.0001){x1=0;x2=0;x3=0;}
	else {x1/=mag; x2/=mag; x3/=mag;}
	//cout<<r<<" "<<g<<" "<<b<<" "<<x1<<" "<<x2<<" "<<x3<<" "<<mag<<endl;
	a.first = (x1-x2)/sqrt(2);
	a.second = (x3*sqrt(2.0/3.0))-((x1+x2)/sqrt(6)) ;
	//cout<<a.first<<" "<<a.second<<endl;
	return a;
}

pdd round(pdd a){
	long int x = (int)(a.first*100);
	long int y = (int)(a.second*100);
	pdd b;
	b.first = (double)1.0*x/100.0;
	b.second = (double)1.0*y/100.0;
	return b;
}

pdd Maxwell2image(pdd a){
	double a11 = 250*sqrt(2);
	double a12 = 0.0;
	double c1 = 250.0;
	double a21 = 0.0;
	double a22 = -432*sqrt(2.0/3.0);
	double c2 = 288 ; 
	pdd b;
	b.first = (int)(a11*a.first + a12*a.second + c1);
	b.second = (int)(a21*a.first + a22*a.second + c2);
	return b;
}

void showMaxwellTriangle(map<pdd,long long int> M){
	cv::Mat im(600,600,CV_8UC1,cv::Scalar(0));
	cv::line(im,cv::Point(250,0),cv::Point(0,433),255,1);
	cv::line(im,cv::Point(250,0),cv::Point(500,433),255,1);
	cv::line(im,cv::Point(0,432),cv::Point(500,432),255,1);
	double a11 = 250*sqrt(2);
	double a12 = 0.0;
	double c1 = 250.0;
	double a21 = 0.0;
	double a22 = -432*sqrt(2.0/3.0);
	double c2 = 288 ; 
	long long int maxCount = 0;
	for(map<pdd,long long int>::iterator it=M.begin();it!=M.end();++it){
		if(it->second > maxCount)maxCount = it->second;
	}
	for(map<pdd,long long int>::iterator it=M.begin();it!=M.end();++it){
		it->second = (int)(it->second*255.0/maxCount);
		if(it->second>255)it->second = 255;
	}
	for(map<pdd,long long int>::iterator it=M.begin();it!=M.end();++it){
		double a=it->first.first;
		double b=it->first.second;
		double x = a11*a + a12*b + c1;
		double y = a21*a + a22*b + c2;
		if((int)im.at<uchar>((int)y,(int)x)<it->second)im.at<uchar>((int)y,(int)x) = (int)it->second;
		//im.at<uchar>((int)y,(int)x) = 255;
		cv::circle(im,cv::Point((int)x,(int)y),2,(int)it->second,-1);
		//cout<<x<<" "<<y<<" "<<it->second<<endl;
	}
	cv::imshow("maxwell",im);
	cv::waitKey(0);
}

void simpleDecomposition(cv::Mat img, cv::Mat predicted, pdd roadColor){
	pdd roadColorInImage = Maxwell2image(round(roadColor));
	int roadID = (int)predicted.at<float>(roadColorInImage.second,roadColorInImage.first);
	cout<<"roadID:"<<roadID<<endl;
	double min,max;
	cv::minMaxLoc(predicted,&min,&max);
	//int numClusters = (int)max;
	cv::Mat disp(img.rows,img.cols,CV_8UC3,cv::Scalar(0,0,0));
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			pdd a =Maxwell2image(round(rgb2maxwell((int)img.at<cv::Vec3b>(i,j)[2],(int)img.at<cv::Vec3b>(i,j)[1],(int)img.at<cv::Vec3b>(i,j)[0])));
			if((int)predicted.at<float>(a.second,a.first)==-1)cout<<"invert x,y"<<endl;
			if((int)predicted.at<float>(a.second,a.first)==roadID){
				disp.at<cv::Vec3b>(i,j)[0] = img.at<cv::Vec3b>(i,j)[0];
				disp.at<cv::Vec3b>(i,j)[1] = img.at<cv::Vec3b>(i,j)[1];
				disp.at<cv::Vec3b>(i,j)[2] = img.at<cv::Vec3b>(i,j)[2];
			}
		}
	}
		string s = "disp_"+to_string(roadID);
		cv::imshow(s,disp);
		cv::waitKey(0);
		cv::imwrite(s+".jpg",disp);
}

void printMat(cv::Mat M){
	for(int i=0;i<M.rows;i++){
		for(int j=0;j<M.cols;j++)cout<<M.at<double>(i,j)<<",";
		cout<<endl;
	}
}

void linearDecomposition(cv::Mat img, cv::Mat means){
	cv::Mat means_tfo(means.rows,means.cols,CV_64FC1);
	for(int i=0;i<means.rows;i++){
		means_tfo.at<double>(i,1) = (means.at<double>(i,1)-250.0)/(250*sqrt(2.0));
		means_tfo.at<double>(i,0) = (means.at<double>(i,0)-288.0)*sqrt(3.0/2.0)/(-432.0);
	}
	cv::Mat mean_colors(3,means.rows,CV_64FC1);
	for(int i=0;i<means.rows;i++){
		double a = means_tfo.at<double>(i,1);
		double b = means_tfo.at<double>(i,0);
		mean_colors.at<double>(0,i) = (a/sqrt(2.0)) - (b/sqrt(6.0)) + (1.0/3.0);
		mean_colors.at<double>(1,i) = (-a/sqrt(2.0)) - (b/sqrt(6.0)) + (1.0/3.0);
		mean_colors.at<double>(2,i) = 1.0 - mean_colors.at<double>(0,i) - mean_colors.at<double>(1,i);
	}
	cv::Mat mean_colors_t,multiplied,inverted,mean_color_tfo_matrix;
	cout<<"1"<<mean_colors.size()<<endl;
	printMat(mean_colors);
	cv::transpose(mean_colors,mean_colors_t);
	cout<<"2"<<mean_colors_t.size()<<endl;
	printMat(mean_colors_t);
	cv::mulTransposed(mean_colors,multiplied,true);
	cout<<"3"<<multiplied.size()<<" "<<cv::determinant(multiplied)<<endl;
	printMat(multiplied);
	//cv::invert(multiplied,inverted);
	inverted = multiplied.inv();
	cout<<"4"<<inverted.size()<<endl;
	printMat(inverted);
	mean_color_tfo_matrix = inverted*mean_colors_t;
	cout<<"5"<<mean_color_tfo_matrix.size()<<endl;
	printMat(mean_color_tfo_matrix);
	cv::Mat disp[means.rows];
	for(int i=0;i<means.rows;i++)disp[i] = cv::Mat(img.rows,img.cols,CV_8UC3,cv::Scalar(255,255,255));
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			cv::Mat colors(3,1,CV_64FC1);
			colors.at<double>(0,0) = log2(255.0/img.at<cv::Vec3b>(i,j)[2]);
			colors.at<double>(1,0) = log2(255.0/img.at<cv::Vec3b>(i,j)[1]);
			colors.at<double>(2,0) = log2(255.0/img.at<cv::Vec3b>(i,j)[0]);
			cout<<colors.at<double>(0,0)<<" "<<colors.at<double>(1,0)<<" "<<colors.at<double>(2,0)<<endl;
			cv::Mat prob = mean_color_tfo_matrix*colors;
			cout<<prob.at<double>(0,0)<<" "<<prob.at<double>(0,1)<<" "<<prob.at<double>(0,2)<<" "<<prob.at<double>(0,3)<<endl;
			cv::waitKey(0);
			for(int k=0;k<means.rows;k++){
				if(prob.at<double>(0,k)<0)continue;
				disp[k].at<cv::Vec3b>(i,j)[0] = (int)prob.at<double>(0,k)*img.at<cv::Vec3b>(i,j)[0];
				disp[k].at<cv::Vec3b>(i,j)[1] = (int)prob.at<double>(0,k)*img.at<cv::Vec3b>(i,j)[1];
				disp[k].at<cv::Vec3b>(i,j)[2] = (int)prob.at<double>(0,k)*img.at<cv::Vec3b>(i,j)[2];
			}
		}
	}
	for(int i=0;i<means.rows;i++){
		string s = "disp-"+to_string(i);
		cv::imshow(s,disp[i]);
		cv::waitKey(0);
		cv::imwrite(s+".jpg",disp[i]);
	}

}

cv::Mat EMMaxwellTriangle(map<pdd,long long int> M, int numClusters, pdd roadColor){
	cv::Mat im(600,600,CV_32FC1,cv::Scalar(0));
	double a11 = 250*sqrt(2);
	double a12 = 0.0;
	double c1 = 250.0;
	double a21 = 0.0;
	double a22 = -432*sqrt(2.0/3.0);
	double c2 = 288 ; 
	/*long long int maxCount = 0;
	for(map<pdd,long long int>::iterator it=M.begin();it!=M.end();++it){
		if(it->second > maxCount)maxCount = it->second;
	}
	for(map<pdd,long long int>::iterator it=M.begin();it!=M.end();++it){
		it->second = (int)(it->second*255.0/maxCount);
		if(it->second>255)it->second = 255;
	}*/
	for(map<pdd,long long int>::iterator it=M.begin();it!=M.end();++it){
		double a=it->first.first;
		double b=it->first.second;
		double x = a11*a + a12*b + c1;
		double y = a21*a + a22*b + c2;
		if(im.at<float>((int)y,(int)x)<it->second)im.at<float>((int)y,(int)x) = it->second;
		//im.at<uchar>((int)y,(int)x) = 255;
		//cv::circle(im,cv::Point((int)x,(int)y),2,(int)it->second,-1);
		//cout<<x<<" "<<y<<" "<<it->second<<endl;
	}
	cv::imshow("maxwell",im);
	cv::waitKey(0);

	vector<pair<pff,float> > V;
	cv::Mat labels;
	for(int i=0;i<im.rows;i++){
		for(int j=0;j<im.cols;j++){
			if(im.at<float>(i,j)==0)continue;
			else V.push_back(make_pair(make_pair(i,j),im.at<float>(i,j)));
		}
	}
	cv::Mat im2(V.size(),2,CV_32FC1);
	for(unsigned int i=0;i<V.size();i++){
		im2.at<float>(i,0) = V[i].first.first;
		im2.at<float>(i,1) = V[i].first.second;
		//im2.at<float>(i,2) = V[i].second;
	}
	cout<<"im2 size "<<im2.size()<<endl;
	cv::Ptr<cv::ml::EM> em_model = cv::ml::EM::create();
	em_model->setClustersNumber(numClusters);
	em_model->setCovarianceMatrixType(cv::ml::EM::COV_MAT_SPHERICAL);
	em_model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 300, 0.1));
	em_model->trainEM( im2, cv::noArray(), labels, cv::noArray() );
	cv::Mat means = em_model->getMeans();
	/*cout<<means.type()<<endl;
	for(int i=0;i<means.rows;i++)
		cout<<means.at<double>(i,0)<<" "<<means.at<double>(i,1)<<endl;
	*/

	cv::Mat img(im.rows,im.cols,CV_8UC3, cv::Scalar(0));
	cv::Mat predicted(im.rows,im.cols,CV_32FC1, cv::Scalar(-1));
	const cv::Scalar colors[] ={cv::Scalar(0,0,255), cv::Scalar(0,255,0), cv::Scalar(0,255,255), cv::Scalar(255,255,0), cv::Scalar(255,0,0)};
	for(int i=0;i<means.rows;i++)
		cv::circle(img,cv::Point(means.at<double>(i,1),means.at<double>(i,0)),10,colors[i],-1);
	for(int i = 0; i < img.rows; i++ )
    {
        for(int j = 0; j < img.cols; j++ )
        {
        	cv::Mat sample( 1, 2, CV_32FC1 );
        	sample.at<float>(0) = (float)i;
            sample.at<float>(1) = (float)j;
            //sample.at<float>(2) = (float)im.at<float>(i,j);
            if((float)im.at<float>(i,j)==0)continue;
            cv::Mat means = em_model->getMeans();
            int response = cvRound(em_model->predict2( sample, cv::noArray() )[1]);
            cv::Scalar c = colors[response];
            predicted.at<float>(i,j) = (float)response;
            cv::circle( img, cv::Point(j, i), 1, c*0.75, -1 );
        }
    }

    pdd roadColorInImage = Maxwell2image(round(roadColor));
    cv::circle(img,cv::Point(roadColorInImage.first,roadColorInImage.second),5,cv::Scalar(255,255,255),-1);
	cv::imshow("maxwell",img);
	cv::waitKey(0);
	return predicted;  /// uncomment this line for using simpleDecomposition
	//return means;	   /// uncomment this line for using linearDecompositon
}


int donePoints=0;
vector<cv::Point> testPoints;
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
     if  ( event == cv::EVENT_LBUTTONDOWN )
     {
     	testPoints.push_back(cv::Point(y,x));
		donePoints++;
     }
     
}


int main(int argc, char** argv){
	if(argc!=5){
		cout<<"\nArguments list:\n1. Image name with address on disk\n2.pyrDown yes/no -> 1 for Yes, 2 for No\n3. No. of road points\n4. Number of clusters"<<endl;
		return 0;
	}
	int numOfPoints = atoi(argv[3]);
	cv::Mat im = cv::imread(argv[1],1);
	if (int(argv[2][0])==49)cv::pyrDown(im,im);
	cout<<"image size: "<<im.rows<<" X "<<im.cols<<endl;
	cv::namedWindow("image",1);
	cv::setMouseCallback("image",CallBackFunc,NULL);
	cv::imshow("image",im);
	while(donePoints<numOfPoints){
		cv::waitKey(30);
	}
	map<pdd,long long int> maxwellMap;
	for(int i=0;i<im.rows;i++){
		for(int j=0;j<im.cols;j++){
			pdd a = rgb2maxwell((int)im.at<cv::Vec3b>(i,j)[2],(int)im.at<cv::Vec3b>(i,j)[1],(int)im.at<cv::Vec3b>(i,j)[0]);
			a = round(a);
			if(maxwellMap.count(a)==0)maxwellMap.insert(make_pair(a,1));
			else maxwellMap[a]+=1;
		}
	}
	/*for (map<pdd,long long int>::iterator it=maxwellMap.begin(); it!=maxwellMap.end(); ++it)
    	cout <<"("<< it->first.first <<","<<it->first.second<<")" << " => " << it->second << endl;*/
	//showMaxwellTriangle(maxwellMap);
	pdd roadColors[numOfPoints];
	pdd avgRoadColor(0,0);
	for(int i=0;i<numOfPoints;i++){
		int x = (int)testPoints[i].x;
		int y = (int)testPoints[i].y;
		cout<<testPoints[i].x<<testPoints[i].y<<endl;
		roadColors[i] = rgb2maxwell((int)im.at<cv::Vec3b>(x,y)[2],(int)im.at<cv::Vec3b>(x,y)[1],(int)im.at<cv::Vec3b>(x,y)[0]);
		avgRoadColor.first+=roadColors[i].first;
		avgRoadColor.second+=roadColors[i].second;
	}
	avgRoadColor.first/=numOfPoints;avgRoadColor.second/=numOfPoints;
	cv::Mat predicted = EMMaxwellTriangle(maxwellMap,atoi(argv[4]),avgRoadColor);
	simpleDecomposition(im,predicted,avgRoadColor);
	return 0;
}