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

void simpleDecomposition(cv::Mat img, cv::Mat predicted){
	double min,max;
	cv::minMaxLoc(predicted,&min,&max);
	int numClusters = (int)max;
	for(int k=0;k<=numClusters;k++){
		cv::Mat disp(img.rows,img.cols,CV_8UC3,cv::Scalar(255,255,255));
		for(int i=0;i<img.rows;i++){
			for(int j=0;j<img.cols;j++){
				pdd a =Maxwell2image(round(rgb2maxwell((int)img.at<cv::Vec3b>(i,j)[2],(int)img.at<cv::Vec3b>(i,j)[1],(int)img.at<cv::Vec3b>(i,j)[0])));
				if((int)predicted.at<float>(a.second,a.first)==-1)cout<<"invert x,y"<<endl;
				if((int)predicted.at<float>(a.second,a.first)==k){
					disp.at<cv::Vec3b>(i,j)[0] = img.at<cv::Vec3b>(i,j)[0];
					disp.at<cv::Vec3b>(i,j)[1] = img.at<cv::Vec3b>(i,j)[1];
					disp.at<cv::Vec3b>(i,j)[2] = img.at<cv::Vec3b>(i,j)[2];
				}
			}
		}
		string s = "disp "+to_string(k);
		cv::imshow(s,disp);
		cv::waitKey(0);
	}
}

cv::Mat EMMaxwellTriangle(map<pdd,long long int> M, int numClusters){
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

	cv::Mat img(im.rows,im.cols,CV_8UC3, cv::Scalar(0));
	cv::Mat predicted(im.rows,im.cols,CV_32FC1, cv::Scalar(-1));
	const cv::Scalar colors[] ={cv::Scalar(0,0,255), cv::Scalar(0,255,0), cv::Scalar(0,255,255), cv::Scalar(255,255,0), cv::Scalar(255,0,0)};

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

	cv::imshow("maxwell",img);
	cv::waitKey(0);
	return predicted;
}

int main(int argc, char** argv){
	cv::Mat im = cv::imread(argv[1],1);
	cv::imshow("image",im);
	cout<<"image size: "<<im.rows<<" X "<<im.cols<<endl;
	cv::waitKey(0);
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
	cv::Mat predicted = EMMaxwellTriangle(maxwellMap,4);
	simpleDecomposition(im,predicted);
	return 0;
}
