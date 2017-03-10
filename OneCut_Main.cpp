//Copyright (c) 2014-2017, Lena Gorelick, Katrina Hoffert
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions are met:
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//    * Neither the name of the University of Western Ontarior nor the
//      names of its contributors may be used to endorse or promote products
//      derived from this software without specific prior written permission.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
//DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//
//THIS SOFTWARE IMPLEMENTS THE OneCut ALGORITHM THAT USES SCRIBBLES AS HARD CONSTRAINTS.
//PLEASE USE THE FOLLOWING CITATION:
//
//@inproceedings{iccv2013onecut,
//	title	= {Grabcut in One Cut},
//	author	= {Tang, Meng and Gorelick, Lena and Veksler, Olga and Boykov, Yuri},
//	booktitle={International Conference on Computer Vision},
//	month	= {December},
//	year	= {2013}}
//
//THIS SOFTWARE USES maxflow/min-cut CODE THAT WAS IMPLEMENTED BY VLADIMIR KOLMOGOROV,
//THAT CAN BE DOWNLOADED FROM http://vision.csd.uwo.ca/code/.
//PLEASE USE THE FOLLOWING CITATION:
//
//@ARTICLE{Boykov01anexperimental,
//    author = {Yuri Boykov and Vladimir Kolmogorov},
//    title = {An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision},
//    journal = {IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE},
//    year = {2001},
//    volume = {26},
//    pages = {359--374}}
//
//THIS SOFTWARE USES OpenCV 2.4.3 THAT CAN BE DOWNLOADED FROM http://opencv.org

#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "graph.h"

using namespace std;
using namespace cv;

// images
Mat inputImg, showImg, binPerPixelImg, showEdgesImg, segMask, segShowImg;
// mask
Mat fgScribbleMask, bgScribbleMask, fgScribbleMaskAll, bgScribbleMaskAll;

int numUsedBins = 0;
float varianceSquared = 0;
int scribbleRadius = 10;

// default arguments
float bha_slope = 0.5f;
int numBinsPerChannel = 16;
float EDGE_STRENGTH_WEIGHT = 0.95f;

int fgLabel = 1;
int bgLabel = 2;

const float INT32_CONST = 1000;
const float HARD_CONSTRAINT_CONST = 1000;

#define NEIGHBORHOOD_4_TYPE 1;
const int NEIGHBORHOOD = NEIGHBORHOOD_4_TYPE;


//************************************
// F u n c t i o n     d e c l a r a t i o n s 

// Print command line usage
void printHelp();

// Init all images/vars
int init(char * imgFileName);

// Loads the strokes file into the FG/BG scribble masks
int loadStrokes(char * strokesFileName, Mat & fgScribbleMask, Mat & bgScribbleMask);

// Set bin index for each image pixel, store it in binPerPixelImg
void getBinPerPixel(Mat & binPerPixelImg, Mat & inputImg, int numBinsPerChannel, int & numUsedBins);

// compute the variance of image edges between neighbors
void getEdgeVariance(Mat & inputImg, Mat & showEdgesImg, float & varianceSquared);

void getColorSepE(int & colorSep_E, int & hardConstraints_E);

typedef Graph<int,int,int> GraphType;
GraphType *myGraph;


//***********************************
// M a i n 

int main(int argc, char *argv[])
{
    char * imgFileName = NULL;
    char * strokesFileName = NULL;
    char * outputFileName = NULL;

    for(int arg = 1; arg < argc; ++arg)
    {
        if (argv[arg] == string("--bins") && argc > arg + 1) {
            numBinsPerChannel = atoi(argv[++arg]);
        }
        else if (argv[arg] == string("--slope") && argc > arg + 1) {
            bha_slope = (float)atof(argv[++arg]);
        }
        else if (argv[arg] == string("--fg-label") && argc > arg + 1) {
            fgLabel = atoi(argv[++arg]);
        }
        else if (argv[arg] == string("--bg-label") && argc > arg + 1) {
            bgLabel = atoi(argv[++arg]);
        }
        else if (argv[arg] == string("--output") && argc > arg + 1) {
            outputFileName = argv[++arg];
        }
        else if (argv[arg] == string("--help")) {
            printHelp();
            return 0;
        }
        else if (!imgFileName) {
            imgFileName = argv[arg];
        }
        else if (!strokesFileName) {
            strokesFileName = argv[arg];
        }
        else {
            cout << "Invalid argument " << argv[arg] << endl;
        }
    }

    // Invalid arguments
    if (!imgFileName || !strokesFileName)
    {
        cout << "Invalid arguments" << endl << endl;
        printHelp();
        return -1;
    }
    
    cout << "Input image: " << imgFileName << endl;
    cout << "Strokes file: " << strokesFileName << endl;
    cout << "FG label: " << fgLabel << ", BG label: " << bgLabel << endl;
	cout << "Using " << numBinsPerChannel <<  " bins per channel " << endl; 
	cout << "Using colorSep_slope = " << bha_slope << endl;
	
	if (init(imgFileName)==-1)
	{
		cout <<  "Could not initialize" << endl ;
		return -1;
	}
    
	if (loadStrokes(strokesFileName, fgScribbleMask, bgScribbleMask)==-1)
	{
		cout <<  "Could not load scribbles" << std::endl;
        return -1;
	}

    // Segment
    cout << "\n--- Segmenting ---" << endl;
	cout << "Setting the hard constraints..." << endl;
	for(int i=0; i<inputImg.rows; i++)
	{
		for(int j=0; j<inputImg.cols; j++) 
		{
			// this is the node id for the current pixel
			GraphType::node_id currNodeId = i * inputImg.cols + j;
	
			// add hard constraints based on scribbles
			if (fgScribbleMask.at<uchar>(i,j) == 255)
				myGraph->add_tweights(currNodeId,(int)ceil(INT32_CONST * HARD_CONSTRAINT_CONST + 0.5),0);
			else if (bgScribbleMask.at<uchar>(i,j) == 255)
				myGraph->add_tweights(currNodeId,0,(int)ceil(INT32_CONST * HARD_CONSTRAINT_CONST + 0.5));
		}
	}
	cout << "Maxflow..." << endl;
	int flow = myGraph -> maxflow();
	cout << "Done maxflow..." << flow << endl;

	int colorSep_E, hardConstraints_E;
	getColorSepE(colorSep_E, hardConstraints_E);
	cout << "Hard Constraints violation cost: " << hardConstraints_E << endl;
	cout << "Color Sep Term: " << colorSep_E << endl;
	cout << "Edge cost: " << flow - colorSep_E - hardConstraints_E << endl;

	// this is where we store the results
	segMask = 0;
	inputImg.copyTo(segShowImg);
	//inputImg.copyTo(showImg);

	// empty scribble masks are ready to record additional scribbles for additional hard constraints
	// to be used next time
	fgScribbleMask = 0;
	bgScribbleMask = 0;

	// copy the segmentation results on to the result images
	for (int i = 0; i<inputImg.rows * inputImg.cols; i++)
	{
		// if it is foreground - color blue
		if (myGraph->what_segment((GraphType::node_id)i ) == GraphType::SOURCE)
		{
			segMask.at<uchar>(i/inputImg.cols, i%inputImg.cols) = 255;
			(uchar)segShowImg.at<Vec3b>(i/inputImg.cols, i%inputImg.cols)[2] =  200;
		}
		// if it is background - color red
		else
		{
			segMask.at<uchar>(i/inputImg.cols, i%inputImg.cols) = 0;
			(uchar)segShowImg.at<Vec3b>(i/inputImg.cols, i%inputImg.cols)[0] =  200;
		}
	}

    // Write the segmentation mask to a file
    if (!outputFileName)
    {
        char buff[256];
        buff[0] = '\0';
        strncat(buff, imgFileName, (unsigned)(strlen(imgFileName) - 4));
        strcat(buff, "_segmented.png");
        imwrite(buff, segMask);
    }
    else
    {
        imwrite(outputFileName, segMask);
    }
	
    return 0;
}

void printHelp() {
    cout << "Usage: OneCut imageToSegment strokesFile [OPTIONS]" << endl;
    cout << endl;
    cout << "--fg-label x  : Label used for the foreground in the strokes file" << endl;
    cout << "--bg-label x  : Label used for the background in the strokes file" << endl;
    cout << "--output x  :  The name of the output file" << endl;
    cout << "--bins x  :  Number of bins per channel" << endl;
    cout << "--slope x  :  Color separator slope" << endl;
}

int loadStrokes(char * strokesFileName, Mat & fgScribbleMask, Mat & bgScribbleMask)
{
    Mat strokesData = imread(strokesFileName, CV_LOAD_IMAGE_GRAYSCALE);

    if (!strokesData.data) {
        cout << "Could not open or find the strokes image" << endl;
        return -1;
    }

    fgScribbleMask = strokesData == fgLabel;
    bgScribbleMask = strokesData == bgLabel;

	fgScribbleMask.copyTo(fgScribbleMaskAll);
	bgScribbleMask.copyTo(bgScribbleMaskAll);
	showImg.setTo(Scalar(0,0,255),fgScribbleMask);
	showImg.setTo(Scalar(255,0,0),bgScribbleMask);

	return 0;
}

int init(char * imgFileName)
{
	// Read the file
    inputImg = imread(imgFileName, CV_LOAD_IMAGE_COLOR);
	showImg = inputImg.clone();
	segShowImg = inputImg.clone();

	// Check for invalid input
    if(!inputImg.data)
    {
        cout <<  "Could not open or find the image: " << imgFileName << std::endl;
        return -1;
    }

	// this is the mask to keep the user scribbles
	fgScribbleMask.create(2,inputImg.size,CV_8UC1);
	fgScribbleMask = 0;
	bgScribbleMask.create(2,inputImg.size,CV_8UC1);
	bgScribbleMask = 0;
	
	fgScribbleMaskAll.create(2,inputImg.size,CV_8UC1);
	fgScribbleMaskAll = 0;
	bgScribbleMaskAll.create(2,inputImg.size,CV_8UC1);
	bgScribbleMaskAll = 0;
	
	segMask.create(2,inputImg.size,CV_8UC1);
	segMask = 0;
	showEdgesImg.create(2, inputImg.size, CV_32FC1);
	showEdgesImg = 0;
	binPerPixelImg.create(2, inputImg.size,CV_32F);

	// get bin index for each image pixel, store it in binPerPixelImg
	getBinPerPixel(binPerPixelImg, inputImg, numBinsPerChannel, numUsedBins);

	// compute the variance of image edges between neighbors
	getEdgeVariance(inputImg, showEdgesImg, varianceSquared);
	
	myGraph = new GraphType(/*estimated # of nodes*/ inputImg.rows * inputImg.cols + numUsedBins, 
		/*estimated # of edges=11 spatial neighbors and one link to auxiliary*/ 12 * inputImg.rows * inputImg.cols); 
	GraphType::node_id currNodeId = myGraph -> add_node((int)inputImg.cols * inputImg.rows + numUsedBins);

	for(int i=0; i<inputImg.rows; i++)
	{
		for(int j=0; j<inputImg.cols; j++) 
		{
			// this is the node id for the current pixel
			GraphType::node_id currNodeId = i * inputImg.cols + j;

			// add hard constraints based on scribbles
			if (fgScribbleMask.at<uchar>(i,j) == 255)
				myGraph->add_tweights(currNodeId,(int)ceil(INT32_CONST * HARD_CONSTRAINT_CONST + 0.5),0);
			else if (bgScribbleMask.at<uchar>(i,j) == 255)
				myGraph->add_tweights(currNodeId,0,(int)ceil(INT32_CONST * HARD_CONSTRAINT_CONST + 0.5));
				
			// You can now access the pixel value with cv::Vec3b
			float b = (float)inputImg.at<Vec3b>(i,j)[0];
			float g = (float)inputImg.at<Vec3b>(i,j)[1];
			float r = (float)inputImg.at<Vec3b>(i,j)[2];

			// go over the neighbors
			for (int si = -NEIGHBORHOOD; si <= NEIGHBORHOOD; si++)
			{
				int ni = i+si;
				// outside the border - skip
				if ( ni < 0 || ni >= inputImg.rows)
					continue;

				for (int sj = 0; sj <= NEIGHBORHOOD; sj++)
				{
					int nj = j+sj;
					// outside the border - skip
					if ( nj < 0 || nj >= inputImg.cols)
						continue;

					// same pixel - skip
					// down pointed edge, this edge will be counted as an up edge for the other pixel
					if (si >= 0 && sj == 0)
						continue;

					// diagonal exceed the radius - skip
					if ((si*si + sj*sj) > NEIGHBORHOOD*NEIGHBORHOOD) 
						continue;

					
					// this is the node id for the neighbor
					GraphType::node_id nNodeId = (i+si) * inputImg.cols + (j + sj);
					
					float nb = (float)inputImg.at<Vec3b>(i+si,j+sj)[0];
					float ng = (float)inputImg.at<Vec3b>(i+si,j+sj)[1];
					float nr = (float)inputImg.at<Vec3b>(i+si,j+sj)[2];

					//   ||I_p - I_q||^2  /   2 * sigma^2
					float currEdgeStrength = exp(-((b-nb)*(b-nb) + (g-ng)*(g-ng) + (r-nr)*(r-nr))/(2*varianceSquared));
					//float currEdgeStrength = 0;
					float currDist = sqrt((float)si*(float)si + (float)sj*(float)sj);

					// this is the edge between the current two pixels (i,j) and (i+si, j+sj)
					currEdgeStrength = ((float)EDGE_STRENGTH_WEIGHT * currEdgeStrength + (float)(1-EDGE_STRENGTH_WEIGHT)) /currDist;
					int edgeCapacity = /* capacities */ (int) ceil(INT32_CONST*currEdgeStrength + 0.5);
					//edgeCapacity = 0;
					myGraph -> add_edge(currNodeId, nNodeId,   edgeCapacity , edgeCapacity);
					
				}
			}
			// add the adge to the auxiliary node
			int currBin =  (int)binPerPixelImg.at<float>(i,j);

			myGraph -> add_edge(currNodeId, (GraphType::node_id)(currBin + inputImg.rows * inputImg.cols),
				/* capacities */ (int) ceil(INT32_CONST*bha_slope+ 0.5), (int)ceil(INT32_CONST*bha_slope + 0.5));
		}
	}
	
	return 0;
}

// get bin index for each image pixel, store it in binPerPixelImg
void getBinPerPixel(Mat & binPerPixelImg, Mat & inputImg, int numBinsPerChannel, int & numUsedBins)
{
	// this vector is used to throw away bins that were not used
	vector<int> occupiedBinNewIdx((int)pow((double)numBinsPerChannel,(double)3),-1);
	
	// go over the image
	int newBinIdx = 0;
	for(int i=0; i<inputImg.rows; i++)
		for(int j=0; j<inputImg.cols; j++) 
		{
			// You can now access the pixel value with cv::Vec3b
			float b = (float)inputImg.at<Vec3b>(i,j)[0];
			float g = (float)inputImg.at<Vec3b>(i,j)[1];
			float r = (float)inputImg.at<Vec3b>(i,j)[2];

			// this is the bin assuming all bins are present
			int bin = (int)(floor(b/256.0 *(float)numBinsPerChannel) + (float)numBinsPerChannel * floor(g/256.0*(float)numBinsPerChannel) 
				+ (float)numBinsPerChannel * (float)numBinsPerChannel * floor(r/256.0*(float)numBinsPerChannel)); 

			
			// if we haven't seen this bin yet
			if (occupiedBinNewIdx[bin]==-1)
			{
				// mark it seen and assign it a new index
				occupiedBinNewIdx[bin] = newBinIdx;
				newBinIdx ++;
			}
			// if we saw this bin already, it has the new index
			binPerPixelImg.at<float>(i,j) = (float)occupiedBinNewIdx[bin];
		}

		double maxBin;
		minMaxLoc(binPerPixelImg, NULL,&maxBin);
		numUsedBins = (int) maxBin + 1;

		occupiedBinNewIdx.clear();
		cout << "Num occupied bins:" << numUsedBins << endl;
}

// compute the variance of image edges between neighbors
void getEdgeVariance(Mat & inputImg, Mat & showEdgesImg, float & varianceSquared)
{
	varianceSquared = 0;
	int counter = 0;
	for(int i=0; i<inputImg.rows; i++)
	{
		for(int j=0; j<inputImg.cols; j++) 
		{
			// You can now access the pixel value with cv::Vec3b
			float b = (float)inputImg.at<Vec3b>(i,j)[0];
			float g = (float)inputImg.at<Vec3b>(i,j)[1];
			float r = (float)inputImg.at<Vec3b>(i,j)[2];
			for (int si = -NEIGHBORHOOD; si <= NEIGHBORHOOD && si + i < inputImg.rows && si + i >= 0 ; si++)
			{
				for (int sj = 0; sj <= NEIGHBORHOOD && sj + j < inputImg.cols ; sj++)

				{
					if ((si == 0 && sj == 0) ||
						(si == 1 && sj == 0) || 
						(si == NEIGHBORHOOD && sj == 0))
						continue;

					float nb = (float)inputImg.at<Vec3b>(i+si,j+sj)[0];
					float ng = (float)inputImg.at<Vec3b>(i+si,j+sj)[1];
					float nr = (float)inputImg.at<Vec3b>(i+si,j+sj)[2];

					varianceSquared+= (b-nb)*(b-nb) + (g-ng)*(g-ng) + (r-nr)*(r-nr); 
					counter ++;
					
				}
				
			}
		}
	}
	varianceSquared /= counter;

	// just for visualization
	for(int i=0; i<inputImg.rows; i++)
	{
		for(int j=0; j<inputImg.cols; j++) 
		{
			float edgeStrength = 0;
			// You can now access the pixel value with cv::Vec3b
			float b = (float)inputImg.at<Vec3b>(i,j)[0];
			float g = (float)inputImg.at<Vec3b>(i,j)[1];
			float r = (float)inputImg.at<Vec3b>(i,j)[2];
			for (int si = -NEIGHBORHOOD; si <= NEIGHBORHOOD && si + i < inputImg.rows && si + i >= 0; si++)
			{
				for (int sj = 0; sj <= NEIGHBORHOOD && sj + j < inputImg.cols   ; sj++)
				{
					if ((si == 0 && sj == 0) ||
						(si == 1 && sj == 0) ||
						(si == NEIGHBORHOOD && sj == 0))
						continue;

					float nb = (float)inputImg.at<Vec3b>(i+si,j+sj)[0];
					float ng = (float)inputImg.at<Vec3b>(i+si,j+sj)[1];
					float nr = (float)inputImg.at<Vec3b>(i+si,j+sj)[2];

					//   ||I_p - I_q||^2  /   2 * sigma^2
					float currEdgeStrength = exp(-((b-nb)*(b-nb) + (g-ng)*(g-ng) + (r-nr)*(r-nr))/(2*varianceSquared));
					float currDist = sqrt((float)si*(float)si + (float)sj * (float)sj);

					
					// this is the edge between the current two pixels (i,j) and (i+si, j+sj)
					edgeStrength = edgeStrength + ((float)0.95 * currEdgeStrength + (float)0.05) /currDist;
					
				}
			}
			// this is the avg edge strength for pixel (i,j) with its neighbors
			showEdgesImg.at<float>(i,j) = edgeStrength;

		}
	}
	
	double maxEdge;
	Point maxPoint;
	minMaxLoc(showEdgesImg,NULL,&maxEdge, NULL, &maxPoint);
	//cout << showEdgesImg.at<float>(maxPoint) << endl;

}

void getColorSepE(int & colorSep_E, int & hardConstraints_E)
{
	colorSep_E = 0;
	hardConstraints_E = 0;

	// copy the segmentation results on to the result images
	for(int i=0; i<inputImg.rows; i++)
	{
		for(int j=0; j<inputImg.cols; j++) 
		{
			// this is the node id for the current pixel
			GraphType::node_id currNodeId = i * inputImg.cols + j;

			// auxiliary node 1
			int currBin =  (int)binPerPixelImg.at<float>(i,j);
			int auxNodeId = currBin + inputImg.rows * inputImg.cols;


			// if it is foreground 
			if (myGraph->what_segment((GraphType::node_id)currNodeId ) == GraphType::SOURCE)
			{
				// but has bg hard constraints
				if (bgScribbleMaskAll.at<uchar>(i,j) == 255)
				{
					hardConstraints_E+=(int)ceil(INT32_CONST * HARD_CONSTRAINT_CONST + 0.5);
				}

				if (myGraph->what_segment((GraphType::node_id)auxNodeId) == GraphType::SINK)
					colorSep_E += (int) ceil(INT32_CONST*bha_slope+ 0.5);
			}
			// if it is background -
			else
			{
				// but has fg hard constraints
				if (fgScribbleMaskAll.at<uchar>(i,j) == 255)
				{
					hardConstraints_E+=(int)ceil(INT32_CONST * HARD_CONSTRAINT_CONST + 0.5);
				}
				if (myGraph->what_segment((GraphType::node_id)auxNodeId) == GraphType::SOURCE)
					colorSep_E += (int) ceil(INT32_CONST*bha_slope+ 0.5);
			}
		}
	}
}
