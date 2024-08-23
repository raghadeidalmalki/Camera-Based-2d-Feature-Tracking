#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ...
      
      // Use FLANN-based matcher
       matcher = cv::FlannBasedMatcher::create();
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // ...
       std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descSource, descRef, knnMatches, 2); // Get the 2 nearest neighbors for each descriptor

        // Apply ratio test to filter matches
        const float ratioThreshold = 0.8f; // Distance ratio threshold
        for (size_t i = 0; i < knnMatches.size(); i++)
        {
            if (knnMatches[i][0].distance < ratioThreshold * knnMatches[i][1].distance)
            {
                matches.push_back(knnMatches[i][0]); // Accept the best match if it is significantly better than the second
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
//void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
//{
    // select appropriate descriptor
  //  cv::Ptr<cv::DescriptorExtractor> extractor;
    //if (descriptorType.compare("BRISK") == 0)
    //{

      //  int threshold = 30;        // FAST/AGAST detection threshold score.
        //int octaves = 3;           // detection octaves (use 0 to do single scale)
        //float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        //extractor = cv::BRISK::create(threshold, octaves, patternScale);
    //}
    //else
    //{

        //...
    //}

    // perform feature description
    //double t = (double)cv::getTickCount();
    //extractor->compute(img, keypoints, descriptors);
    //t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
//}

#include <opencv2/xfeatures2d.hpp>

void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType)
{
    // Create descriptor extractor
    cv::Ptr<cv::DescriptorExtractor> extractor;

    if (descriptorType.compare("BRISK") == 0)
    {
        extractor = cv::BRISK::create();
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        std::cerr << "Invalid descriptor type: " << descriptorType << std::endl;
        return;
    }

    // Perform feature description
    extractor->compute(img, keypoints, descriptors);

    // Output number of descriptors and descriptor size
    std::cout << "Descriptor type: " << descriptorType << ", Keypoints: " << keypoints.size() << ", Descriptors size: " << descriptors.rows << "x" << descriptors.cols << std::endl;
}


void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Parameters for Harris corner detection
    int blockSize = 2;    // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3; // aperture parameter for Sobel operator
    double k = 0.04;      // Harris detector free parameter
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);

    // Detecting corners
    cv::cornerHarris(img, dst, blockSize, apertureSize, k);

    // Normalizing
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Extracting keypoints
    for (size_t i = 0; i < dst_norm.rows; i++)
    {
        for (size_t j = 0; j < dst_norm.cols; j++)
        {
            int response = (int)dst_norm.at<float>(i, j);
            if (response > 100) // threshold for detecting keypoints
            {
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(j, i);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;
                keypoints.push_back(newKeyPoint);
            }
        }
    }

    if (bVis)
    {
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


void detKeypointsFAST(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
    detector->detect(img, keypoints);

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "FAST Keypoint Detector";
        cv::namedWindow(windowName, 5);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsBRISK(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::Ptr<cv::BRISK> detector = cv::BRISK::create();
    detector->detect(img, keypoints);

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "BRISK Keypoint Detector";
        cv::namedWindow(windowName, 5);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


void detKeypointsORB(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->detect(img, keypoints);

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "ORB Keypoint Detector";
        cv::namedWindow(windowName, 5);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsAKAZE(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
    detector->detect(img, keypoints);

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "AKAZE Keypoint Detector";
        cv::namedWindow(windowName, 5);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsSIFT(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
    detector->detect(img, keypoints);

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "SIFT Keypoint Detector";
        cv::namedWindow(windowName, 5);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}