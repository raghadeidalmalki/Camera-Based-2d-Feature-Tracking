# Objective and Overview
In this "Feature tracking" project, we will implement a few detectors, descriptors, and matching algorithms. The project consists of four parts:
1.	The Data Buffer: we will start with loading the images, setting up the data structure, and put everything into the data buffer.
2.	Keypoint Detection: Integrate several keypoint detectors, such as HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT, and compare them to each other based on the number of key points and speed.
3.	Descriptor Extraction & Matching: Extract the descriptors and match them using the brute-force and FLANN approach.
4.	Performance Evaluation: Compare and evaluate which combination of algorithms perform the best concerning performance measurement parameters.

## MP.1 - Data Buffer Optimisation
In task 1, our objective was to optimize the image loading procedure. Initially, images were inefficiently stored in a vector using a for-loop, causing the data structure to expand with each new image. This approach becomes problematic when dealing with large sequences of images and Lidar point clouds overnight, as it rapidly consumes memory and slows down the entire program.
To mitigate these issues, our goal is to limit the number of images held in memory. When a new image arrives, the oldest image is removed from one end of the vector, and the new image is added to the opposite end. This method ensures that memory usage remains manageable and prevents performance degradation, as illustrated in the following diagram.

![image](https://github.com/user-attachments/assets/2e5c38e1-12df-423b-9cc3-153e0e255c44)

Original Code:
```ruby
for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        
        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);

        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;
```
My Implementation: 
```ruby
for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        
// Check if the data buffer is full
if (dataBuffer.size() >= dataBufferSize) {
    // Remove the oldest image from the buffer (the first one in the vector)
    dataBuffer.erase(dataBuffer.begin());
}

// Push the new image into the data frame buffer
DataFrame frame;
frame.cameraImg = imgGray;
dataBuffer.push_back(frame);

        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;
```
## MP.2 - Keypoint Detection
In task 2 the goal was to implementing keypoint detection. The original code already includes an implementation of the Shi-Tomasi detector. We added a variety of alternative detectors, including HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT. The following figure displays the keypoints identified using the SIFT method.

My Implementation: 
In the `main` function:
```ruby
/* DETECT IMAGE KEYPOINTS */

// extract 2D keypoints from current image
vector<cv::KeyPoint> keypoints; // create empty feature list for current image
string detectorType = "SHITOMASI"; // change this to test different detectors: "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"

if (detectorType.compare("SHITOMASI") == 0)
{
    detKeypointsShiTomasi(keypoints, imgGray, false);
}
else if (detectorType.compare("HARRIS") == 0)
{
    detKeypointsHarris(keypoints, imgGray, false);
}
else if (detectorType.compare("FAST") == 0)
{
    detKeypointsFAST(keypoints, imgGray, false);
}
else if (detectorType.compare("BRISK") == 0)
{
    detKeypointsBRISK(keypoints, imgGray, false);
}
else if (detectorType.compare("ORB") == 0)
{
    detKeypointsORB(keypoints, imgGray, false);
}
else if (detectorType.compare("AKAZE") == 0)
{
    detKeypointsAKAZE(keypoints, imgGray, false);
}
else if (detectorType.compare("SIFT") == 0)
{
    detKeypointsSIFT(keypoints, imgGray, false);
}

cout << "#2 : DETECT KEYPOINTS done" << endl;
```
In the `match2D_Student.cpp` file:
```ruby
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detect Harris keypoints in image
    int blockSize = 2;     // for every pixel, a blockSize x blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter

    cv::Mat dst, dstNorm, dstNormScaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dstNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dstNorm, dstNormScaled);

    for (size_t j = 0; j < dstNorm.rows; j++)
    {
        for (size_t i = 0; i < dstNorm.cols; i++)
        {
            int response = (int)dstNorm.at<float>(j, i);
            if (response > minResponse)
            {
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;
                keypoints.push_back(newKeyPoint);
            }
        }
    }

    if (bVis)
    {
        cv::Mat visImage = dstNormScaled.clone();
        cv::drawKeypoints(dstNormScaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector";
        cv::namedWindow(windowName, 5);
        cv::imshow(windowName, visImage);
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

```
Updating the header file:
```ruby
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis = false);
void detKeypointsFAST(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis = false);
void detKeypointsBRISK(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis = false);
void detKeypointsORB(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis = false);
void detKeypointsAKAZE(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis = false);
void detKeypointsSIFT(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis = false);
```
## MP.3 - Keypoint Removal
The third task involves filtering out all keypoints that fall outside the bounding box surrounding the preceding vehicle. You should use the following box parameters: cx = 535, cy = 180, w = 180, and h = 150. These coordinates are based on the Rect datatype in OpenCV. For more details on the origin, refer to the [documentation](https://docs.opencv.org/4.4.0/d2/d44/classcv_1_1Rect__.html) .
This step is crucial because, in a later part of the project, we will be assessing various detectors and descriptors based on specific performance metrics. Since our focus is on a collision detection system, keypoints on the preceding vehicle are particularly important. Therefore, to facilitate a more precise evaluation, we aim to exclude feature points that are not located on the preceding vehicle.

My Implementation: 
```ruby
// only keep keypoints on the preceding vehicle
bool bFocusOnVehicle = true;
cv::Rect vehicleRect(535, 180, 180, 150);  // Define the bounding box for the vehicle
if (bFocusOnVehicle)
{
    vector<cv::KeyPoint> vehicleKeypoints;
    for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
    {
        // Check if the keypoint is within the vehicle's bounding box
        if (vehicleRect.contains(it->pt))
        {
            vehicleKeypoints.push_back(*it);
        }
    }
    keypoints = vehicleKeypoints;  // Replace original keypoints with the filtered ones
}
```
## MP.4 - Keypoint Descriptors
The fourth task involves implementing various keypoint descriptors and making them selectable through the string 'descriptorType'. The methods I integrated include BRIEF, ORB, FREAK, AKAZE, and SIFT.

My Implementation: 
```ruby
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
```
## MP.5 - Descriptor Matching
The fifth task concentrates on the matching process. The initial implementation uses Brute Force matching combined with Nearest-Neighbor selection. I added FLANN as an alternative to Brute Force, along with the K-Nearest-Neighbor approach. 

My Implementation: 
```ruby
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // Configure matcher
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        if (descriptorType.compare("SIFT") == 0)
        {
            normType = cv::NORM_L2;
        }
        matcher = cv::BFMatcher::create(normType);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::FlannBasedMatcher::create();
    }
    else
    {
        std::cerr << "Invalid matcher type: " << matcherType << std::endl;
        return;
    }

    // Perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    {
        matcher->match(descSource, descRef, matches);
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descSource, descRef, knnMatches, 2);

        // Apply ratio test to filter matches
        const float ratioThresh = 0.8f;
        for (size_t i = 0; i < knnMatches.size(); i++)
        {
            if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance)
            {
                matches.push_back(knnMatches[i][0]);
            }
        }
    }
    else
    {
        std::cerr << "Invalid selector type: " << selectorType << std::endl;
        return;
    }

    std::cout << "Matcher type: " << matcherType << ", Selector type: " << selectorType << ", Matches: " << matches.size() << std::endl;
}
```
## MP.6 - Descriptor Distance Ratio
For the sixth task, we will implement the descriptor distance ratio test as a filtering method to eliminate poor keypoint matches. In the matching2D.cpp file, I added KNN match selection and applied descriptor distance ratio filtering with a threshold of 0.8.

My Implementation: 
```ruby
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>

void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // Configure matcher
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        if (descriptorType.compare("SIFT") == 0)
        {
            normType = cv::NORM_L2;
        }
        matcher = cv::BFMatcher::create(normType);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::FlannBasedMatcher::create();
    }
    else
    {
        std::cerr << "Invalid matcher type: " << matcherType << std::endl;
        return;
    }

    // Perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    {
        matcher->match(descSource, descRef, matches);
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descSource, descRef, knnMatches, 2);

        // Apply ratio test to filter matches
        const float ratioThresh = 0.8f;
        for (size_t i = 0; i < knnMatches.size(); i++)
        {
            if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance)
            {
                matches.push_back(knnMatches[i][0]);
            }
        }
    }
    else
    {
        std::cerr << "Invalid selector type: " << selectorType << std::endl;
        return;
    }

    std::cout << "Matcher type: " << matcherType << ", Selector type: " << selectorType << ", Matches: " << matches.size() << std::endl;
}
```

