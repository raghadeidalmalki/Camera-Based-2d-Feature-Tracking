# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

## Objective and Overview
The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, we will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:
1.	The Data Buffer: we will start with loading the images, setting up the data structure, and put everything into the data buffer.
2.	Keypoint Detection: Integrate several keypoint detectors, such as HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT, and compare them to each other based on the number of key points and speed.
3.	Descriptor Extraction & Matching: Extract the descriptors and match them using the brute-force and FLANN approach.
4.	Performance Evaluation: Compare and evaluate which combination of algorithms perform the best concerning performance measurement parameters.



## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.

## MP.1 - Data Buffer Optimisation
In task 1, our objective was to optimize the image loading procedure. Initially, images were inefficiently stored in a vector using a for-loop, causing the data structure to expand with each new image. This approach becomes problematic when dealing with large sequences of images and Lidar point clouds overnight, as it rapidly consumes memory and slows down the entire program.
To mitigate these issues, our goal is to limit the number of images held in memory. When a new image arrives, the oldest image is removed from one end of the vector, and the new image is added to the opposite end. This method ensures that memory usage remains manageable and prevents performance degradation, as illustrated in the following diagram.

![image](https://github.com/user-attachments/assets/bc6abae3-76fd-41a9-b223-47e6665ff686)

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

```ruby
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

      cout << "Harris detection with n=" << keypoints.size() << " keypoints" << endl;

  
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
```

```ruby
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    // assign detector
    cv::Ptr<cv::FeatureDetector> detector;
    if (detectorType.compare("FAST") == 0)
    {
        detector = cv::FastFeatureDetector::create(30, true, cv::FastFeatureDetector::TYPE_9_16);
    }
    else if (detectorType.compare("BRISK") == 0)
    {
        detector = cv::BRISK::create();
    }
    else if (detectorType.compare("ORB") == 0)
    {
        detector = cv::ORB::create();
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        detector = cv::AKAZE::create();
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        detector = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        std::cout << "Invalid keypoints detector.\n";
    }

    // detect
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  
      cout << detectorType << " with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    
  if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
```

## MP.3 - Keypoint Removal
The third task involves filtering out all keypoints that fall outside the bounding box surrounding the preceding vehicle. You should use the following box parameters: cx = 535, cy = 180, w = 180, and h = 150. These coordinates are based on the Rect datatype in OpenCV. For more details on the origin, refer to the [documentation](https://docs.opencv.org/4.4.0/d2/d44/classcv_1_1Rect__.html) .
This step is crucial because, in a later part of the project, we will be assessing various detectors and descriptors based on specific performance metrics. Since our focus is on a collision detection system, keypoints on the preceding vehicle are particularly important. Therefore, to facilitate a more precise evaluation, we aim to exclude feature points that are not located on the preceding vehicle.

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
The fourth task involves implementing various keypoint descriptors and making them selectable through the string 'descriptorType'. The methods I integrated include BRIEF, ORB, BRIEF, FREAK, AKAZE, and SIFT.


```ruby
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType)
{
    // Create descriptor extractor
    cv::Ptr<cv::DescriptorExtractor> extractor;

    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
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
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    
    if (descriptors.empty()) {
        std::cerr << "Descriptor extraction failed." << std::endl;
        return;
    }
    
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "Descriptor type: " << descriptorType << ", Keypoints: " << keypoints.size() 
              << ", Descriptors size: " << descriptors.rows << "x" << descriptors.cols 
              << ", Extraction time: " << 1000 * t << " ms" << std::endl;
}
```


## MP.5 - Descriptor Matching
The fifth task concentrates on the matching process. The initial implementation uses Brute Force matching combined with Nearest-Neighbor selection. I added FLANN as an alternative to Brute Force, along with the K-Nearest-Neighbor approach.

## MP.6 - Descriptor Distance Ratio
For the sixth task, we will implement the descriptor distance ratio test as a filtering method to eliminate poor keypoint matches. In the matching2D.cpp file, I added KNN match selection and applied descriptor distance ratio filtering with a threshold of 0.8.

```ruby
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
      // Use FLANN-based matcher
       if (descSource.type() != CV_32F || descRef.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

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
  // Output the total number of matches found
    std::cout << "Total matches: " << matches.size() << std::endl;

}
```

## MP.7 & MP.8 - Performance Evaluation

For task 7, we were tasked with counting the number of keypoints on the preceding vehicle across all 10 images. For task 9, we needed to count the number of matched keypoints for these images, utilizing every possible combination of detectors and descriptors. During the matching phase, we used the KNN and FLANN matcher as an alternative to the Brute Force method, with the descriptor distance ratio set to 0.8.


Number of keypoints and matched keypoints on the preceding vehicle across all 10 images using every combination of detectors and descriptors:

**SHITOMASI Detector**
<img width="1114" alt="image" src="https://github.com/user-attachments/assets/7c875a14-fc83-42b6-ac30-61f619076c19">

**HARRIS Detector**
<img width="1128" alt="image" src="https://github.com/user-attachments/assets/f2cb872d-c45e-4433-9906-cd09db569504">

**FAST Detector**
<img width="1127" alt="image" src="https://github.com/user-attachments/assets/c154b554-7305-44e3-b35c-3e6b96d628ad">

**BRISK Detector**
<img width="1144" alt="image" src="https://github.com/user-attachments/assets/d8bf8664-ebfb-436f-9d24-91a74f8761e0">

**ORB Detector**
<img width="1144" alt="image" src="https://github.com/user-attachments/assets/12e2a19f-dba6-485f-bde2-5408fe28f90a">

**AKAZE Detector**
<img width="1144" alt="image" src="https://github.com/user-attachments/assets/307b23d1-ae6e-453e-b4ff-f5a6ad5ebd57">

**SIFT Detector**
<img width="1149" alt="image" src="https://github.com/user-attachments/assets/b9608df8-1205-4d48-b4a8-5d8dd6d5e35b">

## MP.9 - Performance Evaluation

In task 9, the time taken for keypoint detection and extraction was recorded to identify the optimal combination of detector and descriptor based on their efficiency.

<img width="1118" alt="image" src="https://github.com/user-attachments/assets/e403cb85-38fc-43e6-ab63-98d0f6cd7750">


<img width="1076" alt="image" src="https://github.com/user-attachments/assets/1f81286a-fafe-4785-9787-a67c985b39a0">

The results show that the FAST detector significantly outperforms all other detectors regarding execution time for detection. In contrast, the FREAK, AKAZE, and SIFT descriptors generally yield slower detection results.

**Recommendation**

Considering the metrics above, the FAST detector offers exceptional speed in keypoint detection while also achieving a substantial number of detected keypoints with certain descriptors. I recommend the combinations of FAST-BRISK, FAST-ORB, and FAST-BRIEF as my top three choices.

