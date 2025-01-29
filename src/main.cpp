#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Open the default camera (camera 0)
    cv::VideoCapture cap(0);

    // Check if the camera opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the webcam." << std::endl;
        return -1;
    }

    std::cout << "Press 'q' to quit." << std::endl;

    // Create a window to display the video feed
    cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    while (true) {
        // Capture a frame from the webcam
        cap >> frame;

        // Check if the frame is empty
        if (frame.empty()) {
            std::cerr << "Error: Could not read a frame from the webcam." << std::endl;
            break;
        }

        // Display the frame in the window
        cv::imshow("Webcam", frame);

        // Break the loop if 'q' is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release the webcam and close all OpenCV windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}

