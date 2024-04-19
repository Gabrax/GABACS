#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

int main(){


    std::string path = "C:/GBX/Photos/spike.jpg";
    cv::Mat img = cv::imread(path);
    cv::Mat imgGray, imgBlur,imgCanny;
    cv::cvtColor(img,imgGray, cv::COLOR_BGR2GRAY);

    if (img.empty()) {
        std::cerr << "Error: Could not open or read the image " << path << '\n';
        return 1;
    }
    
    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    cv::imshow("Image", img);

    cv::GaussianBlur(img,imgBlur, cv::Size(5,5), 10, 0);
    cv::namedWindow("Gauss", cv::WINDOW_NORMAL);
    cv::imshow("Gauss", imgBlur);

    cv::Canny(imgBlur,imgCanny,50,150,3);
    cv::namedWindow("Canny", cv::WINDOW_NORMAL);
    cv::imshow("Canny", imgCanny);
    
    cv::namedWindow("GrayImage", cv::WINDOW_NORMAL);
    cv::imshow("GrayImage", imgGray);
    

    std::string vid_path = "C:/GBX/filmyfranka.mp4";
    cv::VideoCapture video(vid_path);
    cv::Mat frame;
    cv::namedWindow("Video", cv::WINDOW_NORMAL);

    

    //cv::VideoCapture cam(0);
    //cv::Mat cam_frame;
    //cv::namedWindow("Webcam", cv::WINDOW_NORMAL);


    // Read video frames and display them
    while (true) {
        video.read(frame);
        //cam.read(cam_frame);
        
        if (frame.empty()) {
            std::cerr << "End of video" << std::endl;
            break;
        }

        
        cv::imshow("Video", frame);
        cv::resizeWindow("Video",640,480);

        //cv::imshow("Webcam", cam_frame);
        //cv::resizeWindow("Webcam",640,480);

        // Wait for a small amount of time to display the frame
        cv::waitKey(1);

        // Check if the user pressed any key to exit
        if (cv::waitKey(1) == 27) // 27 is the ASCII code for the escape key
            break;
    }

    
    cv::destroyAllWindows();


    
    return 0;
}
