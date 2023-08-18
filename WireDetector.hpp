#include <iostream>
#include <cmath>
#include <tuple>
#include <functional>
#include <vector>
#include <algorithm>
#include <chrono>

#include <opencv2/opencv.hpp>

enum class DetectorStates 
{
    OFF,
    ON
};

class WireDetector 
{
    private:
        bool m_IsWoking{false};
        std::string m_WeightsPath{"model.pth"};
        double m_AngleOffset{0};
        double m_XOffset{0};
        double m_YOffset{0};
        double m_WireWidth{0};

        bool m_CamStreamCap{0};

        std::string m_Name{"WireDetector"};
        cv::VideoWriter m_VideoWriter{};
    public:

        explicit WireDetector(){}

        void Start()
        {

        }


        void ReadCameStream(int &camState)
        {
            cv::VideoCapture cap{"./videos/3.MOV"};

            if(!cap.isOpened())
            {
                std::cout << "Error opening video stream" << std::endl;
                m_CamStreamCap = false;
                exit(EXIT_FAILURE);
            }

            cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

            int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
            int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
           
           camState = 1;

           frame_width -= 3;

           int current_frame_index{1};

           bool paused{false};

           while(cap.isOpened())
           {
                if(!paused)
                {
                    cv::Mat frame;
    
                    cap >> frame;
                
                    if (frame.empty())
                    {
                        m_CamStreamCap = false;

                        std::cout << "| WARN | Couldn't retrieve frame from camera";

                        continue;
                    }
                }
           }
            
            
        }
};

int main() {
    return 0;
}
