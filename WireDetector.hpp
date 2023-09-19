#include <iostream>
#include <tuple>
#include <utility>
#include <vector>
#include <thread>
#include <mutex>
#include <future>
#include <condition_variable>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "torch/torch.h"
#include "torch/script.h"


enum class DetectorStates
{
    ON = true,
    OFF = false
};

torch::jit::Module LoadModel(const std::string &weightsPath)
{
    try
    {
        torch::jit::Module module = torch::jit::load(weightsPath);
        module.to(torch::kCUDA);
        module.eval();
        return module;
    }

    catch(const std::exception& exception)
    {
        throw;
    }
}

class WireDetector
{
private:
    DetectorStates m_State = DetectorStates::OFF;
    std::string m_WeightsPath;
    std::string m_Name{"WireDetector"};

    std::atomic<double> m_AngleOffset{0};
    std::atomic<double> m_XOffset{0};
    std::atomic<double> m_YOffset{0};
    std::atomic<double> m_WireWidth{0};

    std::future<void>m_ActivityFuture;

    std::atomic<bool> m_Paused{true};

    std::thread m_ReaderThread;

    cv::VideoWriter m_VideoWriter{};

    cv::VideoCapture m_VideoCapture;

    std::string m_VideoPath;

    cv::String m_NamedWindow{"Wire Detector Video Player"};

    std::atomic<double> m_CurrentFrameIndex{0};

    std::mutex m_ActivityMutex;

    std::condition_variable m_ActivityConditionVariable;

    torch::jit::script::Module m_Model;

    bool m_Fullscreen{false};


    static std::tuple<int, int> GetPointCenterOffset(int frameWidth, int frameHeight, const cv::Point& point)
    {
        int center_x = frameWidth / 2;
        int center_y = frameHeight / 2;

        int offset_x = (center_x - point.x) / center_x * 100;
        int offset_y = (center_y - point.y) / center_y * 100;
        return std::make_tuple(offset_x, offset_y);
    }
    std::tuple<std::vector<std::vector<cv::Point>>, torch::Tensor>  CalculateOffsets(const cv::Mat &frame, int frameWidth, int frameHeight)
    {
        std::tuple<std::vector<std::vector<cv::Point>>, at::Tensor> returned_tuple = DetectWireContours(frame);
        std::vector<std::vector<cv::Point>> contours = std::get<0>(returned_tuple);
        torch::Tensor segmentation = std::get<1>(returned_tuple);

        std::vector<std::vector<cv::Point>> boxes;// = FindTheLongestBoxes(GetBoxesFromContours(contours));

        //boxes.empty() ? ClearOffsetValues(), m_WireCap = false :  m_WireCap = true;

        for(const auto& box : boxes)
        {
            cv::Point box_vertices = box[0];

        }

        return std::make_tuple(boxes, segmentation);
    }
    std::vector<std::vector<cv::Point>> FindTheLongestBoxes(const std::vector<std::vector<cv::Point>>&boxes)
    {

    }
    static std::vector<std::vector<cv::Point>> GetBoxesFromContours(const std::vector<std::vector<cv::Point>> &contours, float relativeThreshold = 0.4f, int noisyRectArea = 8000)
    {
        std::vector<std::vector<cv::Point>> boxes;

        for (const auto &contour : contours)
        {
            cv::RotatedRect rect = cv::minAreaRect(contour);
            std::vector<cv::Point> box;
//            rect.points(box.data());
//            boxes.push_back({box, rect.center, cv::Point2f(rect.size.width, rect.size.height), rect.angle});
        }
        return boxes;
    }

    std::tuple<std::vector<std::vector<cv::Point>>, torch::Tensor> DetectWireContours(const cv::Mat &frame, int minContourArea = 50)
    {
        // std::vector<torch::jit::IValue> input;
        // Disable gradient calculation for better performance
        torch::NoGradGuard no_grad;

        // Convert frame to torch tensor
        torch::Tensor tensor_frame = torch::from_blob(frame.data, {1, frame.rows, frame.cols, 3}, torch::kByte);
        // input.emplace_back(tensor_frame);
        // Running frame tensor through the model
        //torch::Tensor output = torch::sigmoid(m_Model.forward(input).toTensor()).detach().cpu();
        torch::Tensor output = m_Model.forward({tensor_frame}).toTensor();

        // Display segmentation
        torch::Tensor segmentation = (output > 0.5).to(torch::kByte)[0];

        cv::Mat seg_video(segmentation.size(0), segmentation.size(1), CV_8UC1, segmentation.data_ptr());

        try
        {
            cv::imshow("segment", seg_video);
        }

        catch (const cv::Exception &exception)
        {
            std::cout << exception.msg << std::endl;
            std::cout << seg_video << std::endl;
            std::cout << seg_video.size() << std::endl;
            throw;
        }

        // Find contours in the segmented video
        std::vector<std::vector<cv::Point>> contours;

        cv::findContours(seg_video, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Filter contours based on contour area and longest side
        std::vector<std::vector<cv::Point>> filtered_contours;

        std::vector<double> longest_sides;

        for (const auto &cnt : contours)
        {m_InputThread;
            cv::RotatedRect rect = cv::minAreaRect(cnt);

            double contour_area = cv::contourArea(cnt);

            if (contour_area >= minContourArea)
            {
                filtered_contours.push_back(cnt);
                longest_sides.push_back(std::max(rect.size.width, rect.size.height));
            }
        }

        if (!longest_sides.empty())
        {
            // Getting value of max element. Using '*' because std::max_element returns index but with '*' we get value of index
            double max_longest_side = *std::max_element(longest_sides.begin(), longest_sides.end());

            // Remove contours that are not equal to the longest side
            filtered_contours.erase(std::remove_if(filtered_contours.begin(), filtered_contours.end(), [&](const std::vector<cv::Point> &cnt)
                                    {
                                        cv::RotatedRect rect = cv::minAreaRect(cnt);
                                        return std::max(rect.size.width, rect.size.height) != max_longest_side; }),
                                    filtered_contours.end());
        }

        cv::Mat filtered = cv::Mat::ones(224, 320, CV_8U);

        cv::drawContours(filtered, filtered_contours, -1, cv::Scalar(255, 255, 255), 2);

        cv::imshow("filtered_counters", filtered);

        return std::make_tuple(filtered_contours, segmentation);
    }

    void Initialization()
    {
        m_VideoCapture.open(m_VideoPath);

        if (!m_VideoCapture.isOpened())
            throw std::runtime_error("Could not open a video file");

        cv::namedWindow(m_NamedWindow, cv::WINDOW_NORMAL);
        m_VideoCapture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        m_VideoCapture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        m_Paused = false;
    }

    void ProcessCamStream(const cv::Mat &frame)
    {
        cv::imshow(m_NamedWindow, frame);
    }

    void ReadCamStream()
    {
        cv::Mat frame;

        while (m_VideoCapture.isOpened())
        {
            std::unique_lock<std::mutex>unique_lock(m_ActivityMutex);
            m_ActivityConditionVariable.notify_all();
            unique_lock.unlock();

            if (m_Paused) continue;

            m_VideoCapture >> frame;

            if (frame.empty()) continue;

            ProcessCamStream(frame);

            int key = cv::waitKey(10);

            if (key == -1)
                continue;

            else if (key == 27)
                break;

            else if (key == 32)
                m_Paused = !m_Paused;

            else if(key == 97)
            {
                m_VideoCapture.set(cv::CAP_PROP_POS_FRAMES, m_VideoCapture.get(cv::CAP_PROP_POS_FRAMES) - 2);
            }

            else if(key == 100)
            {
                m_VideoCapture.set(cv::CAP_PROP_POS_FRAMES, m_VideoCapture.get(cv::CAP_PROP_POS_FRAMES) + 2);
            }

            else if(key == 48)
            {
                m_VideoCapture.set(cv::CAP_PROP_POS_FRAMES, 0);
            }

            else if(key == 'f')
            {
                m_Fullscreen = !m_Fullscreen;

                if (m_Fullscreen)
                    cv::setWindowProperty(m_NamedWindow, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
                else
                    cv::setWindowProperty(m_NamedWindow, cv::WND_PROP_FULLSCREEN, cv::WINDOW_NORMAL);
            }
        }

        //FIX IT!!!!!
        cv::destroyAllWindows();
        m_VideoCapture.release();
        Stop();
        exit(EXIT_SUCCESS);
    }

    void ClearOffsetValues()
    {
        m_AngleOffset = 0;
        m_XOffset = 0;
        m_YOffset = 0;
        m_WireWidth = 0;
    }

public:
    explicit WireDetector(std::string  weightsPath, std::string videoPath) noexcept: m_WeightsPath(std::move(weightsPath)), m_VideoPath{std::move(videoPath)}
    {
        m_Model = LoadModel(m_WeightsPath);
        Initialization();

        std::cout << "Model loaded" << std::endl;

        std::cout << "Initialization success" << std::endl;
    }

    void Start()
    {
        if(m_State == DetectorStates::ON)
            return;

        m_State = DetectorStates::ON;
        m_ReaderThread = std::thread([this]{ ReadCamStream();});
        m_ReaderThread.detach();
    }

    void Stop()
    {
        if(m_State == DetectorStates::OFF)
            return;
        ClearOffsetValues();
        m_State = DetectorStates::OFF;
        m_Paused = true;
    }

    void Log()
    {
        std::cout << "\nDetector values:" << std::endl;
        std::cout << "┌─────────────┐" << std::endl;
        std::cout << "│ Angle offset│ " << std::setw(10) << m_AngleOffset << " │" << std::endl;
        std::cout << "├─────────────┤" << std::endl;
        std::cout << "│ X offset    │ " << std::setw(10) << m_XOffset << " │" << std::endl;
        std::cout << "├─────────────┤" << std::endl;
        std::cout << "│ Y offset    │ " << std::setw(10) << m_YOffset << " │" << std::endl;
        std::cout << "├─────────────┤" << std::endl;
        std::cout << "│ Wire width  │ " << std::setw(10) << m_WireWidth << " │" << std::endl;
        std::cout << "└─────────────┘" << std::endl;
    }

    void IsActive(const std::chrono::seconds& howMuchToWait = std::chrono::seconds(1))
    {
        std::unique_lock<std::mutex> unique_lock(m_ActivityMutex);

        if(m_ActivityConditionVariable.wait_for(unique_lock, howMuchToWait) == std::cv_status::timeout)
            throw std::runtime_error("Wire detector is not responding");
    }
};

int main()
{
    const std::string weights_path{" "};
    const std::string video_path{"../videos/wire_on_blue_sky_picamera.mp4"};
    WireDetector wire_detector(weights_path, video_path);
    wire_detector.Start();

    while(true)
    {
        try
        {
            wire_detector.IsActive(std::chrono::seconds(5));
            wire_detector.Log();
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }

        catch(std::exception& exception)
        {
            std::cerr << exception.what() << std::endl;
            wire_detector.Stop();
            exit(EXIT_FAILURE);
        }
    }
}
