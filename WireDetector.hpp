#include <iostream>
#include <cmath>
#include <tuple>
#include <functional>
#include <utility>
#include <vector>
#include <thread>

#include "opencv2/opencv.hpp"
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

    bool m_CamStreamCap{false};
    std::atomic<bool> m_WireCap{false};

    std::thread m_ReaderThread;

    cv::VideoWriter m_VideoWriter{};

    torch::jit::script::Module m_Model;

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

        std::vector<std::vector<cv::Point>> boxes = FindTheLongestBoxes(GetBoxesFromContours(contours));

        boxes.empty() ? ClearOffsetValues(), m_WireCap = false :  m_WireCap = true;

        for(const auto& box : boxes)
        {
            cv::Point box_vertices = box[0];

        }

        return std::make_tuple(boxes, segmentation);
    }

    std::vector<std::vector<cv::Point>> FindTheLongestBoxes(const std::vector<std::vector<cv::Point>>&boxes)
    {

    }

    void ClearOffsetValues()
    {
        m_AngleOffset = 0;
        m_XOffset = 0;
        m_YOffset = 0;
        m_WireWidth = 0;
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
        std::vector<torch::jit::IValue> input;
        // Disable gradient calculation for better performance
        torch::NoGradGuard no_grad;

        // Convert frame to torch tensor
        torch::Tensor frame_tensor = torch::from_blob(frame.data, {1, frame.rows, frame.cols, frame.channels()}, torch::kByte);
        input.emplace_back(frame_tensor);
        // Running frame tensor through the model
        torch::Tensor output = torch::sigmoid(m_Model.forward(input).toTensor()).detach().cpu();

        // Display segmentation
        torch::Tensor segmentation = (output > 0.5).to(torch::kByte)[0];

        cv::Mat seg_video(segmentation.size(1), segmentation.size(2), CV_8UC1, segmentation.data_ptr());

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
        {
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

    void ReadCamStream()
    {
        cv::VideoCapture cap{"./videos/3.MOV"};

        if (!cap.isOpened())
        {
            std::cout << "Error opening video stream" << std::endl;
            m_CamStreamCap = false;
            exit(EXIT_FAILURE);
        }

        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

        double frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        double frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        m_CamStreamCap = true;

        frame_width -= 3;

        double current_frame_index{0};

        bool paused{false};

        cv::namedWindow("Wire Detector Video Player");

        while (cap.isOpened())
        {

            if (!paused)
            {
                cv::Mat frame;

                cap >> frame;

                if (frame.empty())
                {
                    m_CamStreamCap = 0;
                    continue;
                }

                cv::Mat image;
                cv::resize(frame, image, cv::Size(320, 224));

                CalculateOffsets(frame, image.cols, image.rows);

                cv::imshow("video", image);

                int key = cv::waitKey(1);

                if (key == -1)
                    continue;

                else if (key == 27)
                    throw;

                else if (key == 32)
                    paused = !paused;

                else if (key == 82 || key == 52)
                {
                    current_frame_index = std::max((double)0, current_frame_index - 1);
                    cap.set(cv::CAP_PROP_POS_FRAMES, current_frame_index);
                }

                else if (key == 83 || key == 54)
                {
                    current_frame_index = std::min(current_frame_index + 1, cap.get(cv::CAP_PROP_FRAME_COUNT) - 1);

                    cap.set(cv::CAP_PROP_POS_FRAMES, current_frame_index);
                }

                current_frame_index++;
            }
        }

        cap.release();
        cv::destroyAllWindows();
    }

public:
    explicit WireDetector(std::string  path) : m_WeightsPath(std::move(path))
    {
        try
        {
            m_Model = LoadModel(m_WeightsPath);
        }

        catch(const std::exception& exception)
        {
            std::cerr << "ERROR: Model did not load:\n Error message:" << exception.what() << std::endl;
            exit(EXIT_FAILURE);
        }

        std::cout << "Model loaded";
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
    }

    void Log()
    {
        std::cout << "Detector values: " << std::endl << std::endl;
        std::cout << "Wire captured: " <<  m_WireCap << std::endl;
        std::cout << "Angle offset: " << m_AngleOffset << std::endl;
        std::cout << "X offset: " << m_XOffset << std::endl;
        std::cout << "Y offset: " << m_XOffset << std::endl;
        std::cout << "Wire width: " << m_WireWidth << std::endl;
    }
};

int main()
{
    const std::string path{" "};
    WireDetector wire_detector(path);
    wire_detector.Start();

    while(true)
    {
        try
        {
            wire_detector.Log();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        catch(std::exception& exception)
        {
            wire_detector.Stop();
            exit(EXIT_FAILURE);
        }
    }
}
