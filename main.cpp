#include <cstdio>
#include <iostream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "opencv2/opencv.hpp"
#include <string>
#include <fstream>

using namespace tflite;
using namespace std;
using namespace cv;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {

    // label array
    vector<string> label_arr;

    string line;
    ifstream label_file("/home/pi/tflite/model/labelmap.txt");
    while (getline(label_file, line)) {
        label_arr.push_back(line);
    }

    // opencv open camera
    VideoCapture cap(0);
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 256);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 256);

    const char* filename = "/home/pi/tflite/model/detect.tflite";
    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile(filename);
    TFLITE_MINIMAL_CHECK(model != nullptr);

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    interpreter->SetNumThreads(4);

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

    // input tensor id
    int input_id = interpreter->inputs()[0];
    cout << "input tensor id: " << input_id << endl;

    const vector<int> inputs = interpreter->inputs();
    const vector<int> outputs = interpreter->outputs();

    // get input dimension from input tensor
    TfLiteIntArray* dims = interpreter->tensor(input_id)->dims;
    int height = dims->data[1];
    int width = dims->data[2];
    int channels = dims->data[3];

    cout << "input width, height, channels: " << width << " " << height << " " << channels << endl;

    // start inference

    Mat edges;
    namedWindow("detect",1);
    for(;;)
    {

        Mat frame0;
        Mat frame;
        frame.create(300, 300, CV_8UC(2));
        cap.read(frame0); // get a new frame from camera
        resize(frame0, frame, frame.size(), 0, 0, INTER_NEAREST);
        cvtColor(frame, frame, CV_BGR2RGB);
        auto input_tensor = interpreter->typed_tensor<uchar>(input_id);

        // copy Mat pixels to input tensor
        unsigned char *input = (unsigned char*)(frame.data);
        for(int j = 0;j < frame.rows;j++){
            for(int i = 0;i < frame.cols;i++){
                unsigned char r = input[frame.rows * j + i ] ;
                unsigned char g = input[frame.rows * j + i + 1];
                unsigned char b = input[frame.rows * j + i + 2];
                input_tensor[frame.rows * j + i] = r;
                input_tensor[frame.rows * j + i + 1] = g;
                input_tensor[frame.rows * j + i + 2] = b;
            }
        }

        // Run inference
        TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

        // Read output buffers
        auto class_id = interpreter->typed_output_tensor<float>(1)[0];
        auto score = interpreter->typed_output_tensor<float>(2)[0];
        if (score > 0.5) {
            cout << "predict label: " << label_arr[class_id+1] << " scores: " << score << endl;
            auto bbox = interpreter->typed_output_tensor<float>(0);
            cout << "bbox: " << bbox[0] << " " << bbox[1] << " " << bbox[2] << " " << bbox[3] << endl;

            // draw rectangle
            auto top = int(bbox[0] * (float)frame.cols);
            auto left = int(bbox[1] * (float)frame.rows);
            auto bottom = int(bbox[2] * (float)frame.cols);
            auto right = int(bbox[3] * (float)frame.rows);

            top = top < 0 ? 0 : top;
            left = left < 0 ? 0 : left;
            bottom = bottom > frame.cols-1 ? frame.cols-1 : bottom;
            right = right > frame.rows-1 ? frame.rows-1 : right;

            cout << "bbox_refined: " << top << " " << left << " " << bottom << " " << right << endl;
            cout << endl;

            Point pt1(left, top);
            Point pt2(right, bottom);
            rectangle(frame, pt1, pt2, Scalar(0, 255, 0));

        }

        // show image
        cvtColor(frame, frame, CV_RGB2BGR);
        imshow("detect", frame);
        if(waitKey(1) >= 0) break;

    }

    return 0;
}