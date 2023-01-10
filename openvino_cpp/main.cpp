#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

using namespace std;
// define shape of image
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const int CHANNELS = 3;

struct CocoResult
{
    int imageId;
    string category;
    int categoryId;
    vector<cv::Rect> boxes;
    vector<vector<cv::Point>> segmentations;

    CocoResult(
        int imageId, string category, int categoryId,
        vector<cv::Rect> boxes, vector<vector<cv::Point>> segmentations)
        : imageId(imageId),
          category(category),
          categoryId(categoryId),
          boxes(boxes),
          segmentations(segmentations) {}

    CocoResult()
        : imageId(-1),
          category("background"),
          categoryId(0),
          boxes({}),
          segmentations({}) {}
};

/** debug utils **/
void printImg(cv::Mat img)
{
    cout << "Shape: " << img.rows << endl;
    cout << "cv::Mat img = " << endl
         << " " << img << endl
         << endl;
}

void saveFile(string filename, cv::Mat img)
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "Result" << img;
    fs.release();
}

ov::CompiledModel prepareModel(const std::string modeFileName, const std::string deviceName = "CPU")
{   
    // Step 1. Initialize OpenVINO Runtime core
    ov::Core core;
    // Step 2. Read a model
    std::shared_ptr<ov::Model> model = core.read_model(modeFileName);

    if (model->get_parameters().size() != 1)
    {
        throw std::logic_error("Segment model must have only one input");
    }
    // Step 3. Inizialize Preprocessing for the model
    ov::preprocess::PrePostProcessor ppp(model);

    // Specify input image format:
    ppp.input()
        .tensor()
        .set_element_type(ov::element::f32) // f32
        .set_layout("NHWC");

    //  Specify model's input layout
    ppp.input()
        .model()
        .set_layout("NHWC");

    ppp.output()
        .tensor()
        .set_element_type(ov::element::f32);

    // Embed above steps in the graph
    model = ppp.build();
    ov::CompiledModel compiledModel = core.compile_model(model, deviceName);
    return compiledModel;
}

/**
 * @brief Preprocessing image
 * 
 * @param img (cv::Mat) : BGR mode
 * @param width (int) : target width image
 * @param height (int) : target height image
 * @param use_gray_image (bool) 
 * @param scale (float) : 0.00392156862745098 = 1/255.
 * @return cv::Mat : GRAYSCALE mode if use_gray_image is true, otherwise RGB mode.
 */
cv::Mat preprocess(const cv::Mat &img, int width, int height, bool use_gray_image = false, float scale = 0.00392156862745098)
{

    cv::Mat colorImg;
    int data_type = use_gray_image ? CV_32FC1 : CV_32F;
    if (use_gray_image)
    {
        cv::cvtColor(img, colorImg, cv::COLOR_BGR2GRAY);
    }
    else
    {
        cv::cvtColor(img, colorImg, cv::COLOR_BGR2RGB);
    }

    cv::Mat resizedImg;
    cv::resize(colorImg, resizedImg, cv::Size(width, height));

    // // rescale
    cv::Mat processedImg;
    resizedImg.convertTo(processedImg, data_type, 1.0 / 255, 0);    

    return processedImg;
}

cv::Mat tensorToMat(const ov::Tensor &tensor)
{
    // NOTE: OpenVINO runtime sizes are reversed.
    ov::Shape tensorShape = tensor.get_shape();
    std::vector<int> size;
    std::transform(tensorShape.begin(), tensorShape.end(), std::back_inserter(size), [](size_t dim) -> int
                   { return int(dim); });
    ov::element::Type precision = tensor.get_element_type();
    CV_Assert(precision == ov::element::f32);
    for (auto s : size)
    {
        cout << s << "-";
    }
    return cv::Mat(size, CV_32F, (void *)tensor.data());
}

void getSizeImg(cv::Mat img)
{
    const size_t channels = img.channels();
    const size_t height = img.rows;
    const size_t width = img.cols;
    cout << "Size image (h,w,c) = " << height << "," << width << "," << channels << endl;
}

pair<cv::Scalar, cv::Mat> structural_similarity(
    cv::Mat i1,
    cv::Mat i2,
    int data_range = 1,
    int win_size = 11,
    bool gaussian_weights = true,
    bool multichannel = false,
    float sigma = 1.5,
    bool full = true)
{

    const float K1 = 0.01;
    const float K2 = 0.03;
    const float R = data_range; // data_range

    // const double C1 = 6.5025, C2 = 58.5225;  // Ci = (Ki*R)**2
    float C1, C2;
    C1 = pow(K1 * R, 2);
    C2 = pow(K2 * R, 2);
    // cout << "C1 = " << C1 << ", C2 = " << C2 << endl;
    /***************************** INITS **********************************/
    int d = CV_32FC1;
    cv::Mat I1, I2;
    i1.convertTo(I1, d); // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    cv::Mat I2_2 = I2.mul(I2);  // I2^2
    cv::Mat I1_2 = I1.mul(I1);  // I1^2
    cv::Mat I1_I2 = I1.mul(I2); // I1 * I2

    /*************************** END INITS **********************************/
    cv::Mat mu1, mu2; // PRELIMINARY COMPUTING
    cv::GaussianBlur(I1, mu1, cv::Size(win_size, win_size), sigma);
    cv::GaussianBlur(I2, mu2, cv::Size(win_size, win_size), sigma);
    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);
    cv::Mat sigma1_2, sigma2_2, sigma12;
    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(win_size, win_size), sigma);
    sigma1_2 -= mu1_2;
    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(win_size, win_size), sigma);
    sigma2_2 -= mu2_2;
    cv::GaussianBlur(I1_I2, sigma12, cv::Size(win_size, win_size), sigma);
    sigma12 -= mu1_mu2;

    cv::Mat A1, A2, B1, B2, D, S;
    A1 = 2 * mu1_mu2 + C1;
    A2 = 2 * sigma12 + C2;
    B1 = mu1_2 + mu2_2 + C1;
    B2 = sigma1_2 + sigma2_2 + C2;

    D = B1.mul(B2);
    cv::divide(A1.mul(A2), D, S); // S = (A1*A2) / D

    cv::Scalar ssim_score = cv::mean(S);
    pair<cv::Scalar, cv::Mat> output = make_pair(ssim_score, S);
    return output;
}

/**
 * @brief Remove border + object attach with border
 * 
 * @param img cv::Mat : binary image
 * @return cv::Mat : binary image
 */
cv::Mat clear_border(const cv::Mat &img)
{
    // padding img
    cv::Mat padImg;
    cv::copyMakeBorder(img, padImg, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(1));
    cv::Size padImgSize = padImg.size();
    cv::Mat mask = cv::Mat::zeros(padImgSize.height + 2, padImgSize.width + 2, CV_8UC1);

    // floodfill
    cv::Point seedPoint = cv::Point(0, 0);
    cv::Scalar newVal = cv::Scalar(0);
    cv::Rect rect;
    cv::Scalar loDiff = cv::Scalar(0);
    cv::Scalar upDiff = cv::Scalar(0);
    int flag = 8;

    int area = cv::floodFill(padImg, mask, seedPoint, newVal, &rect, loDiff, upDiff, flag);

    // remove padding
    cv::Rect cropBorderRect = cv::Rect(1, 1, padImgSize.width - 2, padImgSize.height - 2); // Rect(x,y,w,h)
    cv::Mat clearedBorderImg;
    padImg(cropBorderRect).copyTo(clearedBorderImg);

    return clearedBorderImg;
}

/**
 * @brief Use connectedComponentsWithStats algorithm to get labeled_image + areas
 * 
 * @param image_th cv::Mat : binary image
 * @return pair<cv::Mat, vector<int>> : labeled_image + areas
 */
pair<cv::Mat, vector<int>> label_images(const cv::Mat &image_th)
{
    // clear_border
    cv::Mat clearedBorderImg = clear_border(image_th);

    // get area + labeled image 
    cv::Mat labels, stats, centroids;
    int connectivity = 4;
    int ltype = CV_16U;
    cv::Mat image_th_u8;
    clearedBorderImg.convertTo(image_th_u8, CV_8UC1);
    int i, nccomps = cv::connectedComponentsWithStats(image_th_u8, labels, stats, centroids, connectivity, ltype);
    vector<int> areas;
    for (int i = 0; i < nccomps; i++)
    {
        // the first connected componnet is always the background, we ignore
        if (i == 0)
            continue;

        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        double cx = centroids.at<double>(i, 0);
        double cy = centroids.at<double>(i, 1);
        areas.push_back(area);
    }
    return make_pair(labels, areas);
}

int is_defective(vector<int> areas, float min_area)
{
    for (auto area : areas)
    {
        if (area >= min_area)
            return 1;
    }
    return 0;
}

CocoResult convertMaskToCocoFormat(
    const cv::Mat &mask,
    const int thresholdArea = 25,
    const int thresholdWidth = 10,
    const int thresholdHeight = 10)
{
    CocoResult cocoResult;

    std::vector<std::vector<cv::Point>> contours;

    // input format of findCountours must be CV_8U/CV_8uC1
    mask.convertTo(mask, CV_8UC1);
    cv::findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    vector<vector<cv::Point>> contours_poly;
    vector<cv::Rect> boxes; // Rect(x, y, w, h)
    double peri, area;
    for (size_t i = 0; i < contours.size(); i++)
    {
        peri = cv::arcLength(contours[i], true);
        vector<cv::Point> contour_poly;
        cv::approxPolyDP(contours[i], contour_poly, 0.02 * peri, true);

        if (contour_poly.size() > 2)
        {
            area = cv::contourArea(contours[i]);
            if (area < thresholdArea)
                continue;

            cv::Rect box = cv::boundingRect(contour_poly);
            if (box.width < thresholdWidth || box.height < thresholdHeight)
                continue;
            contours_poly.push_back(contour_poly);
            boxes.push_back(box);
        }
    }

    if (contours_poly.size() != 0)
    {
        cocoResult = CocoResult(
            -1,            // int imageId
            "abc",         // string category
            -1,            // int categoryId
            boxes,         // vector<cv::Rect> boxes
            contours_poly  //vector<vector<cv::Point>> segmentations
        );
    }

    return cocoResult;
}

pair<int, CocoResult> predictClasses(
    const cv::Mat &resmap,
    float min_area,
    float threshold,
    const cv::Size oriSize,
    bool is_get_coco_result = false)
{
    // get binary image
    cv::Mat resmap_th;
    double thr = cv::threshold(resmap, resmap_th, threshold, 1, cv::THRESH_BINARY);

    // get area + labeled image from binary resmap
    pair<cv::Mat, vector<int>> output_label = label_images(resmap_th);
    cv::Mat image_labels = output_label.first;
    vector<int> areas = output_label.second;    

    // check if image is defective
    int y_pred = is_defective(areas, min_area);

    // convert to COCO format result
    CocoResult cocoResult;
    if (is_get_coco_result)
    {
        cv::Mat image_labels_thr, mask;
        double thresh = 0;
        double maxVal = 1;        
        double thr = cv::threshold(image_labels, image_labels_thr, thresh, maxVal, cv::THRESH_BINARY);
        image_labels_thr.convertTo(image_labels_thr, CV_8UC1);
        cv::resize(image_labels_thr, mask, oriSize, 0, 0, cv::INTER_NEAREST);
        cocoResult = convertMaskToCocoFormat(mask, min_area, 0, 0);
    }

    pair<int, CocoResult> outputPair = make_pair(y_pred, cocoResult);
    return outputPair;
}

pair<cv::Scalar, cv::Mat> resmaps_l2(const cv::Mat &im1, const cv::Mat &im2)
{
    cv::Mat subMat = im1 - im2;
    cv::Mat resmap = subMat.mul(subMat); // resmap = (im1 - im2) **2

    // score = list(np.sqrt(np.sum(resmaps, axis=0)).flatten())
    cv::Scalar score = cv::Scalar(0);
    return make_pair(score, resmap);
}

pair<cv::Scalar, cv::Mat> resmaps_ssim(const cv::Mat &im1_gray, const cv::Mat &im2_gray)
{
    pair<cv::Scalar, cv::Mat> similar_output = structural_similarity(im1_gray, im2_gray, 1);
    cv::Scalar score = similar_output.first;
    cv::Mat resmap = similar_output.second;

    resmap = 1 - resmap;
    cv::Mat mask, dst;
    cv::inRange(resmap, cv::Scalar(-1), cv::Scalar(1), mask);
    resmap.copyTo(dst, mask);
    return make_pair(score, dst);
}

cv::Mat calculate_resmap(const cv::Mat &im1, const cv::Mat &im2,
                         bool use_gray_image = false, string loss_method = "l2")
{
    // convert rgb image to bgr image
    cv::Mat im1_gray, im2_gray;
    if (!use_gray_image)
    {
        cv::cvtColor(im1, im1_gray, cv::COLOR_RGB2GRAY);
        cv::cvtColor(im2, im2_gray, cv::COLOR_RGB2GRAY);
    }
    else
    {
        im1_gray = im1;
        im2_gray = im2;
    }

    pair<cv::Scalar, cv::Mat> output;
    if (loss_method == "l2")
    {
        output = resmaps_l2(im1_gray, im2_gray);
    }
    else
    {
        output = resmaps_ssim(im1_gray, im2_gray);
    }

    cv::Scalar score = output.first;
    cv::Mat resmap = output.second;

    return resmap;
}

pair<int, CocoResult> postprocess(
    const ov::Tensor output,
    const cv::Mat &inputTensor,
    cv::Size oriSize,
    float min_area_ratio = 1.22e-05,
    float threshold = 0.082,
    bool use_gray_image = false,
    string loss_method = "l2",
    bool is_get_heatmap = false,
    bool is_get_coco_result = false)
{
    // Convert tensor output to cv::Mat output
    ov::Shape tensorShape = output.get_shape();
    static const ov::Layout layout("NHWC");
    const size_t width = tensorShape[ov::layout::width_idx(layout)];
    const size_t height = tensorShape[ov::layout::height_idx(layout)];
    const size_t channels = tensorShape[ov::layout::channels_idx(layout)];
    const float *data = output.data<float>(); // NHWC
    int output_type = channels == 3 ? CV_32FC3 : CV_32FC1;
    cv::Mat predictions(height, width, output_type, output.data<float>());

    // cal resmap
    cv::Mat resmap = calculate_resmap(inputTensor, predictions, use_gray_image, loss_method);

    float min_area = min_area_ratio * INPUT_HEIGHT * INPUT_WIDTH;
    cout << "Config: " << endl;
    cout << "- min area: " << min_area << endl;
    cout << "- threshold: " << threshold << endl;

    // prediction class
    pair<int, CocoResult> pairOutput = predictClasses(resmap, min_area, threshold, oriSize, is_get_coco_result);

    if (is_get_heatmap)
    {
        cv::Mat heatmap;

        cv::resize(resmap, heatmap, oriSize, 0, 0, cv::INTER_NEAREST);
        heatmap.convertTo(heatmap, CV_8UC1);

        // save heatmap : viridis color
        cv::Mat colorMat;
        cv::applyColorMap(heatmap, colorMat, cv::COLORMAP_JET);
        cv::imwrite("heatmap.jpg", colorMat);
    }

    return pairOutput;
}

void visualize(const cv::Mat &oriImg, CocoResult result)
{
    cv::Mat showImg = oriImg.clone();
    vector<vector<cv::Point>> segmentations;
    vector<cv::Rect> boxes;
    string category;
    cv::Mat drawImg;
    cv::Scalar color;

    segmentations = result.segmentations;
    boxes = result.boxes;
    category = result.category;
    color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
    cout << "Visualize inputs: " << endl;
    cout << "- Num of segments: " << segmentations.size() << endl;
    cout << "- Num of bboxes: " << boxes.size() << endl;    

    for (int i = 0; i < boxes.size(); i++)
    {
        drawImg = cv::Mat(oriImg.size(), CV_8UC3, CV_RGB(0, 0, 0));
        cv::fillPoly(drawImg, segmentations[i], color);
        cv::addWeighted(showImg, 0.8, drawImg, 0.2, 0, showImg);
        cv::rectangle(showImg, boxes[i].tl(), boxes[i].br(), color, 1);        
    }

    cv::Mat image_bin = cv::Mat::zeros(oriImg.size().height, oriImg.size().width, CV_8UC1);
    cv::drawContours(image_bin, segmentations, -1, cv::Scalar(255), 3);

    cv::imwrite("../../examples/output_openvino_cpp/mask.jpg", image_bin);
    cv::imwrite("../../examples/output_openvino_cpp/segmented.jpg", showImg);
    cout << "Save mask path: examples/output_openvino_cpp/mask.jpg" << endl;
    cout << "Save segmented path:  examples/output_openvino_cpp/segmented.jpg" << endl;
}

int main()
{
    // configs
    cout << "Start ....." << endl;
    std::string modeFileName = "../../weights/openvino/model_20221222_9229_664304.xml";
    std::string img_path = "../../examples/inputs/demo.png";
    std::string out_img_path = "../../demo_cpp.png";
    const float threshold = 0.065;
    const float min_area_ratio = 2.44e-05;
    const bool use_gray_image = false;
    const string loss_method = "l2"; // "ssim" / "mmsim"
    const bool is_get_heatmap = true;
    const bool is_get_coco_result = true;
    using timer = std::chrono::high_resolution_clock;

    // prepare model
    ov::CompiledModel compiledModel = prepareModel(modeFileName);    
    ov::InferRequest req = compiledModel.create_infer_request();    
    // prepare cv::Mat input
    cv::Mat img = cv::imread(img_path);
    cv::Mat processedImg = preprocess(img, INPUT_WIDTH, INPUT_HEIGHT, use_gray_image);

    // cv::Mat to ov::Tensor
    float *input_data = (float *)processedImg.data;
    ov::Tensor input_tensor = ov::Tensor(compiledModel.input().get_element_type(), compiledModel.input().get_shape(), input_data);
    cout << "Input model shape: " << input_tensor.get_shape() << endl;

    timer::time_point lastTime = timer::now();

    // inference
    req.set_input_tensor(input_tensor);
    req.start_async();
    req.wait();
    ov::Tensor output = req.get_output_tensor();

    // post processing
    pair<int, CocoResult> result = postprocess(
        output,
        processedImg,
        img.size(),
        min_area_ratio,
        threshold,
        use_gray_image,
        loss_method,
        is_get_heatmap,
        is_get_coco_result);

    // print result 
    int y_pred = result.first;
    CocoResult cocoResult = result.second;
    string pred_text = y_pred == 1 ? "bad" : "good";
    cout << "Result: " << pred_text << endl;

    // print time
    auto currTime = timer::now();
    auto timeInfer = (currTime - lastTime);
    cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(timeInfer).count() << "ms" << endl;

    // visualize output: mask + segment + boxes
    visualize(img, cocoResult);

    return 0;
}