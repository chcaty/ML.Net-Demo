using Microsoft.ML;
using TransferLearningTF.Models;

var assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
var imagesFolder = Path.Combine(assetsPath, "images");
var trainTagsTsv = Path.Combine(imagesFolder, "tags.tsv");
var testTagsTsv = Path.Combine(imagesFolder, "test-tags.tsv");
var predictSingleImage = Path.Combine(imagesFolder, "toaster3.jpg");
var inceptionTensorFlowModel = Path.Combine(assetsPath, "inception", "tensorflow_inception_graph.pb");

var mlContext = new MLContext();
var model = GenerateModel(mlContext);
ClassifySingleImage(mlContext, model);

// 显示实用工具
void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
{
    foreach (var prediction in imagePredictionData)
    {
        Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
    }
}

// 进行预测
void ClassifySingleImage(MLContext mlContext, ITransformer model)
{
    var imageData = new ImageData()
    {
        ImagePath = predictSingleImage
    };
    // Make prediction function (input = ImageData, output = ImagePrediction)
    var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
    var prediction = predictor.Predict(imageData);
    Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
}

// 构造 ML.NET 模型管道
ITransformer GenerateModel(MLContext mlContext)
{
    IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
        // The image transforms transform the images into the model's expected format.
        .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
        .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
        .Append(mlContext.Model.LoadTensorFlowModel(inceptionTensorFlowModel).
            ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
        .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
        .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
        .AppendCacheCheckpoint(mlContext);
    var trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path: trainTagsTsv, hasHeader: false);
    var model = pipeline.Fit(trainingData);
    var testData = mlContext.Data.LoadFromTextFile<ImageData>(path: testTagsTsv, hasHeader: false);
    var predictions = model.Transform(testData);

    // Create an IEnumerable for the predictions for displaying results
    var imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
    DisplayResults(imagePredictionData);
    var metrics =
        mlContext.MulticlassClassification.Evaluate(predictions,
            labelColumnName: "LabelKey",
            predictedLabelColumnName: "PredictedLabel");
    Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
    Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");
    return model;
}

// Inception 模型参数结构
internal struct InceptionSettings
{
    public const int ImageHeight = 224;
    public const int ImageWidth = 224;
    public const float Mean = 117;
    public const float Scale = 1;
    public const bool ChannelsLast = true;
}
