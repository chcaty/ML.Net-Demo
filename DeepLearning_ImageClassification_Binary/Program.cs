using DeepLearning_ImageClassification_Binary.Models;
using Microsoft.ML;
using Microsoft.ML.Vision;
using static Microsoft.ML.DataOperationsCatalog;

var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");
var assetsRelativePath = Path.Combine(projectDirectory, "assets");

var mlContext = new MLContext();

// 准备数据
var images = LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);
var imageData = mlContext.Data.LoadFromEnumerable(images);
var shuffledData = mlContext.Data.ShuffleRows(imageData);
var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
        inputColumnName: "Label",
        outputColumnName: "LabelAsKey")
    .Append(mlContext.Transforms.LoadRawImageBytes(
        outputColumnName: "Image",
        imageFolder: assetsRelativePath,
        inputColumnName: "ImagePath"));
var preProcessedData = preprocessingPipeline
    .Fit(shuffledData)
    .Transform(shuffledData);
var trainSplit = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.3);
var validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);
var trainSet = trainSplit.TrainSet;
var validationSet = validationTestSplit.TrainSet;
var testSet = validationTestSplit.TestSet;

// 定义训练管道
var classifierOptions = new ImageClassificationTrainer.Options()
{
    FeatureColumnName = "Image",
    LabelColumnName = "LabelAsKey",
    ValidationSet = validationSet,
    Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
    MetricsCallback = (metrics) => Console.WriteLine(metrics),
    TestOnTrainSet = false,
    ReuseTrainSetBottleneckCachedValues = true,
    ReuseValidationSetBottleneckCachedValues = true
};
var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
ITransformer trainedModel = trainingPipeline.Fit(trainSet);

ClassifySingleImage(mlContext, testSet, trainedModel);
ClassifyImages(mlContext, testSet, trainedModel);

// 加载数据
IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
{
    var files = Directory.GetFiles(folder, "*",
        searchOption: SearchOption.AllDirectories);
    foreach (var file in files)
    {
        if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
            continue;
        var label = Path.GetFileName(file);

        if (useFolderNameAsLabel)
            label = Directory.GetParent(file)?.Name;
        else
        {
            for (var index = 0; index < label.Length; index++)
            {
                if (char.IsLetter(label[index])) continue;
                label = label.Substring(0, index);
                break;
            }
        }
        yield return new ImageData()
        {
            ImagePath = file,
            Label = label
        };
    }
}

static void OutputPrediction(ModelOutput prediction)
{
    var imageName = Path.GetFileName(prediction.ImagePath);
    Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
}

// 对单个图像进行分类
void ClassifySingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
{
    var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);
    var image = mlContext.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: true).First();
    var prediction = predictionEngine.Predict(image);
    Console.WriteLine("Classifying single image");
    OutputPrediction(prediction);
}

void ClassifyImages(MLContext mlContext, IDataView data, ITransformer trainedModel)
{
    var predictionData = trainedModel.Transform(data);
    var predictions = mlContext.Data.CreateEnumerable<ModelOutput>(predictionData, reuseRowObject: true).Take(10);
    Console.WriteLine("Classifying multiple images");
    foreach (var prediction in predictions)
    {
        OutputPrediction(prediction);
    }
}