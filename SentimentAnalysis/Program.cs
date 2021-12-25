using Microsoft.ML;
using SentimentAnalysis.Models;
using static Microsoft.ML.DataOperationsCatalog;

var dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");

var mlContext = new MLContext();
var splitDataView = LoadData(mlContext); 
var model = BuildAndTrainModel(mlContext, splitDataView.TrainSet); 
Evaluate(mlContext, model, splitDataView.TestSet); 
UseModelWithSingleItem(mlContext, model);
UseModelWithBatchItems(mlContext, model);

// 加载数据。
// 将加载的数据集拆分为训练数据集和测试数据集。
// 返回拆分的训练数据集和测试数据集。
TrainTestData LoadData(MLContext mlContext)
{
    var dataView = mlContext.Data.LoadFromTextFile<SentimentData>(dataPath, hasHeader: false); 
    TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
    return splitDataView;
}

// 提取并转换数据。
// 定型模型。
// 根据测试数据预测情绪。
// 返回模型。
ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
{
    var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
    Console.WriteLine("=============== Create and Train the Model ===============");
    var model = estimator.Fit(splitTrainSet);
    Console.WriteLine("=============== End of training ===============");
    Console.WriteLine();
    return model;
}

// 加载测试数据集。
// 创建 BinaryClassification 计算器。
// 评估模型并创建指标。
// 显示指标。
void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
{
    Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
    var predictions = model.Transform(splitTestSet);
    var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
    Console.WriteLine();
    Console.WriteLine("Model quality metrics evaluation");
    Console.WriteLine("--------------------------------");
    Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
    Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
    Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
    Console.WriteLine("=============== End of model evaluation ===============");
}

// 创建测试数据的单个注释。
// 根据测试数据预测情绪。
// 结合测试数据和预测进行报告。
// 显示预测结果。
void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
{
    var predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
    var sampleStatement = new SentimentData
    {
        SentimentText = "This was a very bad steak"
    };
    var resultPrediction = predictionFunction.Predict(sampleStatement);

    Console.WriteLine();
    Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

    Console.WriteLine();
    Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

    Console.WriteLine("=============== End of Predictions ===============");
    Console.WriteLine();
}

// 创建批处理测试数据。
// 根据测试数据预测情绪。
// 结合测试数据和预测进行报告。
// 显示预测结果。
void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
{
    IEnumerable<SentimentData> sentiments = new[]
    {
        new SentimentData
        {
            SentimentText = "This was a horrible meal"
        },
        new SentimentData
        {
            SentimentText = "I love this spaghetti."
        }
    };

    var batchComments = mlContext.Data.LoadFromEnumerable(sentiments);
    var predictions = model.Transform(batchComments);
    // Use model to predict whether comment data is Positive (1) or Negative (0).
    var predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

    Console.WriteLine();
    Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
    foreach (var prediction in predictedResults)
    {
        Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
    }
    Console.WriteLine("=============== End of predictions ===============");
}