using GitHubIssueClassification.Models;
using Microsoft.ML;

var appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
var trainDataPath = Path.Combine(appPath, "..", "..", "..", "Data", "issues_train.tsv");
var testDataPath = Path.Combine(appPath, "..", "..", "..", "Data", "issues_test.tsv");
var modelPath = Path.Combine(appPath, "..", "..", "..", "Models", "model.zip");

MLContext context;
PredictionEngine<GitHubIssue, IssuePrediction> predEngine;
ITransformer trainedModel;
IDataView dataView;

context = new MLContext(seed: 0);

dataView = context.Data.LoadFromTextFile<GitHubIssue>(trainDataPath, hasHeader: true);
var pipeline = ProcessData(); 
var trainingPipeline = BuildAndTrainModel(dataView, pipeline);
Evaluate(dataView.Schema);
PredictIssue();

// 提取并转换数据。
// 返回处理管道。
IEstimator<ITransformer> ProcessData()
{
    var data = context.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
        .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
        .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
        .Append(context.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
        // 对小/中型数据集使用 AppendCacheCheckpoint 可以降低训练时间。 在处理大型数据集时不使用它（删除 .AppendCacheCheckpoint()）。
        .AppendCacheCheckpoint(context);
    return data;
}

// 创建定型算法类。
// 定型模型。
// 根据定型数据预测区域。
// 返回模型。
IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
{
    var model = pipeline.Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
        .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
    trainedModel = model.Fit(trainingDataView);
    predEngine = context.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(trainedModel);
    var issue = new GitHubIssue
    {
        Title = "WebSockets communication is slow in my machine",
        Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
    };
    var prediction = predEngine.Predict(issue);
    Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");
    return model;
}

// 加载测试数据集。
// 创建多类评估程序。
// 评估模型并创建指标。
// 显示指标。
void Evaluate(DataViewSchema trainingDataViewSchema)
{
    var testDataView = context.Data.LoadFromTextFile<GitHubIssue>(testDataPath, hasHeader: true);
    var testMetrics = context.MulticlassClassification.Evaluate(trainedModel.Transform(testDataView));
    Console.WriteLine($"*************************************************************************************************************");
    Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
    Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
    Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
    Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
    Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
    Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
    Console.WriteLine($"*************************************************************************************************************");
    SaveModelAsFile(context, trainingDataViewSchema, trainedModel);
}

// 将模型保存到文件
void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
{
    mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
}

// 加载已保存的模型
// 创建测试数据的单个问题。
// 根据测试数据预测区域。
// 结合测试数据和预测进行报告。
// 显示预测结果。
void PredictIssue()
{
    var loadedModel = context.Model.Load(modelPath, out var modelInputSchema);
    var singleIssue = new GitHubIssue() { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" };
    predEngine = context.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);
    var prediction = predEngine.Predict(singleIssue);
    Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
}