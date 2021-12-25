using IrisFlowerClustering.Models;
using Microsoft.ML;

var _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
var _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");

var mlContext = new MLContext(seed: 0); 
var dataView = mlContext.Data.LoadFromTextFile<IrisData>(_dataPath, hasHeader: false, separatorChar: ',');

// 创建学习管道
var featuresColumnName = "Features";
var pipeline = mlContext.Transforms
    .Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
    .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));

// 定型模型
var model = pipeline.Fit(dataView);

// 保存模型
using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
{
    mlContext.Model.Save(model, dataView.Schema, fileStream);
}

var predictor = mlContext.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model);
var prediction = predictor.Predict(TestIrisData.Setosa);
Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");