using Microsoft.ML;
using TaxiFarePrediction.Models;

var trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
var testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");

var mlContext = new MLContext(seed: 0);
var model = Train(mlContext, trainDataPath); 
Evaluate(mlContext, model);
TestSinglePrediction(mlContext, model);

// 加载数据。
// 提取并转换数据。
// 定型模型。
// 返回模型。
ITransformer Train(MLContext mlContext, string dataPath)
{
    var dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');
    var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
        .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
        .Append(mlContext.Regression.Trainers.FastTree());
    var model = pipeline.Fit(dataView);
    return model;
}

// 加载测试数据集。
// 创建回归计算器。
// 评估模型并创建指标。
// 显示指标。
void Evaluate(MLContext mlContext, ITransformer model)
{
    var dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(testDataPath, hasHeader: true, separatorChar: ',');
    var predictions = model.Transform(dataView);
    var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
    Console.WriteLine();
    Console.WriteLine($"*************************************************");
    Console.WriteLine($"*       Model quality metrics evaluation         ");
    Console.WriteLine($"*------------------------------------------------");
    Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
    Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
}

// 创建测试数据的单个注释。
// 根据测试数据预测费用。
// 结合测试数据和预测进行报告。
// 显示预测结果。
void TestSinglePrediction(MLContext mlContext, ITransformer model)
{
    var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);
    var taxiTripSample = new TaxiTrip()
    {
        VendorId = "VTS",
        RateCode = "1",
        PassengerCount = 1,
        TripTime = 1140,
        TripDistance = 3.75f,
        PaymentType = "CRD",
        FareAmount = 0 // To predict. Actual/Observed = 15.5
    };
    var prediction = predictionFunction.Predict(taxiTripSample);
    Console.WriteLine($"**********************************************************************");
    Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
    Console.WriteLine($"**********************************************************************");
}