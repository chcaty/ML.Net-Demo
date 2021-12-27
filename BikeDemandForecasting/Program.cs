using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using System.Data.SqlClient;
using BikeDemandForecasting.Models;

var rootDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "../../../"));
var dbFilePath = Path.Combine(rootDir, "Data", "DailyDemand.mdf");
var modelPath = Path.Combine(rootDir, "MLModel.zip");

var connectionString = $"Data Source=(LocalDB)\\MSSQLLocalDB;AttachDbFilename={dbFilePath};Integrated Security=True;Connect Timeout=30;";
const string query = "SELECT RentalDate, CAST(Year as REAL) as Year, CAST(TotalRentals as REAL) as TotalRentals FROM Rentals";
var dbSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, query);

var mlContext = new MLContext();
var loader = mlContext.Data.CreateDatabaseLoader<ModelInput>();
var dataView = loader.Load(dbSource);

var firstYearData = mlContext.Data.FilterRowsByColumn(dataView, "Year", upperBound: 1);
var secondYearData = mlContext.Data.FilterRowsByColumn(dataView, "Year", lowerBound: 1);

var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
    outputColumnName: "ForecastedRentals",
    inputColumnName: "TotalRentals",
    windowSize: 7,
    seriesLength: 30,
    trainSize: 365,
    horizon: 7,
    confidenceLevel: 0.95f,
    confidenceLowerBoundColumn: "LowerBoundRentals",
    confidenceUpperBoundColumn: "UpperBoundRentals");

var forecaster = forecastingPipeline.Fit(firstYearData);

Evaluate(secondYearData, forecaster, mlContext);

var forecastEngine = forecaster.CreateTimeSeriesEngine<ModelInput, ModelOutput>(mlContext);
forecastEngine.CheckPoint(mlContext, modelPath);

Forecast(secondYearData, 7, forecastEngine, mlContext);

// 评估模型
void Evaluate(IDataView testData, ITransformer model, MLContext mlContext)
{
    // Make predictions
    var predictions = model.Transform(testData);

    // Actual values
    var actual =
        mlContext.Data.CreateEnumerable<ModelInput>(testData, true)
            .Select(observed => observed.TotalRentals);

    // Predicted values
    var forecast =
        mlContext.Data.CreateEnumerable<ModelOutput>(predictions, true)
            .Select(prediction => prediction.ForecastedRentals[0]);

    // Calculate error (actual - forecast)
    var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);

    // Get metric averages
    var MAE = metrics.Average(error => Math.Abs(error)); // Mean Absolute Error
    var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2))); // Root Mean Squared Error

    // Output metrics
    Console.WriteLine("Evaluation Metrics");
    Console.WriteLine("---------------------");
    Console.WriteLine($"Mean Absolute Error: {MAE:F3}");
    Console.WriteLine($"Root Mean Squared Error: {RMSE:F3}\n");
}

// 使用模型预测需求
void Forecast(IDataView testData, int horizon, TimeSeriesPredictionEngine<ModelInput, ModelOutput> forecaster, MLContext mlContext)
{

    var forecast = forecaster.Predict();

    var forecastOutput =
        mlContext.Data.CreateEnumerable<ModelInput>(testData, reuseRowObject: false)
            .Take(horizon)
            .Select((rental, index) =>
            {
                var rentalDate = rental.RentalDate.ToShortDateString();
                var actualRentals = rental.TotalRentals;
                var lowerEstimate = Math.Max(0, forecast.LowerBoundRentals[index]);
                var estimate = forecast.ForecastedRentals[index];
                var upperEstimate = forecast.UpperBoundRentals[index];
                return $"Date: {rentalDate}\n" +
                $"Actual Rentals: {actualRentals}\n" +
                $"Lower Estimate: {lowerEstimate}\n" +
                $"Forecast: {estimate}\n" +
                $"Upper Estimate: {upperEstimate}\n";
            });

    // Output predictions
    Console.WriteLine("Rental Forecast");
    Console.WriteLine("---------------------");
    foreach (var prediction in forecastOutput)
    {
        Console.WriteLine(prediction);
    }
}
