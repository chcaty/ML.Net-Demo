using Microsoft.ML;
using ProductSalesAnomalyDetection.Models;

var dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "product-sales.csv");
//assign the Number of records in dataset file to constant variable
const int docsize = 36;

var mlContext = new MLContext();
var dataView = mlContext.Data.LoadFromTextFile<ProductSalesData>(path: dataPath, hasHeader: true, separatorChar: ',');
DetectSpike(mlContext, docsize, dataView);

IDataView CreateEmptyDataView(MLContext mlContext)
{
    // Create empty DataView. We just need the schema to call Fit() for the time series transforms
    IEnumerable<ProductSalesData> enumerableData = new List<ProductSalesData>();
    return mlContext.Data.LoadFromEnumerable(enumerableData);
}

void DetectSpike(MLContext mlContext, int docSize, IDataView productSales)
{
    var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(ProductSalesPrediction.Prediction), inputColumnName: nameof(ProductSalesData.numSales), confidence: 95, pvalueHistoryLength: docSize / 4);
    ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView(mlContext));
    var transformedData = iidSpikeTransform.Transform(productSales);
    var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);
    Console.WriteLine("Alert\tScore\tP-Value");
    foreach (var p in predictions)
    {
        var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";

        if (p.Prediction[0] == 1)
        {
            results += " <-- Spike detected";
        }

        Console.WriteLine(results);
    }
    Console.WriteLine("");
}