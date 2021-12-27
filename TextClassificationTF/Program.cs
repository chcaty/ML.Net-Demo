using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

string _modelPath = Path.Combine(Environment.CurrentDirectory, "sentiment_model");
MLContext mlContext = new MLContext();
var lookupMap = mlContext.Data.LoadFromTextFile(Path.Combine(_modelPath, "imdb_word_index.csv"),
    columns: new[]
    {
        new TextLoader.Column("Words", DataKind.String, 0),
        new TextLoader.Column("Ids", DataKind.Int32, 1),
    },
    separatorChar: ','
);

void ResizeFeaturesAction(VariableLength s, FixedLength f)
{
    var features = s.VariableLengthFeatures;
    Array.Resize(ref features, Config.FeatureLength);
    f.Features = features;
}

TensorFlowModel tensorFlowModel = mlContext.Model.LoadTensorFlowModel(_modelPath);
DataViewSchema schema = tensorFlowModel.GetModelSchema();
Console.WriteLine(" =============== TensorFlow Model Schema =============== ");
var featuresType = (VectorDataViewType)schema["Features"].Type;
Console.WriteLine($"Name: Features, Type: {featuresType.ItemType.RawType}, Size: ({featuresType.Dimensions[0]})");
var predictionType = (VectorDataViewType)schema["Prediction/Softmax"].Type;
Console.WriteLine($"Name: Prediction/Softmax, Type: {predictionType.ItemType.RawType}, Size: ({predictionType.Dimensions[0]})");

IEstimator<ITransformer> pipeline =
    // Split the text into individual words
    mlContext.Transforms.Text.TokenizeIntoWords("TokenizedWords", "ReviewText")
    // Map each word to an integer value. The array of integer makes up the input features.
        .Append(mlContext.Transforms.Conversion.MapValue("VariableLengthFeatures", lookupMap,
            lookupMap.Schema["Words"], lookupMap.Schema["Ids"], "TokenizedWords"))
    // Resize variable length vector to fixed length vector.
        .Append(mlContext.Transforms.CustomMapping((Action<VariableLength, FixedLength>)ResizeFeaturesAction, "Resize"))
    // Retrieves the 'Prediction' from TensorFlow and copies to a column
        .Append(mlContext.Transforms.CopyColumns("Prediction", "Prediction/Softmax"));

// Create an executable model from the estimator pipeline
IDataView dataView = mlContext.Data.LoadFromEnumerable(new List<MovieReview>());
ITransformer model = pipeline.Fit(dataView);
PredictSentiment(mlContext, model);

void PredictSentiment(MLContext mlContext, ITransformer model)
{
    var engine = mlContext.Model.CreatePredictionEngine<MovieReview, MovieReviewSentimentPrediction>(model);
    var review = new MovieReview()
    {
        ReviewText = "this film is really good"
    };
    var sentimentPrediction = engine.Predict(review);
    Console.WriteLine("Number of classes: {0}", sentimentPrediction.Prediction.Length);
    Console.WriteLine("Is sentiment/review positive? {0}", sentimentPrediction.Prediction[1] > 0.5 ? "Yes." : "No.");
}

/// <summary>
/// Class to hold original sentiment data.
/// </summary>
public class MovieReview
{
    public string ReviewText { get; set; }
}

/// <summary>
/// Class to hold the variable length feature vector. Used to define the
/// column names used as input to the custom mapping action.
/// </summary>
public class VariableLength
{
    /// <summary>
    /// This is a variable length vector designated by VectorType attribute.
    /// Variable length vectors are produced by applying operations such as 'TokenizeWords' on strings
    /// resulting in vectors of tokens of variable lengths.
    /// </summary>
    [VectorType]
    public int[] VariableLengthFeatures { get; set; }
}

/// <summary>
/// Class to hold the fixed length feature vector. Used to define the
/// column names used as output from the custom mapping action,
/// </summary>
public class FixedLength
{
    /// <summary>
    /// This is a fixed length vector designated by VectorType attribute.
    /// </summary>
    [VectorType(Config.FeatureLength)]
    public int[] Features { get; set; }
}

/// <summary>
/// Class to contain the output values from the transformation.
/// </summary>
public class MovieReviewSentimentPrediction
{
    [VectorType(2)]
    public float[] Prediction { get; set; }
}

static class Config
{
    public const int FeatureLength = 600;
}