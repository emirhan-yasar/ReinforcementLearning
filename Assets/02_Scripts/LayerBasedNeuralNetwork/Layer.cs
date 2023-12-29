namespace LayerBasedNeuralNetwork
{
    public class Layer
    {
        private readonly uint _numNeuronsIn;
        private readonly uint _numNeuronsOut;
        private readonly float[,] _weights;
        private readonly float[] _biases;

        public Layer(uint numNeuronsIn, uint numNeuronsOut)
        {
            _numNeuronsIn = numNeuronsIn;
            _numNeuronsOut = numNeuronsOut;

            _weights = new float[numNeuronsOut,numNeuronsIn]; // numOfRows = numNeuronsOut, numOfColumns = numNeuronsIn
            _biases = new float[numNeuronsOut];
        }

        public float[] CalculateOutputs(float[] inputs)
        {
            var layerOutputs = new float[_numNeuronsOut];

            for (var row = 0; row < _numNeuronsOut; row++)
            {
                var output = _biases[row];
                for (var column = 0; column < _numNeuronsIn; column++)
                {
                    output += _weights[row, column] * inputs[column];
                    output = ActivationFunctions.Sigmoid(output);
                }
                layerOutputs[row] = output;
            }

            return layerOutputs;
        }

    }
}