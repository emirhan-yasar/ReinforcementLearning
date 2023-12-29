using UnityEngine;

namespace LayerBasedNeuralNetwork.Network
{
    public class Layer
    {
        private readonly uint _numNeuronsIn;
        private readonly uint _numNeuronsOut;
        private readonly float[,] _weights;
        private readonly float[] _biases;

        public Layer(uint numNeuronsIn, uint numNeuronsOut, ComputeShader layerComputeShader)
        {
            _numNeuronsIn = numNeuronsIn;
            _numNeuronsOut = numNeuronsOut;

            _weights = new float[numNeuronsOut,numNeuronsIn]; // numOfRows = numNeuronsOut, numOfColumns = numNeuronsIn
            _biases = new float[numNeuronsOut];
            RandomlyInitializeLayer();
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

        private void RandomlyInitializeLayer()
        {
            for (var row = 0; row < _weights.GetLength(0); row++)
            for (var column = 0; column < _weights.GetLength(1); column++)
            {
                _weights[row, column] = Random.Range(-2f, 2f);
            }

            for (var i = 0; i < _biases.Length; i++)
            {
                _biases[i] = Random.Range(-2f, 2f);
            }
        }

    }
}