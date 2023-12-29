namespace LayerBasedNeuralNetwork
{
    public class NeuralNetwork
    {
        private readonly Layer[] _layers;

        public NeuralNetwork(params uint[] layerSizes)
        {
            // Not include input layer
            _layers = new Layer[layerSizes.Length - 1];

            for (var i = 0; i < _layers.Length; i++)
            {
                _layers[i] = new Layer(layerSizes[i], layerSizes[i + 1]);
            }
        }

        public float[] Evaluate(float[] inputs)
        {
            foreach (var t in _layers)
            {
                inputs = t.CalculateOutputs(inputs);
            }

            return inputs;
        }
    }
}