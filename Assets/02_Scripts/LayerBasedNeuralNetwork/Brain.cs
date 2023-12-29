using UnityEngine;

namespace LayerBasedNeuralNetwork
{
    public class Brain : MonoBehaviour
    {
        [SerializeField] private uint[] layerSizes;

        private NeuralNetwork _neuralNetwork;
        private void Awake()
        {
            _neuralNetwork = new NeuralNetwork(layerSizes);
        }

        public float[] Evaluate(float[] inputs)
        {
            return _neuralNetwork.Evaluate(inputs);
        }
    }
}