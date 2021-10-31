using UnityEngine;
namespace Unity
{
	public class RandomSeedData : MonoBehaviour
	{
		[SerializeField] private uint randomSeed = 1;

		public uint RandomSeed => randomSeed;
	}
}