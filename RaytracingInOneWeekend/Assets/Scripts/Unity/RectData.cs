using System;
using Unity.Mathematics;
using UnityEngine;

namespace Unity
{
	[Serializable]
	class RectData
	{
		[SerializeField] Vector2 size = Vector2.one;

		public RectData(float2 size)
		{
			this.size = size;
		}

		public Vector2 Size
		{
			get => size;
			set => size = value;
		}
	}
}