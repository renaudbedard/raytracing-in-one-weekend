using System;
using UnityEngine;

namespace RaytracerInOneWeekend
{
	[Serializable]
	class RectData
	{
		[SerializeField] float distance = 0;
		[SerializeField] Vector2 center = Vector2.zero;
		[SerializeField] Vector2 size = Vector2.one;

		public float Distance
		{
			get => distance;
			set => distance = value;
		}

		public Vector2 Center
		{
			get => center;
			set => center = value;
		}

		public Vector2 Size
		{
			get => size;
			set => size = value;
		}
	}
}