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

		public float Distance => distance;
		public Vector2 Center => center;
		public Vector2 Size => size;
	}
}