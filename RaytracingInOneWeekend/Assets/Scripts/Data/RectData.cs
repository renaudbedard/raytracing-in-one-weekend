using System;
using UnityEngine;

namespace RaytracerInOneWeekend
{
	[Serializable]
	class RectData
	{
		[SerializeField] Vector2 size = Vector2.one;

		public Vector2 Size
		{
			get => size;
			set => size = value;
		}
	}
}