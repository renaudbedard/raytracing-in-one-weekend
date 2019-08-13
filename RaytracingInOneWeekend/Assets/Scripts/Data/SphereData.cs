using System;
using UnityEngine;
using static Unity.Mathematics.math;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
#else
using Title = UnityEngine.HeaderAttribute;
using OdinMock;
#endif

namespace RaytracerInOneWeekend
{
	[Serializable]
	class SphereData
	{
		[SerializeField] float radius;
		[SerializeField] [LabelWidth(178)] bool excludeFromOverlapTest = false;

		public SphereData(float radius)
		{
			this.radius = radius;
		}

		public float Radius
		{
			get => radius;
			set => radius = value;
		}

		public bool ExcludeFromOverlapTest => excludeFromOverlapTest;
	}
}