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
		[HorizontalGroup("Bools")] [SerializeField] [LabelWidth(46)]
		protected bool moving = false;

		[HorizontalGroup("Bools")] [SerializeField] [LabelWidth(158)]
		bool excludeFromOverlapTest = false;

		[HideIf(nameof(moving))] [SerializeField] Vector3 center;
		[ShowIf(nameof(moving))] [SerializeField] Vector3 centerFrom, centerTo;
		[ShowIf(nameof(moving))] [SerializeField] Vector2 timeRange;

		[SerializeField] float radius;

		public SphereData(Vector3 center, float radius)
		{
			this.center = center;
			this.radius = radius;
		}

		public SphereData(Vector3 center, Vector3 offset, float t0, float t1, float radius)
		{
			centerFrom = center;
			centerTo = center + offset;
			moving = true;
			timeRange = float2(t0, t1);
			this.radius = radius;
		}

		public Vector3 CenterFrom => !moving ? center : centerFrom;
		public Vector3 CenterTo => !moving ? center : centerTo;
		public Vector3 Center(float t) => lerp(CenterFrom, CenterTo, saturate(unlerp(FromTime, ToTime, t)));
		public float FromTime => timeRange.x;
		public float ToTime => timeRange.y;
		public float MidTime => (timeRange.x + timeRange.y) / 2;
		public float Radius => radius;
		public bool ExcludeFromOverlapTest => excludeFromOverlapTest;
	}
}