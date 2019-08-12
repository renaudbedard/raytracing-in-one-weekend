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

		public Vector3 CenterFrom
		{
			get => !moving ? center : centerFrom;
			set => centerFrom = value;
		}

		public Vector3 CenterTo
		{
			get => !moving ? center : centerTo;
			set => centerTo = value;
		}

		public Vector3 CenterAt(float t) =>
			!moving ? center : (Vector3) lerp(centerFrom, centerTo, saturate(unlerp(FromTime, ToTime, t)));

		public Vector3 Center
		{
			get => center;
			set => center = value;
		}

		public float Radius
		{
			get => radius;
			set => radius = value;
		}

		public float FromTime => timeRange.x;
		public float ToTime => timeRange.y;
		public float MidTime => (timeRange.x + timeRange.y) / 2;

		public bool ExcludeFromOverlapTest => excludeFromOverlapTest;
	}
}