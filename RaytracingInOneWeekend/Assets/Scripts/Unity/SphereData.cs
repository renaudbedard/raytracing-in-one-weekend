using System;
using Sirenix.OdinInspector;
using UnityEngine;
#if ODIN_INSPECTOR

#else
using Title = UnityEngine.HeaderAttribute;
using OdinMock;
#endif

namespace Unity
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