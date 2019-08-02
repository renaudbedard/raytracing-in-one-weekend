using System;
using UnityEngine;
using static Unity.Mathematics.math;

#if UNITY_EDITOR
using UnityEditor;
using System.Linq;
using System.Collections.Generic;
#endif

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
		[HorizontalGroup("Bools")]
		[SerializeField]
		[LabelWidth(48)]
		bool enabled = true;

		[HorizontalGroup("Bools")]
		[SerializeField]
		[LabelWidth(44)]
		bool moving = false;

		[HorizontalGroup("Bools")]
		[SerializeField]
		[LabelWidth(147)]
		bool excludeFromOverlapTest = false;

		[HideIf(nameof(moving))]
		[SerializeField]
		Vector3 center;

		[ShowIf(nameof(moving))] [SerializeField] Vector3 centerFrom, centerTo;
		[ShowIf(nameof(moving))] [SerializeField] Vector2 timeRange;

		[SerializeField] float radius;

		[SerializeField]
#if UNITY_EDITOR
		[ValueDropdown(nameof(GetMaterialAssets))]
#endif
		MaterialData material;

#if UNITY_EDITOR
		[ShowInInspector]
		[InlineEditor(DrawHeader = false)]
		[ShowIf(nameof(material))]
		MaterialData MaterialData
		{
			get => material;
			set => material = value;
		}
#endif

		public SphereData() { }

		public SphereData(Vector3 center, float radius, MaterialData material)
		{
			this.center = center;
			this.radius = radius;
			this.material = material;
		}

		public SphereData(Vector3 center, Vector3 offset, float t0, float t1, float radius, MaterialData material)
		{
			centerFrom = center;
			centerTo = center + offset;
			moving = true;
			timeRange = float2(t0, t1);
			this.radius = radius;
			this.material = material;
		}

		public bool Enabled => enabled;
		public bool ExcludeFromOverlapTest => excludeFromOverlapTest;
		public bool Moving => moving;
		public Vector3 CenterFrom => !moving ? center : centerFrom;
		public Vector3 CenterTo => !moving ? center : centerTo;
		public Vector3 Center(float t) => lerp(CenterFrom, CenterTo, saturate(unlerp(FromTime, ToTime, t)));
		public float FromTime => timeRange.x;
		public float ToTime => timeRange.y;
		public float MidTime => (timeRange.x + timeRange.y) / 2;
		public float Radius => radius;

		public MaterialData Material
		{
			get => material;
			set => material = value;
		}

#if UNITY_EDITOR
		IEnumerable<ValueDropdownItem<MaterialData>> GetMaterialAssets => AssetDatabase.FindAssets("t:MaterialData")
			.Select(AssetDatabase.GUIDToAssetPath)
			.Select(AssetDatabase.LoadAssetAtPath<MaterialData>)
			.Select(asset => new ValueDropdownItem<MaterialData>(asset.name, asset))
			.Concat(new[] { new ValueDropdownItem<MaterialData>("Null", null) })
			.OrderBy(x => x.Value != null).ThenBy(x => x.Text);
#endif

#if UNITY_EDITOR
		public bool Dirty => Material && Material.Dirty;

		public void ClearDirty()
		{
			if (Material) Material.ClearDirty();
		}
#endif
	}
}