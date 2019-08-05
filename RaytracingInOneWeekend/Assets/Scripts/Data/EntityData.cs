using System;
using System.Collections.Generic;
using UnityEngine;
using static Unity.Mathematics.math;

#if UNITY_EDITOR
using System.Linq;
using UnityEditor;
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
	class EntityData
	{
		[HorizontalGroup("FirstRow")] [SerializeField] [LabelWidth(35)]
		EntityType type;

		[HorizontalGroup("FirstRow")] [SerializeField] [LabelWidth(49)]
		bool enabled = true;

		[ShowIf(nameof(type), EntityType.Sphere)]
		[SerializeField]
		SphereData sphereData;

		[ShowIf(nameof(type), EntityType.Rect)]
		[SerializeField]
		RectData rectData;

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

		protected IEnumerable<ValueDropdownItem<MaterialData>> GetMaterialAssets => AssetDatabase
			.FindAssets("t:MaterialData")
			.Select(AssetDatabase.GUIDToAssetPath)
			.Select(AssetDatabase.LoadAssetAtPath<MaterialData>)
			.Select(asset => new ValueDropdownItem<MaterialData>(asset.name, asset))
			.Concat(new[] { new ValueDropdownItem<MaterialData>("Null", null) })
			.OrderBy(x => x.Value != null).ThenBy(x => x.Text);
#endif

		public static EntityData Sphere(SphereData s, MaterialData m)
		{
			return new EntityData
			{
				type = EntityType.Sphere,
				sphereData = s,
				material = m
			};
		}

		public static EntityData Rect(RectData r, MaterialData m)
		{
			return new EntityData
			{
				type = EntityType.Rect,
				rectData = r,
				material = m
			};
		}

		public EntityType Type => type;
		public bool Enabled => enabled;

		public MaterialData Material
		{
			get => material;
			set => material = value;
		}

		public Vector3 Center
		{
			get
			{
				switch (type)
				{
					case EntityType.Sphere: return sphereData.Center(sphereData.MidTime);
					case EntityType.Rect: return float3(rectData.Center, rectData.Distance);
				}
				return default;
			}
		}

		public SphereData SphereData => sphereData;
		public RectData RectData => rectData;

#if UNITY_EDITOR
		public bool Dirty => Material && Material.Dirty;

		public void ClearDirty()
		{
			if (Material) Material.ClearDirty();
		}
#endif
	}
}