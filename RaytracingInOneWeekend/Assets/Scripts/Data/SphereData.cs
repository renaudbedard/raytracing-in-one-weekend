using System;
using UnityEngine;

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
		[SerializeField] bool enabled = true;
		[SerializeField] [LabelWidth(150)] bool excludeFromOverlapTest = false;
		[SerializeField] Vector3 center;
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

		public bool Enabled => enabled;
		public bool ExcludeFromOverlapTest => excludeFromOverlapTest;
		public Vector3 Center => center;
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