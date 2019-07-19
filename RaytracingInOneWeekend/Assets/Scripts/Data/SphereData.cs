using System;
using UnityEngine;

#if ODIN_INSPECTOR && UNITY_EDITOR
using Sirenix.OdinInspector;
using UnityEditor;
using System.Linq;
using System.Collections.Generic;
#endif

namespace RaytracerInOneWeekend
{
	[Serializable]
	class SphereData
	{
		[SerializeField] bool enabled = true;
		[SerializeField] Vector3 center;
		[SerializeField] float radius;

		[SerializeField]
#if ODIN_INSPECTOR && UNITY_EDITOR
		[ValueDropdown(nameof(GetMaterialAssets))]
#endif
		MaterialData material;

#if ODIN_INSPECTOR && UNITY_EDITOR
		[ShowInInspector]
		[InlineEditor(DrawHeader = false)]
		[ShowIf(nameof(material))]
		MaterialData MaterialData
		{
			get => material;
			set => material = value;
		}
#endif

		public SphereData(Vector3 center, float radius, MaterialData material)
		{
			this.center = center;
			this.radius = radius;
			this.material = material;
		}

		public bool Enabled => enabled;
		public Vector3 Center => center;
		public float Radius => radius;
		public MaterialData Material => material;

#if ODIN_INSPECTOR && UNITY_EDITOR
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