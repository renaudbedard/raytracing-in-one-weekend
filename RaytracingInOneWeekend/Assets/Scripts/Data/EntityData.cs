using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using Unity.Mathematics;
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
		[HorizontalGroup("FirstRow")] [SerializeField] [LabelWidth(35)] EntityType type;
		[HorizontalGroup("FirstRow")] [SerializeField] [LabelWidth(49)] bool enabled = true;

		[SerializeField] Vector3 position;
		[SerializeField] Vector3 rotation;

		[ShowIf(nameof(type), EntityType.Sphere)]
		[SerializeField] [HideLabel] SphereData sphereData;

		[ShowIf(nameof(type), EntityType.Rect)]
		[SerializeField] [HideLabel] RectData rectData;

		[ShowIf(nameof(type), EntityType.Box)]
		[SerializeField] [HideLabel] BoxData boxData;

		[SerializeField]
#if UNITY_EDITOR
		[AssetList]
		[FoldoutGroup("$MaterialTitle")]
#endif
		MaterialData material;

#if UNITY_EDITOR
		[ShowInInspector]
		[InlineEditor(DrawHeader = false, ObjectFieldMode = InlineEditorObjectFieldModes.Hidden)]
		[ShowIf(nameof(material))]
		[FoldoutGroup("$MaterialTitle")]
		MaterialData MaterialData
		{
			get => material;
			set => material = value;
		}

		public bool Selected { get; set; }

		[UsedImplicitly] string MaterialTitle => $"Material ({(material ? material.name : null)})";
#endif

		public static EntityData Sphere(float3 position, float radius, MaterialData m)
		{
			return new EntityData
			{
				type = EntityType.Sphere,
				position = position,
				sphereData = new SphereData(radius),
				material = m
			};
		}

		public static EntityData Rect(float3 position, quaternion rotation, float2 size, MaterialData m)
		{
			return new EntityData
			{
				type = EntityType.Rect,
				rectData = new RectData(size),
				position = position,
				Rotation = rotation,
				material = m
			};
		}

		public static EntityData Box(float3 position, quaternion rotation, float3 size, MaterialData m)
		{
			return new EntityData
			{
				type = EntityType.Box,
				boxData = new BoxData(size),
				position = position,
				Rotation = rotation,
				material = m
			};
		}

		public EntityType Type => type;

		public bool Enabled
		{
			get => enabled;
			set => enabled = value;
		}

		public Vector3 Position
		{
			get => position;
			set => position = value;
		}

		public Quaternion Rotation
		{
			get => Quaternion.Euler(rotation);
			set => rotation = value.eulerAngles;
		}

		public MaterialData Material
		{
			get => material;
			set => material = value;
		}

		public Vector3 Size
		{
			get
			{
				switch (type)
				{
					case EntityType.Sphere: return sphereData.Radius * 2 * Vector3.one;
					case EntityType.Rect: return rectData.Size;
					case EntityType.Box: return boxData.Size;
				}
				return default;
			}
		}

		public SphereData SphereData => sphereData;
		public RectData RectData => rectData;
		public BoxData BoxData => boxData;

#if UNITY_EDITOR
		bool dirty = false;
		public bool Dirty => (Material && Material.Dirty) || dirty;

		public void ClearDirty()
		{
			if (Material) Material.ClearDirty();
			dirty = false;
		}

		public void MarkDirty()
		{
			dirty = true;
		}
#endif
	}
}