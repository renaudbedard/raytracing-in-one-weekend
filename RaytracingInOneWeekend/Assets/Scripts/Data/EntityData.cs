using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using Sirenix.Utilities;
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
		[FoldoutGroup("$EntityTitle")] [SerializeField] string name = null;

		[FoldoutGroup("$EntityTitle")] [SerializeField] [LabelWidth(35)] EntityType type;
		[FoldoutGroup("$EntityTitle")] [SerializeField] [LabelWidth(49)] bool enabled = true;

		[FoldoutGroup("$EntityTitle")] [SerializeField] Vector3 position;
		[FoldoutGroup("$EntityTitle")] [SerializeField] Vector3 rotation;

		[FoldoutGroup("$EntityTitle")] [SerializeField] bool moving;
		[FoldoutGroup("$EntityTitle")] [ShowIf(nameof(moving))] [SerializeField] Vector3 destinationOffset;
		[FoldoutGroup("$EntityTitle")] [ShowIf(nameof(moving))] [SerializeField] [MinMaxSlider(0, 1)] Vector2 timeRange;

		[FoldoutGroup("$EntityTitle")] [ShowIf(nameof(type), EntityType.Sphere)]
		[FoldoutGroup("$EntityTitle")] [SerializeField] [HideLabel] SphereData sphereData;

		[FoldoutGroup("$EntityTitle")] [ShowIf(nameof(type), EntityType.Rect)]
		[FoldoutGroup("$EntityTitle")] [SerializeField] [HideLabel] RectData rectData;

		[FoldoutGroup("$EntityTitle")] [ShowIf(nameof(type), EntityType.Box)]
		[FoldoutGroup("$EntityTitle")] [SerializeField] [HideLabel] BoxData boxData;

		[SerializeField]
#if UNITY_EDITOR
		[AssetList] [FoldoutGroup("$MaterialTitle")]
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

		[UsedImplicitly] string MaterialTitle => $"Material ({(material ? material.name : "none")})";
		[UsedImplicitly] string EntityTitle => $"{(name.IsNullOrWhitespace() ? "Entity" : name)} ({type})";
#endif

		public EntityType Type
		{
			get => type;
			set => type = value;
		}

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

		public bool Moving
		{
			get => moving;
			set => moving = value;
		}

		public Vector3 DestinationOffset
		{
			get => destinationOffset;
			set => destinationOffset = value;
		}

		public Vector2 TimeRange
		{
			get => timeRange;
			set => timeRange = value;
		}

		public SphereData SphereData
		{
			get => sphereData;
			set => sphereData = value;
		}

		public RectData RectData
		{
			get => rectData;
			set => rectData = value;
		}

		public BoxData BoxData
		{
			get => boxData;
			set => boxData = value;
		}

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