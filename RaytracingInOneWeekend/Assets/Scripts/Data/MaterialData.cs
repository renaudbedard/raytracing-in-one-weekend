using System;
using Unity.Mathematics;
using UnityEngine;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
#endif

namespace RaytracerInOneWeekend
{
	[CreateAssetMenu]
	class MaterialData : ScriptableObject, IEquatable<MaterialData>
	{
		[SerializeField] MaterialType type = MaterialType.None;
		[SerializeField] Color albedo = Color.white;

#if ODIN_INSPECTOR
		[ShowIf(nameof(Type), MaterialType.Metal)]
#endif
		[Range(0, 1)] [SerializeField] float fuzz = 0;

#if ODIN_INSPECTOR
		[ShowIf(nameof(Type), MaterialType.Dielectric)]
#endif
		[Range(1, 2.65f)] [SerializeField] float refractiveIndex = 1;

		public MaterialType Type => type;
		public Color Albedo => albedo;
		public float Fuzz => fuzz;
		public float RefractiveIndex => refractiveIndex;

		public static MaterialData Lambertian(float3 albedo)
		{
			var data = CreateInstance<MaterialData>();
			data.hideFlags = HideFlags.HideAndDontSave;
			data.type = MaterialType.Lambertian;
			data.albedo = new Color(albedo.x, albedo.y, albedo.z);
			return data;
		}

		public static MaterialData Metal(float3 albedo, float fuzz = 0)
		{
			var data = CreateInstance<MaterialData>();
			data.hideFlags = HideFlags.HideAndDontSave;
			data.type = MaterialType.Metal;
			data.albedo = new Color(albedo.x, albedo.y, albedo.z);
			data.fuzz = fuzz;
			return data;
		}

		public static MaterialData Dielectric(float refractiveIndex)
		{
			var data = CreateInstance<MaterialData>();
			data.hideFlags = HideFlags.HideAndDontSave;
			data.type = MaterialType.Dielectric;
			data.refractiveIndex = refractiveIndex;
			return data;
		}

#if UNITY_EDITOR
		public bool Dirty { get; private set; }

		public void ClearDirty()
		{
			Dirty = false;
		}

		void OnValidate()
		{
			Dirty = true;
		}
#endif

		public bool Equals(MaterialData other)
		{
			if (ReferenceEquals(null, other)) return false;
			if (ReferenceEquals(this, other)) return true;
			return base.Equals(other) &&
				   type == other.type &&
				   albedo.Equals(other.albedo) &&
				   fuzz.Equals(other.fuzz) &&
				   refractiveIndex.Equals(other.refractiveIndex);
		}

		public override bool Equals(object obj)
		{
			if (ReferenceEquals(null, obj)) return false;
			if (ReferenceEquals(this, obj)) return true;
			if (obj.GetType() != this.GetType()) return false;
			return Equals((MaterialData) obj);
		}

		public override int GetHashCode()
		{
			unchecked
			{
				int hashCode = base.GetHashCode();
				hashCode = (hashCode * 397) ^ (int) type;
				hashCode = (hashCode * 397) ^ albedo.GetHashCode();
				hashCode = (hashCode * 397) ^ fuzz.GetHashCode();
				hashCode = (hashCode * 397) ^ refractiveIndex.GetHashCode();
				return hashCode;
			}
		}

		public static bool operator ==(MaterialData left, MaterialData right)
		{
			return Equals(left, right);
		}

		public static bool operator !=(MaterialData left, MaterialData right)
		{
			return !Equals(left, right);
		}
	}
}